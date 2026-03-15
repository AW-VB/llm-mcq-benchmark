#!/usr/bin/env python3
"""Improved benchmark runner with a slightly wider generation budget and a stricter parser.

This script is a drop-in alternative to `llm_benchmark.py`. It keeps the same
dataset/task/model setup, but makes two deliberate changes based on error
analysis from the original version:

1. Default `max_new_tokens` is increased from 6 to 8.
   In the original setup, some otherwise-valid short answers from Qwen-family
   models were truncated before the final option label was emitted. A small
   increase to 8 preserves the "short-answer" regime while reducing formatting
   artifacts caused by overly aggressive truncation.

2. Answer parsing is made substantially stricter.
   The previous parser could extract stray option letters from long generated
   reasoning text. That behavior was especially misleading for models such as
   DeepSeek-R1-Distill-Qwen-1.5B, which often produced chain-of-thought-like
   text without reaching a final answer under short generation limits. The new
   parser only accepts short, explicit answer formats on the final non-empty
   line and returns `None` otherwise.

These changes intentionally do not add model-specific accommodations. The goal
is to keep one uniform benchmark setting while making the scoring logic more
faithful to concise multiple-choice answering.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path

import torch
from datasets import Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from llm_benchmark import (
    MODEL_REGISTRY,
    RunResult,
    TASK_FILTER_ALIASES,
    append_summary,
    build_prompt,
    ensure_dir,
    extract_gold_label,
    extract_option_pairs,
    extract_question,
    infer_one,
    load_model_and_tokenizer,
    load_task_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the improved Open-LLM benchmark evaluation.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["tinyllama", "qwen25_3b", "deepseek_r1_1p5b", "qwen3_4b"],
        choices=sorted(MODEL_REGISTRY.keys()),
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["commonsenseqa", "openbookqa", "piqa"],
        choices=sorted(TASK_FILTER_ALIASES.keys()),
    )
    parser.add_argument(
        "--prompt-styles",
        nargs="+",
        default=["baseline", "constrained", "fewshot"],
        choices=["baseline", "constrained", "fewshot"],
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=None, help="Optional subset size for debugging.")
    parser.add_argument("--output-dir", type=str, default="results_improved")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=8,
        help="Short generation budget for MCQ answering. Default is 8.",
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help="Passed to from_pretrained, e.g. auto / cuda:0 / cpu.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Enable only if a model requires it.",
    )
    parser.add_argument(
        "--use-fast-tokenizer",
        action="store_true",
        help="Try fast tokenizer if available.",
    )
    return parser.parse_args()


def parse_predicted_label(text: str, valid_labels: set[str]) -> str | None:
    """
    Strict parser for short-generation MCQ benchmarking.

    Design goals:
    - High precision over high recall
    - Only accept explicit final-answer formats
    - Focus on the tail of the output
    - Never guess from long reasoning text
    """
    if not text or not valid_labels:
        return None

    cleaned = text.strip().upper()
    if not cleaned:
        return None

    labels = sorted(valid_labels)
    label_group = "|".join(re.escape(label) for label in labels)

    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    candidates = []

    if lines:
        # First try the last non-empty line.
        candidates.append(lines[-1])

        # Then try the last two non-empty lines joined together.
        if len(lines) >= 2:
            candidates.append(" ".join(lines[-2:]))
    else:
        candidates.append(cleaned)

    patterns = [
        # A / (A) / A. / **A** / **A
        rf"^\*{{0,2}}\(?({label_group})\)?\*{{0,2}}[\s\.,:;!\?-]*$",

        # Answer: A / Answer is A / Answer is: **A**
        rf"^ANSWER(?:\s+IS)?\s*[:：]?\s*\*{{0,2}}\(?({label_group})\)?\*{{0,2}}[\s\.,:;!\?-]*$",

        # Final answer: A / Final answer is A / Final answer is: **A**
        rf"^FINAL\s+ANSWER(?:\s+IS)?\s*[:：]?\s*\*{{0,2}}\(?({label_group})\)?\*{{0,2}}[\s\.,:;!\?-]*$",

        # The correct answer is A / The correct answer is: **A**
        # The most appropriate answer is A / The most appropriate answer is: **A**
        rf"^.*ANSWER(?:\s+IS)?\s*[:：]?\s*\*{{0,2}}\(?({label_group})\)?\*{{0,2}}[\s\.,:;!\?-]*$",
    ]

    for candidate in candidates:
        candidate = candidate.strip()
        if len(candidate) > 40:
            continue

        for pattern in patterns:
            match = re.match(pattern, candidate)
            if match:
                return match.group(1)

    return None


def evaluate_dataset(
    model_key: str,
    model_name: str,
    task_key: str,
    split: str,
    dataset: Dataset,
    prompt_style: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    output_dir: Path,
    max_new_tokens: int,
    seed: int,
    limit: int | None,
) -> RunResult:
    predictions_path = output_dir / f"predictions__{model_key}__{task_key}__{prompt_style}.jsonl"
    correct = 0
    total = len(dataset)
    used_chat_template_any = False

    start = time.time()
    with predictions_path.open("w", encoding="utf-8") as f:
        for idx, sample in enumerate(tqdm(dataset, desc=f"{model_key} | {task_key} | {prompt_style}")):
            option_pairs = extract_option_pairs(sample)
            question = extract_question(sample)
            gold = extract_gold_label(sample, option_pairs)
            valid_labels = {label.upper() for label, _ in option_pairs}

            prompt = build_prompt(question, option_pairs, prompt_style)
            raw_output, used_chat_template = infer_one(
                model_key=model_key,
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
            )
            used_chat_template_any = used_chat_template_any or used_chat_template
            pred = parse_predicted_label(raw_output, valid_labels)
            is_correct = pred == gold
            correct += int(is_correct)

            row = {
                "index": idx,
                "question": question,
                "gold": gold,
                "prediction": pred,
                "is_correct": is_correct,
                "raw_output": raw_output,
                "options": [{"label": label, "text": text} for label, text in option_pairs],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    elapsed = time.time() - start
    accuracy = correct / total if total else 0.0
    return RunResult(
        model_key=model_key,
        model_name=model_name,
        task_key=task_key,
        prompt_style=prompt_style,
        split=split,
        n_samples=total,
        n_correct=correct,
        accuracy=accuracy,
        elapsed_sec=elapsed,
        avg_sec_per_sample=(elapsed / total if total else 0.0),
        generation_max_new_tokens=max_new_tokens,
        used_chat_template=used_chat_template_any,
        seed=seed,
        limit=limit,
    )


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)
    summary_csv = Path(args.output_dir) / "summary.csv"
    config_json = Path(args.output_dir) / "run_config.json"

    random.seed(args.seed)
    set_seed(args.seed)

    with config_json.open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    print("[INFO] Configuration:")
    print(json.dumps(vars(args), indent=2))

    loaded_tasks: dict[str, Dataset] = {}
    for task_key in args.tasks:
        loaded_tasks[task_key] = load_task_dataset(
            task_key=task_key,
            split=args.split,
            limit=args.limit,
            seed=args.seed,
        )

    for model_key in args.models:
        model_name = MODEL_REGISTRY[model_key]
        print(f"\n[INFO] Loading model: {model_key} -> {model_name}")
        model, tokenizer = load_model_and_tokenizer(
            model_name=model_name,
            dtype=args.dtype,
            device_map=args.device_map,
            trust_remote_code=args.trust_remote_code,
            use_fast_tokenizer=args.use_fast_tokenizer,
        )

        try:
            for task_key in args.tasks:
                dataset = loaded_tasks[task_key]
                for prompt_style in args.prompt_styles:
                    result = evaluate_dataset(
                        model_key=model_key,
                        model_name=model_name,
                        task_key=task_key,
                        split=args.split,
                        dataset=dataset,
                        prompt_style=prompt_style,
                        model=model,
                        tokenizer=tokenizer,
                        output_dir=Path(args.output_dir),
                        max_new_tokens=args.max_new_tokens,
                        seed=args.seed,
                        limit=args.limit,
                    )
                    append_summary(summary_csv, result)
                    print(
                        f"[DONE] model={result.model_key} task={result.task_key} "
                        f"prompt={result.prompt_style} acc={result.accuracy:.4f} "
                        f"time={result.elapsed_sec:.1f}s"
                    )
        finally:
            del model
            del tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"\n[INFO] Finished. Summary saved to: {summary_csv}")


if __name__ == "__main__":
    main()
