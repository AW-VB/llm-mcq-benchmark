#!/usr/bin/env python3
"""Benchmark runner with per-example error-analysis outputs.

This script intentionally does not modify `llm_benchmark.py`.
It reuses the original benchmark utilities, but writes richer per-question
artifacts for debugging parser issues and prompt effects.

Outputs:
- `summary.csv`: same run-level summary as the original benchmark
- `analysis__<model>__<task>__<prompt>.csv`: one row per question
- `analysis__<model>__<task>__<prompt>.jsonl`: same information in JSONL

Example:
    python3 llm_benchmark_error_analysis.py \
        --models deepseek_r1_1p5b tinyllama \
        --tasks commonsenseqa openbookqa piqa \
        --prompt-styles baseline constrained fewshot \
        --max-new-tokens 32 \
        --output-dir error_analysis_results
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from dataclasses import asdict
from pathlib import Path

import torch
from datasets import Dataset
from transformers import set_seed

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
    parse_predicted_label,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the benchmark and save detailed per-example error-analysis files."
    )
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
    parser.add_argument("--output-dir", type=str, default="error_analysis_results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=6,
        help="Maximum number of generated tokens per question. Increase this for reasoning-heavy models.",
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


def write_analysis_rows(
    csv_path: Path,
    jsonl_path: Path,
    rows: list[dict],
) -> None:
    fieldnames = [
        "index",
        "model_key",
        "model_name",
        "task_key",
        "split",
        "prompt_style",
        "n_options",
        "gold",
        "parsed_label",
        "is_correct",
        "parse_failed",
        "raw_output",
        "question",
        "prompt",
        "used_chat_template",
        "generation_max_new_tokens",
        "options_json",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def evaluate_dataset_with_error_analysis(
    model_key: str,
    model_name: str,
    task_key: str,
    split: str,
    dataset: Dataset,
    prompt_style: str,
    model,
    tokenizer,
    output_dir: Path,
    max_new_tokens: int,
    seed: int,
    limit: int | None,
) -> RunResult:
    analysis_csv = output_dir / f"analysis__{model_key}__{task_key}__{prompt_style}.csv"
    analysis_jsonl = output_dir / f"analysis__{model_key}__{task_key}__{prompt_style}.jsonl"

    analysis_rows: list[dict] = []
    correct = 0
    total = len(dataset)
    used_chat_template_any = False

    start = time.time()
    for idx, sample in enumerate(dataset):
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

        parsed_label = parse_predicted_label(raw_output, valid_labels)
        is_correct = parsed_label == gold
        correct += int(is_correct)

        analysis_rows.append(
            {
                "index": idx,
                "model_key": model_key,
                "model_name": model_name,
                "task_key": task_key,
                "split": split,
                "prompt_style": prompt_style,
                "n_options": len(option_pairs),
                "gold": gold,
                "parsed_label": parsed_label,
                "is_correct": is_correct,
                "parse_failed": parsed_label is None,
                "raw_output": raw_output,
                "question": question,
                "prompt": prompt,
                "used_chat_template": used_chat_template,
                "generation_max_new_tokens": max_new_tokens,
                "options_json": json.dumps(
                    [{"label": label, "text": text} for label, text in option_pairs],
                    ensure_ascii=False,
                ),
            }
        )

    write_analysis_rows(analysis_csv, analysis_jsonl, analysis_rows)

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
                    result = evaluate_dataset_with_error_analysis(
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
