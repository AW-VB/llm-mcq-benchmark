#!/usr/bin/env python3
"""Benchmark evaluation for AI6130 Assignment 1.

Assignment-aligned defaults:
- Uses only Open-Style/Open-LLM-Benchmark with the "questions" config
- Uses the course-notebook split (train)
- Evaluates 4 Hugging Face causal LLMs on 3 benchmark tasks
- Compares baseline / constrained / few-shot prompt styles
- Uses deterministic generation for stable, reproducible accuracy
- Uses plain text for base models and chat templates for instruct/chat models
- Robustly parses multiple-choice answers based on valid option labels
- Saves per-run predictions and summary metrics
- Defaults to full dataset unless --limit is specified

Example:
    python llm_benchmark.py \
        --models tinyllama qwen25_3b deepseek_r1_1p5b qwen3_4b \
        --tasks commonsenseqa openbookqa piqa \
        --prompt-styles baseline constrained fewshot \
        --output-dir results
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable

import torch
from datasets import Dataset, load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed


MODEL_REGISTRY = {
    "tinyllama": "TinyLlama/TinyLlama_v1.1",
    "qwen25_3b": "Qwen/Qwen2.5-3B-Instruct",
    "deepseek_r1_1p5b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "qwen3_4b": "Qwen/Qwen3-4B-Instruct-2507",
}

# Models that require chat template formatting.
CHAT_TEMPLATE_MODELS = {
    "qwen25_3b",
    "deepseek_r1_1p5b",
    "qwen3_4b",
}

# Preferred source (as used in course notebook): one config that contains all questions,
# then filter by the per-sample dataset field.
QUESTIONS_DATASET_SOURCE = ("Open-Style/Open-LLM-Benchmark", "questions")

# Normalized aliases used when filtering QUESTIONS_DATASET_SOURCE.
TASK_FILTER_ALIASES = {
    "commonsenseqa": {"commonsenseqa", "commonsense_qa"},
    "openbookqa": {"openbookqa", "openbook_qa"},
    "piqa": {"piqa"},
}

# Legacy / alternative sources for robustness in different environments.

FEWSHOT_EXAMPLES = {
    "default": (
        "Example 1:\n"
        "Question: What do people usually use to cut paper?\n"
        "A: pillow\nB: spoon\nC: scissors\nD: soap\n"
        "Answer: C\n\n"
        "Example 2:\n"
        "Question: Which object is best for pounding a nail into wood?\n"
        "A: hammer\nB: towel\nC: plate\nD: notebook\n"
        "Answer: A\n\n"
    )
}


@dataclass
class RunResult:
    model_key: str
    model_name: str
    task_key: str
    prompt_style: str
    split: str
    n_samples: int
    n_correct: int
    accuracy: float
    elapsed_sec: float
    avg_sec_per_sample: float
    generation_max_new_tokens: int
    used_chat_template: bool
    seed: int
    limit: int | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Open-LLM benchmark evaluation.")
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
    parser.add_argument("--output-dir", type=str, default="results")
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


def resolve_dtype(dtype: str):
    if dtype == "auto":
        return "auto"
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype]


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def normalize_name(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.strip().lower())


def pick_split(ds_all: dict[str, Dataset], split: str, task_key: str) -> tuple[Dataset, str]:
    if split in ds_all:
        return ds_all[split], split
    available = list(ds_all.keys())
    fallback = "validation" if "validation" in available else available[0]
    print(f"[WARN] Split '{split}' not found for {task_key}; using '{fallback}' instead.")
    return ds_all[fallback], fallback


def load_from_questions_config(task_key: str, split: str) -> tuple[Dataset, str]:
    dataset_name, subset_name = QUESTIONS_DATASET_SOURCE
    ds_all = load_dataset(dataset_name, subset_name)
    dataset, used_split = pick_split(ds_all, split, task_key)

    task_columns = [c for c in ["dataset", "task", "subset", "source", "benchmark", "name"] if c in dataset.column_names]
    if not task_columns:
        raise RuntimeError(
            f"Could not find task column in ({dataset_name}, {subset_name}). "
            f"Columns: {dataset.column_names}"
        )

    alias_set = TASK_FILTER_ALIASES[task_key]
    task_column = task_columns[0]

    def keep_row(sample: dict[str, Any]) -> bool:
        value = normalize_name(str(sample.get(task_column, "")))
        return any(alias in value for alias in alias_set)

    filtered = dataset.filter(keep_row)
    if len(filtered) == 0:
        raise RuntimeError(
            f"No rows matched task='{task_key}' using column='{task_column}' "
            f"in ({dataset_name}, {subset_name}) split='{used_split}'."
        )

    print(
        f"[INFO] Loaded task={task_key} from ({dataset_name}, {subset_name}) "
        f"split={used_split} filtered_on={task_column} rows={len(filtered)}"
    )
    return filtered, used_split


def load_task_dataset(task_key: str, split: str, limit: int | None, seed: int) -> Dataset:
    dataset, used_split = load_from_questions_config(task_key, split)

    if limit is not None:
        n = min(limit, len(dataset))
        # Use a fixed shuffled subset for fair comparison across models.
        dataset = dataset.shuffle(seed=seed).select(range(n))
        print(f"[INFO] Using subset of {n} samples for task={task_key} from split={used_split}")
    else:
        print(f"[INFO] Using full dataset of {len(dataset)} samples for task={task_key} from split={used_split}")

    return dataset


def extract_option_pairs(sample: dict[str, Any]) -> list[tuple[str, str]]:
    """Handle several common MCQ schemas from HF datasets."""
    # Common benchmark notebook schema: sample["options"] = [{label, text}, ...]
    if "options" in sample and isinstance(sample["options"], Iterable):
        pairs = []
        for option in sample["options"]:
            if isinstance(option, dict) and "label" in option and "text" in option:
                pairs.append((str(option["label"]).strip(), str(option["text"]).strip()))
        if pairs:
            return pairs

    # CommonsenseQA / OpenBookQA often use sample["choices"] = {"label": [...], "text": [...]}.
    if "choices" in sample and isinstance(sample["choices"], dict):
        labels = sample["choices"].get("label", [])
        texts = sample["choices"].get("text", [])
        if labels and texts and len(labels) == len(texts):
            return [(str(l).strip(), str(t).strip()) for l, t in zip(labels, texts)]

    # PiQA often has sol1 / sol2 instead of explicit A/B options.
    if "sol1" in sample and "sol2" in sample:
        return [("A", str(sample["sol1"]).strip()), ("B", str(sample["sol2"]).strip())]

    raise ValueError(f"Could not extract answer options from sample keys: {list(sample.keys())}")


def extract_question(sample: dict[str, Any]) -> str:
    for key in ["question", "question_stem", "goal", "query", "input", "prompt"]:
        if key in sample and sample[key] is not None:
            return str(sample[key]).strip()
    raise ValueError(f"Could not find question text in sample keys: {list(sample.keys())}")


def extract_gold_label(sample: dict[str, Any], option_pairs: list[tuple[str, str]]) -> str:
    if "answerKey" in sample and sample["answerKey"] is not None:
        return str(sample["answerKey"]).strip().upper()

    if "label" in sample and sample["label"] is not None:
        label_value = sample["label"]
        # Integer label: map to option letter by index.
        if isinstance(label_value, int):
            return option_pairs[label_value][0].strip().upper()
        if isinstance(label_value, str):
            value = label_value.strip().upper()
            if value.isdigit():
                return option_pairs[int(value)][0].strip().upper()
            return value

    if "answer" in sample and sample["answer"] is not None:
        value = str(sample["answer"]).strip().upper()
        if value.isdigit():
            return option_pairs[int(value)][0].strip().upper()
        return value

    raise ValueError(f"Could not find gold answer in sample keys: {list(sample.keys())}")


def build_prompt(question: str, option_pairs: list[tuple[str, str]], prompt_style: str) -> str:
    lines = [f"Question: {question}", "Choices:"]
    for label, text in option_pairs:
        lines.append(f"{label}. {text}")
    choices_block = "\n".join(lines)

    if prompt_style == "baseline":
        return f"{choices_block}\nAnswer:"

    if prompt_style == "constrained":
        return (
            "You are answering a multiple-choice question.\n"
            f"{choices_block}\n"
            "Return only the correct option letter and nothing else.\n"
            "Answer:"
        )

    if prompt_style == "fewshot":
        return (
            "You are answering multiple-choice questions. Return only the option letter.\n\n"
            f"{FEWSHOT_EXAMPLES['default']}"
            f"{choices_block}\n"
            "Answer:"
        )

    raise ValueError(f"Unknown prompt style: {prompt_style}")



def should_use_chat_template(model_key: str) -> bool:
    return model_key in CHAT_TEMPLATE_MODELS


def format_for_model(model_key: str, tokenizer: AutoTokenizer, prompt: str) -> tuple[str, bool]:
    """Use model-specific formatting: plain text for base models, chat template for instruct/chat models."""
    if should_use_chat_template(model_key):
        if not getattr(tokenizer, "chat_template", None):
            raise RuntimeError(
                f"Model '{model_key}' is configured to use a chat template, but tokenizer.chat_template is missing."
            )
        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return formatted, True
    return prompt, False


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
        # Prefer the tail of the output, but also inspect the first line because
        # some models answer with "E. walked" and then continue with a truncated
        # explanation on later lines.
        candidates.append(lines[-1])
        if len(lines) >= 2:
            candidates.append(" ".join(lines[-2:]))
        candidates.append(lines[0])
    else:
        candidates.append(cleaned)

    patterns = [
        # A / (A) / A. / **A** / **A
        rf"^\*{{0,2}}\(?({label_group})\)?\*{{0,2}}[\s\.,:;!\?-]*$",
        # A. walked / (B) bank / **C** lots of attention
        rf"^\*{{0,2}}\(?({label_group})\)?\*{{0,2}}[\s\.,:;!\?-]+.+$",
        # Answer: A / Answer is A / Answer is: **A**
        rf"^ANSWER(?:\s+IS)?\s*[:：]?\s*\*{{0,2}}\(?({label_group})\)?\*{{0,2}}[\s\.,:;!\?-]*$",
        # Final answer: A / Final answer is A / Final answer is: **A**
        rf"^FINAL\s+ANSWER(?:\s+IS)?\s*[:：]?\s*\*{{0,2}}\(?({label_group})\)?\*{{0,2}}[\s\.,:;!\?-]*$",
        # The correct answer is A / The most appropriate answer is A
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



def infer_one(
    model_key: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int,
) -> tuple[str, bool]:
    formatted_prompt, used_chat_template = format_for_model(model_key, tokenizer, prompt)
    inputs = tokenizer(formatted_prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.eos_token_id,
        "attention_mask": inputs.get("attention_mask"),
        "do_sample": False,
    }

    with torch.no_grad():
        output_ids = model.generate(inputs["input_ids"], **generation_kwargs)

    # Only decode newly generated tokens.
    gen_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    generated_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    return generated_text, used_chat_template


def load_model_and_tokenizer(
    model_name: str,
    dtype: str,
    device_map: str,
    trust_remote_code: bool,
    use_fast_tokenizer: bool,
):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
        use_fast=use_fast_tokenizer,
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=resolve_dtype(dtype),
        device_map=device_map,
        trust_remote_code=trust_remote_code,
    )
    model.eval()
    return model, tokenizer


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


def append_summary(summary_csv: Path, result: RunResult) -> None:
    write_header = not summary_csv.exists()
    with summary_csv.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(result).keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(asdict(result))


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

    # Cache datasets once so that all models see the exact same data.
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
            # Free memory before loading the next model.
            del model
            del tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"\n[INFO] Finished. Summary saved to: {summary_csv}")


if __name__ == "__main__":
    main()
