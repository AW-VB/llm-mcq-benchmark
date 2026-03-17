# LLM Benchmark on Multiple-Choice QA

This repository evaluates several open-weight language models on short-form multiple-choice question answering. The benchmark compares prompt styles, measures both accuracy and latency, and includes an auxiliary script for per-example error analysis.

The project was developed for a course assignment (AI6130@NTU), but the code is organized as a small reproducible benchmark runner rather than a one-off notebook.

## What This Repo Does

- Evaluates 4 Hugging Face causal LLMs:
  - `TinyLlama/TinyLlama_v1.1`
  - `Qwen/Qwen2.5-3B-Instruct`
  - `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`
  - `Qwen/Qwen3-4B-Instruct-2507`
- Runs on 3 multiple-choice reasoning tasks from `Open-Style/Open-LLM-Benchmark`:
  - CommonsenseQA
  - OpenBookQA
  - PIQA
- Compares 3 prompt styles:
  - `baseline`
  - `constrained`
  - `fewshot`
- Uses deterministic decoding and a strict answer parser for concise MCQ evaluation
- Saves per-run predictions and summary metrics
- Provides a separate error-analysis script with per-example outputs

## Main Files

- [llm_benchmark.py](llm_benchmark.py): main benchmark runner
- [llm_benchmark_error_analysis.py](llm_benchmark_error_analysis.py): exports detailed per-example error-analysis artifacts
- [benchmark_report_visualization.ipynb](benchmark_report_visualization.ipynb): notebook for summarizing results and generating report figures
- [requirements.txt](requirements.txt): Python dependencies
- `archive/`: earlier script versions kept for project history

## Evaluation Setup

- Dataset source: `Open-Style/Open-LLM-Benchmark` with the `questions` config
- Split: `train`
- Decoding: greedy (`do_sample=False`)
- Default generation budget: `max_new_tokens=8`
- Prompting:
  - `baseline`: plain MCQ prompt
  - `constrained`: explicitly asks for only the option letter
  - `fewshot`: constrained prompt plus two demonstrations
- Formatting:
  - TinyLlama uses plain text prompts
  - Qwen2.5, Qwen3, and DeepSeek use tokenizer chat templates
- Parsing:
  - outputs are scored with a strict rule-based parser
  - only short explicit answer formats are accepted
  - ambiguous or incomplete generations are treated as unparseable

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

This project is intended for local execution with Hugging Face models. A GPU is strongly recommended for the larger models.

## Run the Benchmark

Run the full benchmark:

```bash
python llm_benchmark.py \
  --models tinyllama qwen25_3b deepseek_r1_1p5b qwen3_4b \
  --tasks commonsenseqa openbookqa piqa \
  --prompt-styles baseline constrained fewshot \
  --output-dir results
```

Run a smaller debug subset:

```bash
python llm_benchmark.py \
  --models tinyllama \
  --tasks openbookqa \
  --prompt-styles constrained \
  --limit 20 \
  --output-dir debug_results
```

Outputs:

- `summary.csv`: run-level metrics
- `run_config.json`: saved CLI configuration
- `predictions__<model>__<task>__<prompt>.jsonl`: per-sample predictions

For report-ready plots and summary visualizations, see [benchmark_report_visualization.ipynb](benchmark_report_visualization.ipynb).

## Run Error Analysis

```bash
python llm_benchmark_error_analysis.py \
  --models deepseek_r1_1p5b tinyllama \
  --tasks commonsenseqa openbookqa piqa \
  --prompt-styles baseline constrained fewshot \
  --max-new-tokens 32 \
  --output-dir error_analysis_results
```

Additional outputs:

- `analysis__<model>__<task>__<prompt>.csv`
- `analysis__<model>__<task>__<prompt>.jsonl`

These files include raw generations, parsed labels, prompts, gold answers, parse-failure flags, and option metadata for each example.

## Summary of Findings

- Qwen3-4B-Instruct-2507 was the strongest model in this evaluation setting.
- Qwen2.5-3B-Instruct also performed well, especially with constrained prompting.
- TinyLlama remained much weaker and often near chance on harder tasks.
- Constrained and few-shot prompting improved both accuracy and latency relative to the baseline prompt.
- DeepSeek-R1-Distill-Qwen-1.5B produced no parseable answers under the shared short-generation protocol, suggesting a mismatch between the benchmark setting and the model's reasoning-oriented generation style.

One important takeaway from the project is that measured benchmark accuracy depends not only on underlying answer quality, but also on output formatting and whether a model can emit a scoreable final label under a short generation budget.

## Limitations

- The benchmark uses the training split rather than a held-out evaluation split.
- Public benchmark contamination may affect absolute scores.
- The strict parser intentionally favors precision over recall.
- The short generation budget is appropriate for concise MCQ answering, but it disadvantages reasoning-oriented models that expect longer outputs.

## Project Evolution

This repository reflects the final evaluation setup used in the report rather than the very first implementation. After inspecting early generations and parse failures, two changes were carried into the final benchmark runner:

- the default `max_new_tokens` was increased from 6 to 8 to reduce truncation of otherwise valid short answers
- the answer parser was made stricter so that stray option letters in long, incomplete, or reasoning-style outputs would not be counted as valid predictions

The goal of these changes was not to optimize for any single model, but to make the benchmark more faithful to short-form multiple-choice answering under a shared evaluation protocol.

## Notes

This repository keeps the final benchmark runner in the project root and preserves earlier trial-and-error versions in `archive/`. The current `llm_benchmark.py` reflects the final evaluation setup used in the report, including the stricter answer parser and the 8-token generation budget.
