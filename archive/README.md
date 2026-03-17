# Archive Notes

This folder keeps earlier script versions from the development process.

The archived files are preserved for project history and comparison, but they are not the recommended entry points for running the benchmark. The final benchmark runner is [llm_benchmark.py](../llm_benchmark.py) in the repository root.

## Archived Files

- [llm_benchmark_initial.py](./llm_benchmark_initial.py): early benchmark runner before the final evaluation refinements

## Why the Benchmark Changed

The final benchmark setup differs from the initial version in two main ways.

First, the default generation budget was increased from `max_new_tokens=6` to `max_new_tokens=8`. Early runs showed that some models sometimes produced short answer templates such as `The correct answer is:` but were truncated before emitting the final option label. Increasing the budget to 8 reduced these truncation artifacts while preserving the short-answer nature of the benchmark.

Second, the answer parser was made stricter. The earlier version could extract option letters too permissively from longer generated text, which risked counting incomplete or reasoning-style outputs as valid answers. The final parser only accepts short, explicit answer formats so that scored predictions are easier to interpret and defend.

These changes came out of error analysis on early runs. They were intended to improve the reliability of the evaluation protocol, not to add model-specific tuning.
