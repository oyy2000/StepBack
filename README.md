# StepBack: Step-Level Error-Localized Resampling for Efficient Test-Time Reasoning

This folder contains a runnable version of the StepBack loop:

```text
draft -> find rollback point -> resample suffixes -> vote
```

The script focuses on the method itself: generate reasoning drafts, identify
where to roll back, continue from the kept prefix, and aggregate the resulting
answers.

## Files

```text
stepback.py        # core implementation
requirements.txt  # runtime dependencies
README.md
```

## Core Ideas

For a single question:

1. Generate `n_drafts` independent chain-of-thought drafts.
2. Split each draft into reasoning steps.
3. Choose a rollback point.
   - `gpt`: ask an OpenAI model for the first erroneous step.
   - `last`: use the last-step fallback.
   - `fixed`: keep a user-specified number of steps.
4. Keep only the prefix before the rollback point.
5. Resample `n_suffixes` continuations from that prefix.
6. Extract final answers and majority-vote over suffix answers.

The script also includes helper functions for PRM-drop and NLL-drop rollback
selection, so the rollback policy is visible even when those scorers are not
loaded.

## Install

```bash
pip install -r requirements.txt
```

For GPT rollback:

```bash
export OPENAI_API_KEY=...
```

## Run With GPT Rollback

```bash
python stepback.py \
  --model Qwen/Qwen2.5-3B-Instruct \
  --question "If a box has 3 red balls and 5 blue balls, how many balls are there?" \
  --gold-answer "8" \
  --rollback-signal gpt \
  --gpt-model gpt-5.1 \
  --n-drafts 4 \
  --n-suffixes 2 \
  --out outputs/demo.json
```

## Run Without GPT

Use the last-step fallback:

```bash
python stepback.py \
  --model Qwen/Qwen2.5-3B-Instruct \
  --question "If a box has 3 red balls and 5 blue balls, how many balls are there?" \
  --rollback-signal last \
  --n-drafts 4 \
  --n-suffixes 2
```

Or keep a fixed number of draft steps:

```bash
python stepback.py \
  --model Qwen/Qwen2.5-3B-Instruct \
  --question "..." \
  --rollback-signal fixed \
  --fixed-rollback-step 2
```

`fixed-rollback-step` means "number of draft steps to keep". For example,
`2` keeps steps 1 and 2, then resamples from step 3 onward.

## Output

The output JSON contains:

- `drafts`: generated drafts, split steps, rollback step, kept prefix, and
  suffix samples.
- `suffix_answers`: all extracted suffix answers.
- `majority_answer`: majority vote over suffix answers.
- `gold_correct`: whether the majority answer matches `--gold-answer`, when a
  gold answer is provided.

## Notes

`stepback.py` keeps the core method loop in one place. Larger experiment code
can add batching, GPU sharding, PRM scoring, NLL scoring, GPT caching,
self-consistency baselines, token accounting, and multi-dataset summaries on
top of this loop.
