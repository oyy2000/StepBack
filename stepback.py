#!/usr/bin/env python3
"""StepBack loop.

The full experiment code has batching, checkpointing, PRM/NLL scoring, GPT
caches, and evaluation tables. This file keeps only the method core:

    draft -> choose rollback point -> resample suffixes -> vote

Rollback-step convention:
    rollback_step is the number of draft steps to keep.
    rollback_step=0 means resample from scratch.
    rollback_step=2 means keep steps[0:2] and resample from step 3 onward.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


MATH_USER_PROMPT = (
    "Solve the following math problem. Present the final answer "
    "in the format: Final Answer: \\boxed{your_answer}.\n"
    "Problem: {question}\n"
    "Answer:"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draft -> rollback -> resample StepBack demo."
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--question", default="")
    parser.add_argument("--question-file", default="")
    parser.add_argument("--gold-answer", default="")
    parser.add_argument(
        "--rollback-signal",
        choices=["gpt", "last", "fixed"],
        default="gpt",
        help="How to select the rollback point.",
    )
    parser.add_argument(
        "--fallback",
        choices=["last", "start"],
        default="last",
        help="Fallback when GPT finds no clear error.",
    )
    parser.add_argument(
        "--fixed-rollback-step",
        type=int,
        default=-1,
        help="Number of steps to keep for --rollback-signal fixed.",
    )
    parser.add_argument("--gpt-model", default="gpt-5.1")
    parser.add_argument("--n-drafts", type=int, default=4)
    parser.add_argument("--n-suffixes", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--out", default="outputs/stepback.json")
    return parser.parse_args()


def read_question(args: argparse.Namespace) -> str:
    if args.question_file:
        path = Path(args.question_file)
        text = path.read_text(encoding="utf-8").strip()
        if path.suffix.lower() == ".json":
            data = json.loads(text)
            return str(data.get("question", data.get("problem", ""))).strip()
        return text
    return args.question.strip()


def build_prompt(tokenizer: Any, question: str) -> str:
    user_prompt = MATH_USER_PROMPT.format(question=question)
    messages = [{"role": "user", "content": user_prompt}]
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            pass
    return (
        "<|im_start|>user\n"
        f"{user_prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def stop_tokens_for(model_id: str) -> List[str]:
    if "llama" in model_id.lower():
        return ["<|eot_id|>", "<|end_of_text|>"]
    return ["<|im_end|>", "<|endoftext|>"]


def split_steps(text: str) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []

    blocks = [part.strip() for part in text.split("\n\n") if part.strip()]
    if len(blocks) > 1:
        return blocks

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) > 1:
        return lines

    return [text]


def extract_boxed_answer(text: str) -> str:
    idx = (text or "").rfind("\\boxed")
    if idx < 0:
        return ""

    depth = 0
    start = None
    for i in range(idx, len(text)):
        if text[i] == "{":
            if depth == 0:
                start = i
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0 and start is not None:
                return text[start + 1 : i].strip()
    return ""


def extract_answer(text: str) -> str:
    boxed = extract_boxed_answer(text)
    if boxed:
        return normalize_answer(boxed)

    match = re.search(r"Final\s+Answer\s*:\s*(.+?)(?:\n|$)", text, re.I)
    if match:
        return normalize_answer(match.group(1))

    nums = re.findall(r"-?\d[\d,]*\.?\d*", text or "")
    if nums:
        return normalize_answer(nums[-1])
    return ""


def normalize_answer(answer: str) -> str:
    return str(answer).strip().strip("$").replace(",", "").rstrip(".")


def answers_match(pred: str, gold: str) -> bool:
    pred_norm = normalize_answer(pred)
    gold_norm = normalize_answer(gold)
    if not pred_norm or not gold_norm:
        return False
    if pred_norm == gold_norm:
        return True
    try:
        return abs(float(pred_norm) - float(gold_norm)) < 1e-6
    except ValueError:
        return False


def majority_vote(answers: Sequence[str]) -> str:
    valid = [a for a in answers if a]
    if not valid:
        return ""
    return Counter(valid).most_common(1)[0][0]


def first_prm_drop(
    step_scores: Sequence[float],
    threshold: float = 0.1,
    fallback: str = "last",
) -> int:
    """Return number of steps to keep based on first large PRM score drop."""
    n = len(step_scores)
    for i in range(max(0, n - 1)):
        if step_scores[i] - step_scores[i + 1] > threshold:
            return i + 1
    return max(n - 1, 0) if fallback == "last" else 0


def first_nll_jump(
    step_nlls: Sequence[float],
    threshold: float = 0.2,
    fallback: str = "last",
) -> int:
    """Return number of steps to keep based on first large NLL increase."""
    n = len(step_nlls)
    for i in range(1, n):
        if step_nlls[i] - step_nlls[i - 1] > threshold:
            return i + 1
    return max(n - 1, 0) if fallback == "last" else 0


def build_first_error_prompt(
    question: str,
    gold_answer: str,
    steps: Sequence[str],
) -> str:
    numbered_steps = "\n".join(
        f"[{i + 1}] {step}" for i, step in enumerate(steps)
    )
    return f"""You are a rigorous math reasoning verifier.

Question:
{question}

Correct final answer:
{gold_answer or "(not provided)"}

The model produced this step-by-step solution:

{numbered_steps}

Find the first step that introduces a mathematical error. A later step that
only propagates an earlier wrong value is not the first error.

Return strict JSON only:
{{
  "first_error_step": <int, 1-indexed; or -1 if no clear error>,
  "reason": "<short reason>"
}}
"""


def parse_json_object(text: str) -> Optional[Dict[str, Any]]:
    text = (text or "").strip()
    if text.startswith("```"):
        lines = [line for line in text.splitlines() if not line.startswith("```")]
        text = "\n".join(lines)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                return None
    return None


def call_gpt_first_error(
    question: str,
    gold_answer: str,
    steps: Sequence[str],
    model: str,
) -> Tuple[Optional[int], Dict[str, Any]]:
    from openai import OpenAI

    client = OpenAI()
    prompt = build_first_error_prompt(question, gold_answer, steps)

    try:
        response = client.responses.create(
            model=model,
            input=prompt,
            temperature=0,
            max_output_tokens=500,
        )
        raw = (response.output_text or "").strip()
    except AttributeError:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=500,
        )
        raw = (response.choices[0].message.content or "").strip()

    parsed = parse_json_object(raw) or {}
    tau = parsed.get("first_error_step")
    if isinstance(tau, int):
        return tau, {"parsed": parsed, "raw": raw}
    return None, {"parsed": parsed, "raw": raw}


def rollback_from_tau(tau: Optional[int], n_steps: int, fallback: str) -> int:
    if tau is not None and tau >= 1:
        return max(0, min(tau - 1, n_steps - 1))
    return max(n_steps - 1, 0) if fallback == "last" else 0


def choose_rollback_step(
    args: argparse.Namespace,
    question: str,
    gold_answer: str,
    steps: Sequence[str],
) -> Tuple[int, Dict[str, Any]]:
    n_steps = len(steps)
    if n_steps == 0:
        return 0, {"signal": args.rollback_signal, "reason": "empty draft"}

    if args.rollback_signal == "last":
        return max(n_steps - 1, 0), {
            "signal": "last",
            "reason": "last-step fallback",
        }

    if args.rollback_signal == "fixed":
        if args.fixed_rollback_step < 0:
            raise ValueError("--fixed-rollback-step is required for fixed mode")
        keep = max(0, min(args.fixed_rollback_step, n_steps))
        return keep, {"signal": "fixed", "reason": f"keep {keep} steps"}

    tau, info = call_gpt_first_error(
        question=question,
        gold_answer=gold_answer,
        steps=steps,
        model=args.gpt_model,
    )
    keep = rollback_from_tau(tau, n_steps, args.fallback)
    info.update({
        "signal": "gpt",
        "first_error_step": tau,
        "fallback": args.fallback,
        "kept_steps": keep,
    })
    return keep, info


def load_vllm(model_id: str, max_model_len: int, gpu_memory_utilization: float):
    from vllm import LLM

    return LLM(
        model=model_id,
        trust_remote_code=True,
        dtype="half",
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
    )


def sample_texts(
    llm: Any,
    prompts: Sequence[str],
    n: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
    stop: Sequence[str],
    seed: int,
) -> List[List[str]]:
    from vllm import SamplingParams

    sampling_params = SamplingParams(
        n=n,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        stop=list(stop),
        seed=seed if seed >= 0 else None,
    )
    outputs = llm.generate(list(prompts), sampling_params=sampling_params)
    return [[sample.text for sample in output.outputs] for output in outputs]


def prefix_from_steps(steps: Sequence[str], rollback_step: int) -> str:
    keep = max(0, min(rollback_step, len(steps)))
    if keep == 0:
        return ""
    return "\n\n".join(steps[:keep]).strip() + "\n\n"


def run_stepback(args: argparse.Namespace) -> Dict[str, Any]:
    question = read_question(args)
    if not question:
        raise ValueError("Provide --question or --question-file")

    llm = load_vllm(
        args.model,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    tokenizer = llm.get_tokenizer()
    base_prompt = build_prompt(tokenizer, question)
    stop = stop_tokens_for(args.model)

    draft_groups = sample_texts(
        llm=llm,
        prompts=[base_prompt],
        n=args.n_drafts,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stop=stop,
        seed=args.seed,
    )
    draft_texts = draft_groups[0]

    draft_records: List[Dict[str, Any]] = []
    suffix_prompts: List[str] = []

    for draft_idx, draft_text in enumerate(draft_texts):
        steps = split_steps(draft_text)
        rollback_step, rollback_info = choose_rollback_step(
            args=args,
            question=question,
            gold_answer=args.gold_answer,
            steps=steps,
        )
        prefix = prefix_from_steps(steps, rollback_step)
        suffix_prompt = base_prompt + prefix
        suffix_prompts.append(suffix_prompt)
        draft_records.append({
            "draft_idx": draft_idx,
            "draft_text": draft_text,
            "draft_answer": extract_answer(draft_text),
            "steps": steps,
            "rollback_step": rollback_step,
            "kept_prefix": prefix,
            "rollback_info": rollback_info,
            "suffixes": [],
        })

    suffix_groups = sample_texts(
        llm=llm,
        prompts=suffix_prompts,
        n=args.n_suffixes,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stop=stop,
        seed=args.seed,
    )

    suffix_answers: List[str] = []
    for draft_record, suffix_texts in zip(draft_records, suffix_groups):
        for suffix_idx, suffix_text in enumerate(suffix_texts):
            answer = extract_answer(suffix_text)
            suffix_answers.append(answer)
            draft_record["suffixes"].append({
                "suffix_idx": suffix_idx,
                "suffix_text": suffix_text,
                "suffix_answer": answer,
            })

    majority_answer = majority_vote(suffix_answers)
    result = {
        "question": question,
        "gold_answer": args.gold_answer,
        "model": args.model,
        "rollback_signal": args.rollback_signal,
        "n_drafts": args.n_drafts,
        "n_suffixes": args.n_suffixes,
        "drafts": draft_records,
        "suffix_answers": suffix_answers,
        "majority_answer": majority_answer,
        "gold_correct": (
            answers_match(majority_answer, args.gold_answer)
            if args.gold_answer else None
        ),
    }
    return result


def main() -> None:
    args = parse_args()
    result = run_stepback(args)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    print(f"Saved: {out_path}")
    print(f"Majority answer: {result['majority_answer']}")
    if result["gold_correct"] is not None:
        print(f"Correct: {result['gold_correct']}")


if __name__ == "__main__":
    main()
