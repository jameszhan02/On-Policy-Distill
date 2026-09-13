#!/usr/bin/env python3
"""Quickly inspect GSM8K generations and answer-format compliance."""

from __future__ import annotations

import argparse
import json
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


STRICT_FORMAT_INSTRUCTION = (
    "\n\nSolve the problem step by step. Your final line must be exactly "
    "`#### <number>`. Replace `<number>` with the numeric answer, and do not "
    "stop immediately after writing `####`."
)
NUMBER = r"-?\d[\d,]*(?:\.\d+)?"
STRICT_FINAL_RE = re.compile(rf"(?:^|\n)[ \t]*####[ \t]*({NUMBER})[ \t]*\Z")
LOOSE_ANSWER_RE = re.compile(rf"####\s*({NUMBER})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="HuggingFace model directory or model ID")
    parser.add_argument("--data", required=True, help="GSM8K validation parquet")
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=640)
    parser.add_argument("--show", type=int, default=8, help="Number of responses to print")
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument(
        "--prompt-mode",
        choices=("strict", "dataset"),
        default="strict",
        help="Use the strict SFT instruction or the prompt stored in the parquet",
    )
    parser.add_argument("--output", help="Optional JSONL path for all generations")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def extract_question(example: dict, prompt_mode: str) -> str:
    if prompt_mode == "dataset":
        messages = example.get("prompt")
        if isinstance(messages, list):
            for message in reversed(messages):
                if isinstance(message, dict) and message.get("role") == "user":
                    return str(message.get("content", ""))

    extra_info = example.get("extra_info") or {}
    question = extra_info.get("question") if isinstance(extra_info, dict) else None
    if question is None:
        messages = example.get("prompt") or []
        question = next(
            (
                message.get("content", "")
                for message in reversed(messages)
                if isinstance(message, dict) and message.get("role") == "user"
            ),
            "",
        )
    question = str(question)
    if prompt_mode == "strict" and "#### <number>" not in question:
        question += STRICT_FORMAT_INSTRUCTION
    return question


def extract_ground_truth(example: dict) -> str | None:
    reward_model = example.get("reward_model") or {}
    if isinstance(reward_model, dict) and reward_model.get("ground_truth") is not None:
        return str(reward_model["ground_truth"])

    extra_info = example.get("extra_info") or {}
    answer = extra_info.get("answer", "") if isinstance(extra_info, dict) else ""
    matches = LOOSE_ANSWER_RE.findall(str(answer))
    return matches[-1] if matches else None


def numbers_equal(left: str | None, right: str | None) -> bool:
    if left is None or right is None:
        return False
    left = left.replace(",", "").strip()
    right = right.replace(",", "").strip()
    try:
        return Decimal(left) == Decimal(right)
    except InvalidOperation:
        return left == right


def classify_response(response: str, ground_truth: str | None) -> dict:
    stripped = response.strip()
    strict_match = STRICT_FINAL_RE.search(stripped)
    loose_matches = LOOSE_ANSWER_RE.findall(stripped)
    prediction = loose_matches[-1] if loose_matches else None

    if strict_match:
        failure_type = None
    elif "####" not in stripped:
        failure_type = "missing_marker"
    elif not loose_matches:
        failure_type = "missing_number_after_marker"
    else:
        failure_type = "trailing_or_nonfinal_text"

    return {
        "format_valid": strict_match is not None,
        "prediction": prediction,
        "loose_correct": numbers_equal(prediction, ground_truth),
        "strict_correct": strict_match is not None and numbers_equal(strict_match.group(1), ground_truth),
        "failure_type": failure_type,
    }


def main() -> None:
    args = parse_args()
    if args.num_samples < 1 or args.batch_size < 1:
        raise ValueError("--num-samples and --batch-size must be positive")

    dataset = load_dataset("parquet", data_files=args.data, split="train")
    sample_count = min(args.num_samples, len(dataset))
    dataset = dataset.shuffle(seed=args.seed).select(range(sample_count))

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    model.to(args.device).eval()

    records = []
    for batch_start in range(0, sample_count, args.batch_size):
        examples = [dataset[index] for index in range(batch_start, min(batch_start + args.batch_size, sample_count))]
        questions = [extract_question(example, args.prompt_mode) for example in examples]
        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": question}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for question in questions
        ]
        inputs = tokenizer(prompts, padding=True, return_tensors="pt").to(args.device)

        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        prompt_width = inputs["input_ids"].shape[1]
        for offset, example in enumerate(examples):
            response_ids = generated[offset, prompt_width:]
            response = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
            ground_truth = extract_ground_truth(example)
            result = classify_response(response, ground_truth)
            records.append(
                {
                    "index": batch_start + offset,
                    "question": questions[offset],
                    "response": response,
                    "ground_truth": ground_truth,
                    "response_tokens": len(tokenizer(response, add_special_tokens=False)["input_ids"]),
                    **result,
                }
            )
        print(f"generated {len(records)}/{sample_count}", flush=True)

    failures = [record for record in records if not record["format_valid"]]
    correct_format = [record for record in records if record["format_valid"]]
    display_records = (failures + correct_format)[: args.show]
    for record in display_records:
        print(f"\n{'=' * 24} SAMPLE {record['index']} {'=' * 24}")
        print(record["response"])
        print(
            f"\nformat_valid={record['format_valid']} "
            f"failure={record['failure_type']} pred={record['prediction']} "
            f"gt={record['ground_truth']} strict_correct={record['strict_correct']}"
        )

    failure_counts = {
        name: sum(record["failure_type"] == name for record in records)
        for name in ("missing_marker", "missing_number_after_marker", "trailing_or_nonfinal_text")
    }
    format_valid = sum(record["format_valid"] for record in records)
    loose_correct = sum(record["loose_correct"] for record in records)
    strict_correct = sum(record["strict_correct"] for record in records)
    mean_tokens = sum(record["response_tokens"] for record in records) / len(records)

    print(f"\n{'=' * 24} SUMMARY {'=' * 24}")
    print(f"model:                 {args.model}")
    print(f"prompt mode:           {args.prompt_mode}")
    print(f"samples:               {sample_count}")
    print(f"format valid:          {format_valid / sample_count:.2%} ({format_valid}/{sample_count})")
    print(f"loose answer accuracy: {loose_correct / sample_count:.2%} ({loose_correct}/{sample_count})")
    print(f"strict accuracy:       {strict_correct / sample_count:.2%} ({strict_correct}/{sample_count})")
    print(f"mean response tokens:  {mean_tokens:.1f}")
    for name, count in failure_counts.items():
        print(f"{name + ':':23} {count}")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as output_file:
            for record in records:
                output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"saved generations:     {output_path}")


if __name__ == "__main__":
    main()
