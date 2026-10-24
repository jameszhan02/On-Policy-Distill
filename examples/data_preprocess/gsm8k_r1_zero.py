#!/usr/bin/env python3
"""Preprocess GSM8K into an lllm r1_zero prompt/answer protocol.

This mirrors examples/data_preprocess/gsm8k.py's extraction logic exactly
(ground_truth via the "#### number" regex on the *source* answer, which GSM8K
itself always uses), but builds the canonical lllm `r1_zero` prompt. The prompt
ends at ``Assistant: <think>`` and the model emits the reasoning continuation,
``</think>``, and ``<answer>...</answer>``.

Why this exists: the OPD training pipeline's own diagnostics (`#### number`)
and the actual eval harness (lgsm8k_eval.py's r1_zero grader, which requires
<think></think> <answer></answer> tags and has no notion of "####" at all)
were checking two incompatible target formats. This script produces training
data in the eval's actual target format instead.

The `prompt` column is a single user-turn message whose *content* is the full
r1_zero text (instructions + the real question).
Combined with `data.apply_chat_template_kwargs.chat_template` set to a raw
passthrough template (see cross_distill_smoke_1gpu.sh's R1_ZERO_RAW_PROMPT
mode), this renders as literally `{bos_token}{r1_zero text}` -- no chat-template
role markers from either tokenizer -- matching the raw-prompt format R1-Zero
style training/eval actually uses.
"""

from __future__ import annotations

import argparse
import os
import re

import pandas as pd

# Canonical lllm/alignment/prompts/r1_zero.prompt. The prompt owns the opening
# <think> tag; BOS/EOS are owned by each model's tokenizer.
R1_ZERO_TEMPLATE = (
    "A conversation between User and Assistant. The User asks a question, and the Assistant solves it. "
    "The Assistant first thinks about the reasoning process in the mind and then provides the User with "
    "the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within "
    "<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> "
    "<answer> answer here </answer>.\n"
    "User: {question}\n"
    "Assistant: <think>"
)

# Backwards-compatible names. Both variants intentionally render the exact
# same prompt so old launch commands and cached metadata remain readable.
R1_ZERO_LLAMA_TEMPLATE = R1_ZERO_TEMPLATE
R1_ZERO_ONE_SHOT_TEMPLATE = R1_ZERO_TEMPLATE


def extract_solution(solution_str: str) -> str:
    """Identical to gsm8k.py's extraction: GSM8K's own source answers always
    use "#### number" regardless of what prompt format we train the model on.
    """
    solution = re.search(r"#### (\-?[0-9\.\,]+)", solution_str)
    assert solution is not None, f"no #### marker in: {solution_str!r}"
    return solution.group(0).split("#### ")[1].replace(",", "")


def build_prompt(question: str, prompt_variant: str = "r1_zero") -> list[dict]:
    if prompt_variant not in {"r1_zero", "r1_zero_llama"}:
        raise ValueError(f"Unsupported R1 prompt variant: {prompt_variant}")
    r1_zero_text = R1_ZERO_TEMPLATE.format(question=question)
    return [{"role": "user", "content": r1_zero_text}]


def convert(dataset, split: str, prompt_variant: str = "r1_zero") -> pd.DataFrame:
    rows = []
    for idx, ex in enumerate(dataset):
        question_raw = ex["question"]
        answer_raw = ex["answer"]
        rows.append(
            {
                "data_source": "openai/gsm8k",
                "prompt": build_prompt(question_raw, prompt_variant),
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": extract_solution(answer_raw)},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": answer_raw,
                    "question": question_raw,
                    "prompt_variant": prompt_variant,
                },
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--local_save_dir", default="data/gsm8k_r1zero", help="Output directory for train.parquet/test.parquet."
    )
    parser.add_argument(
        "--prompt-variant",
        choices=("r1_zero", "r1_zero_llama"),
        default="r1_zero",
        help="Raw lllm prompt format to store in the parquet prompt column.",
    )
    args = parser.parse_args()

    import datasets

    dataset = datasets.load_dataset("openai/gsm8k", "main")
    train_df = convert(dataset["train"], "train", args.prompt_variant)
    test_df = convert(dataset["test"], "test", args.prompt_variant)

    os.makedirs(args.local_save_dir, exist_ok=True)
    train_path = os.path.join(args.local_save_dir, "train.parquet")
    test_path = os.path.join(args.local_save_dir, "test.parquet")
    train_df.to_parquet(train_path)
    test_df.to_parquet(test_path)
    print(f"wrote {len(train_df)} rows -> {train_path}")
    print(f"wrote {len(test_df)} rows -> {test_path}")


if __name__ == "__main__":
    main()
