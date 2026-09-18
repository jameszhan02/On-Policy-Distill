#!/usr/bin/env python3
"""Print raw teacher rollouts, including special tokens.

This uses the same raw passthrough chat template as the R1-zero smoke run.
It is intentionally independent of training so teacher behavior can be checked
before changing the student or reward code.
"""

from __future__ import annotations

import argparse

import pandas as pd
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


RAW_CHAT_TEMPLATE = "{{ bos_token }}{{ messages[0]['content'] }}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Teacher model or checkpoint path")
    parser.add_argument("--data", required=True, help="Parquet file produced by gsm8k_r1_zero.py")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=640)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = pd.read_parquet(args.data).head(args.num_samples)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.chat_template = RAW_CHAT_TEMPLATE

    messages = [row["prompt"] for _, row in dataset.iterrows()]
    prompt_texts = [
        tokenizer.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True,
        )
        for message in messages
    ]

    llm = LLM(model=args.model, tensor_parallel_size=args.tensor_parallel_size)
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )
    results = llm.generate(prompt_texts, sampling_params)

    for index, (prompt, result) in enumerate(zip(prompt_texts, results, strict=True)):
        output = result.outputs[0]
        raw_response = tokenizer.decode(output.token_ids, skip_special_tokens=False)
        clean_response = tokenizer.decode(output.token_ids, skip_special_tokens=True)
        print(f"\n=== teacher sample {index} ===")
        print("[prompt]")
        print(prompt)
        print("[response with special tokens]")
        print(raw_response)
        print("[response without special tokens]")
        print(clean_response)
        print(f"[token ids] {output.token_ids}")
        print(f"[eos token] {tokenizer.eos_token!r} id={tokenizer.eos_token_id}")
        print(f"[last token] {output.token_ids[-1] if output.token_ids else None}")
        print(f"[stop reason] {output.finish_reason}")


if __name__ == "__main__":
    main()
