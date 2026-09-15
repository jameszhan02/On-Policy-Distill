#!/usr/bin/env python3
"""Run a real generation on the teacher checkpoint and show EVERY token, including
special tokens, with nothing hidden by skip_special_tokens=True.

This is an empirical check, not a config-file check: it tells you what the model
actually emits, not what its tokenizer_config.json/chat_template/generation_config.json
declare it *should* emit. Use it to confirm whether a given special token (e.g.
"<|im_start|>", "<|im_end|>") ever shows up in real output for this checkpoint.

Usage:
    python3 tools/opd_probe_teacher_raw_output.py \
        --model /path/to/olmo-2-instruct \
        --max-new-tokens 64

Requires `transformers` + `torch` and enough local compute/memory to load the
model (run this on the box that hosts the checkpoint, not necessarily locally).
"""

from __future__ import annotations

import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Marker tokens from every family this repo's OPD code currently distinguishes,
# so a single run flags exactly which ones (if any) actually appear.
WATCH_TOKENS = [
    "<|begin_of_text|>", "<|eot_id|>", "<|end_of_text|>",       # llama
    "<|im_start|>", "<|im_end|>",                                # qwen (also present-but-unused in OLMo's vocab)
    "<｜begin▁of▁sentence｜>", "<｜end▁of▁sentence｜>",              # deepseek
    "<|user|>", "<|assistant|>", "<|system|>", "<|endoftext|>",  # olmo / tulu
]


def compact(text: str, width: int = 100) -> str:
    text = repr(text)
    return text if len(text) <= width else text[: width - 3] + "..."


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Local checkpoint path or HF model ID.")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--do-sample", action="store_true", help="Sample instead of greedy decode.")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--messages-json",
        default=None,
        help="Optional JSON list of chat messages. Defaults to a short GSM8K-style prompt.",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    if args.messages_json:
        messages = json.loads(args.messages_json)
    else:
        messages = [
            {"role": "user", "content": "What is 2+2? Give the final answer in the form #### number."},
        ]

    print(f"Loading tokenizer/model from: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    prompt_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    print("\n=== Rendered prompt (from chat_template) ===")
    print(prompt_text)

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **prompt_ids,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature if args.do_sample else None,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    prompt_len = prompt_ids["input_ids"].shape[-1]
    generated_ids = output_ids[0, prompt_len:].tolist()

    print("\n=== Raw generated text (skip_special_tokens=False) ===")
    raw_text = tokenizer.decode(generated_ids, skip_special_tokens=False)
    print(raw_text)

    print("\n=== Token-by-token (id, decoded text) ===")
    for pos, tok_id in enumerate(generated_ids):
        tok_text = tokenizer.decode([tok_id], skip_special_tokens=False)
        print(f"{pos:03d} {tok_id:>8} {compact(tok_text, 80)}")

    print("\n=== Watch-list marker check (did any actually appear?) ===")
    for marker in WATCH_TOKENS:
        marker_id = tokenizer.convert_tokens_to_ids(marker)
        present_in_vocab = isinstance(marker_id, int) and marker_id != tokenizer.unk_token_id
        seen_in_output = isinstance(marker_id, int) and marker_id in generated_ids
        if present_in_vocab:
            print(f"  {marker!r:24} in_vocab=True  id={marker_id:<8} seen_in_this_output={seen_in_output}")


if __name__ == "__main__":
    main()
