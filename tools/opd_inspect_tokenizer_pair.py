#!/usr/bin/env python3
"""Inspect tokenizer families and chat templates for OPD mapping work."""

from __future__ import annotations

import argparse
import json
from textwrap import shorten

from transformers import AutoTokenizer


def detect_family(tokenizer) -> str:
    vocab = tokenizer.get_vocab()
    if "<|begin_of_text|>" in vocab:
        return "llama"
    if "<|im_start|>" in vocab:
        return "qwen"
    if "<\uff5cbegin\u2581of\u2581sentence\uff5c>" in vocab:
        return "deepseek"
    return "unknown"


def compact(value, width: int = 180) -> str:
    text = repr(value)
    return shorten(text, width=width, placeholder=" ...")


def render_chat(tokenizer, messages):
    if not getattr(tokenizer, "chat_template", None):
        return None, "tokenizer has no chat_template"
    try:
        rendered = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        return rendered, None
    except Exception as exc:  # noqa: BLE001 - inspection script should report any tokenizer issue.
        return None, f"{type(exc).__name__}: {exc}"


def print_tokenizer_report(label: str, model_path: str, messages, max_tokens: int) -> None:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    rendered, render_error = render_chat(tokenizer, messages)

    print(f"\n## {label}")
    print(f"path: {model_path}")
    print(f"class: {tokenizer.__class__.__name__}")
    print(f"family_by_repo_heuristic: {detect_family(tokenizer)}")
    print(f"vocab_size: {len(tokenizer.get_vocab())}")
    print(f"bos_token: {compact(tokenizer.bos_token)} id={tokenizer.bos_token_id}")
    print(f"eos_token: {compact(tokenizer.eos_token)} id={tokenizer.eos_token_id}")
    print(f"pad_token: {compact(tokenizer.pad_token)} id={tokenizer.pad_token_id}")
    print("special_tokens_map:")
    print(json.dumps(tokenizer.special_tokens_map, ensure_ascii=False, indent=2, default=str))

    added_vocab = tokenizer.get_added_vocab()
    interesting_added = {
        token: idx
        for token, idx in added_vocab.items()
        if any(marker in token.lower() for marker in ("user", "assistant", "system", "im_", "header", "eot", "begin", "end"))
    }
    print("interesting_added_vocab:")
    print(json.dumps(interesting_added, ensure_ascii=False, indent=2))

    chat_template = getattr(tokenizer, "chat_template", None)
    print("chat_template:")
    print(chat_template if chat_template else "<none>")

    print("rendered_sample_chat:")
    if render_error:
        print(f"<error: {render_error}>")
        return

    print(rendered)
    token_ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
    print(f"rendered_token_count: {len(token_ids)}")
    print(f"first_{max_tokens}_token_ids: {token_ids[:max_tokens]}")
    print(f"first_{max_tokens}_decoded_tokens:")
    for pos, token_id in enumerate(token_ids[:max_tokens]):
        print(f"{pos:03d} {token_id:>8} {compact(tokenizer.decode([token_id], skip_special_tokens=False), 120)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student", required=True, help="Student checkpoint path or HuggingFace model ID.")
    parser.add_argument("--teacher", required=True, help="Teacher checkpoint path or HuggingFace model ID.")
    parser.add_argument("--max-tokens", type=int, default=80, help="Number of rendered tokens to print per tokenizer.")
    parser.add_argument(
        "--messages-json",
        default=None,
        help="Optional JSON list of chat messages. Defaults to a short system/user conversation.",
    )
    args = parser.parse_args()

    if args.messages_json:
        messages = json.loads(args.messages_json)
    else:
        messages = [
            {"role": "system", "content": "You are a concise assistant."},
            {"role": "user", "content": "What is 1+1? Answer briefly."},
        ]

    print("Sample messages:")
    print(json.dumps(messages, ensure_ascii=False, indent=2))
    print_tokenizer_report("student", args.student, messages, args.max_tokens)
    print_tokenizer_report("teacher", args.teacher, messages, args.max_tokens)


if __name__ == "__main__":
    main()
