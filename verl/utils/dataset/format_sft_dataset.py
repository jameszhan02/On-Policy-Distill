# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Format-span SFT dataset for GSM8K."""

import re
import torch

from verl.utils.dataset.sft_dataset import SFTDataset


DEFAULT_FORMAT_INSTRUCTION = (
    "\n\nSolve the problem step by step. Your final line must be exactly "
    "`#### <number>`. Replace `<number>` with the numeric answer, and do not "
    "stop immediately after writing `####`."
)


class FormatSFTDataset(SFTDataset):
    """Train exactly the ``#### <number>`` suffix while masking reasoning.

    The complete gold reasoning remains in the teacher-forced context, but only
    the exact answer marker, numeric answer, and terminating special token
    contribute to loss. ``full`` mode remains available for ordinary SFT.
    """

    _ANSWER_PATTERN = re.compile(r"####\s*-?[\d,]+(?:\.\d+)?")

    def __init__(self, parquet_files, tokenizer, config, max_samples=-1):
        super().__init__(parquet_files, tokenizer, config, max_samples=max_samples)
        instruction = config.get("format_instruction", DEFAULT_FORMAT_INSTRUCTION)
        self.format_loss_mode = str(config.get("format_loss_mode", "full")).lower()
        if self.format_loss_mode not in {"format", "full"}:
            raise ValueError("data.format_loss_mode must be 'format' or 'full'")

        # extra_info.question is the raw GSM8K question. Append an explicit
        # contract compatible with the format requested by OPD prompts.
        self.prompts = [
            self._with_format_instruction(prompt, instruction) for prompt in self.prompts
        ]

    @staticmethod
    def _with_format_instruction(prompt, instruction):
        prompt = str(prompt).rstrip()
        if "#### <number>" in prompt:
            return prompt
        return prompt + instruction

    def __getitem__(self, item):
        sample = super().__getitem__(item)
        if self.format_loss_mode == "full":
            return sample

        tokenizer = self.tokenizer
        prompt = self.prompts[item]
        response = str(self.responses[item])
        answer_matches = list(self._ANSWER_PATTERN.finditer(response))
        if not answer_matches:
            raise ValueError(f"SFT response {item} has no `#### <number>` answer span")
        answer_char_start = answer_matches[-1].start()

        prompt_chat = [{"role": "user", "content": prompt}]
        prompt_chat_str = tokenizer.apply_chat_template(
            prompt_chat,
            add_generation_prompt=True,
            tokenize=False,
            **self.apply_chat_template_kwargs,
        )
        prompt_length = len(tokenizer(prompt_chat_str, add_special_tokens=False)["input_ids"])

        response_chat_str = response + tokenizer.eos_token
        response_encoding = tokenizer(response_chat_str, add_special_tokens=False)
        response_ids = response_encoding["input_ids"]

        # Offset mappings locate the exact `####` start. Slow tokenizers may
        # lack offsets, so their fallback includes one safe boundary token.
        try:
            offsets = tokenizer(
                response_chat_str,
                add_special_tokens=False,
                return_offsets_mapping=True,
            )["offset_mapping"]
            answer_token_start = next(
                index for index, (_, end) in enumerate(offsets) if end > answer_char_start
            )
        except (NotImplementedError, KeyError, StopIteration, TypeError, ValueError):
            prefix_length = len(
                tokenizer(response[:answer_char_start], add_special_tokens=False)["input_ids"]
            )
            answer_token_start = max(0, prefix_length - 1)

        original_sequence_length = prompt_length + len(response_ids)
        if original_sequence_length <= self.max_length or self.truncation == "right":
            left_truncation = 0
        else:
            left_truncation = original_sequence_length - self.max_length

        format_loss_mask = torch.zeros_like(sample["loss_mask"])
        for response_index in range(answer_token_start, len(response_ids)):
            original_index = prompt_length + response_index
            sample_index = original_index - left_truncation
            if 0 <= sample_index < self.max_length and sample["attention_mask"][sample_index]:
                format_loss_mask[sample_index] = 1

        if not torch.any(format_loss_mask):
            raise ValueError(
                f"SFT response {item} answer span was truncated; "
                "increase data.max_length or use truncation=left"
            )
        sample["loss_mask"] = format_loss_mask
        return sample
