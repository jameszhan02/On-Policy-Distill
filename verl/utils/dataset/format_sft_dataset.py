# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Suffix-focused SFT dataset for teaching GSM8K answer formatting."""

import torch

from verl.utils.dataset.sft_dataset import SFTDataset


DEFAULT_FORMAT_INSTRUCTION = (
    "\n\nSolve the problem step by step. Your final line must be exactly "
    "`#### <number>`. Replace `<number>` with the numeric answer, and do not "
    "stop immediately after writing `####`."
)


class FormatSFTDataset(SFTDataset):
    """Use normal solutions but train only their final response tokens.

    The base GSM8K response contains the full worked solution and ends in
    ``#### <answer>``. Keeping only the last few loss positions makes that
    suffix matter without teaching the model to replace reasoning with a bare
    answer.
    """

    def __init__(self, parquet_files, tokenizer, config, max_samples=-1):
        super().__init__(parquet_files, tokenizer, config, max_samples=max_samples)
        instruction = config.get("format_instruction", DEFAULT_FORMAT_INSTRUCTION)
        self.format_suffix_tokens = int(config.get("format_suffix_tokens", 16))
        if self.format_suffix_tokens < 1:
            raise ValueError("data.format_suffix_tokens must be at least 1")

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
        full_loss_mask = sample["loss_mask"]
        active_positions = torch.nonzero(full_loss_mask, as_tuple=False).flatten()
        suffix_positions = active_positions[-self.format_suffix_tokens :]

        suffix_loss_mask = torch.zeros_like(full_loss_mask)
        suffix_loss_mask[suffix_positions] = full_loss_mask[suffix_positions]
        sample["loss_mask"] = suffix_loss_mask
        return sample
