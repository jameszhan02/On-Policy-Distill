# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict
from concurrent.futures import Future
from contextlib import nullcontext
from datetime import datetime
from types import SimpleNamespace
from codetiming import Timer
from functools import partial

import torch
import zmq
import io
import os
import json
import time
import queue
import threading
import concurrent.futures
import unicodedata
import warnings

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager

from transformers import AutoConfig, AutoTokenizer

# config.json "model_type" -> our family label. This is set at conversion/save
# time from the model's actual architecture class -- unlike eos_token or
# chat_template, it exists and is reliable even for a base/template-less
# checkpoint, since AutoModelForCausalLM needs it to pick the right class
# regardless of instruction-tuning status.
#
# Known limitation: this maps ARCHITECTURE, not chat convention. A distilled
# checkpoint can mix the two (e.g. deepseek-ai/DeepSeek-R1-Distill-Qwen-7B has
# model_type=="qwen2" but uses DeepSeek's chat template/special tokens) -- for
# that kind of case, the eos_token/chat_template checks above must take
# priority, and this table is only consulted when those found nothing.
_MODEL_TYPE_TO_FAMILY = {
    "llama": "llama",
    "olmo": "olmo",
    "olmo2": "olmo",
    "olmoe": "olmo",
    "qwen": "qwen",
    "qwen2": "qwen",
    "qwen2_moe": "qwen",
    "qwen3": "qwen",
    "qwen3_moe": "qwen",
    "deepseek": "deepseek",
    "deepseek_v2": "deepseek",
    "deepseek_v3": "deepseek",
}

teacher_topk_logps_padded, teacher_topk_indices_padded, teacher_chunk_ids_padded = None, None, None
DEBUG = False

# R1_ZERO_MODE: student/teacher prompts are rendered via a raw passthrough
# chat_template (see cross_distill_smoke_1gpu.sh) -- literally
# "{bos_token}{content}", no Llama/Qwen/OLMo role markers baked in by either
# tokenizer's own native template. The family-based chat-template mapping
# below assumes those role markers exist to translate between; they don't in
# this mode, so retokenize_batch and the response-boundary detection take a
# separate, simpler path gated on this flag instead.
R1_ZERO_MODE = os.environ.get("R1_ZERO_MODE", "0").strip() == "1"

# ---- Alignment dump configuration (controlled by environment variables) ----
# OPD_DUMP_DIR: directory to write alignment debug dumps (disabled if unset/empty)
# OPD_DUMP_NUM_SEQS: number of sequences to dump per step (default: 2)
# OPD_DUMP_MAX_STEPS: stop dumping after this many steps (default: 50)
OPD_DUMP_DIR = os.environ.get("OPD_DUMP_DIR", "")
OPD_DUMP_NUM_SEQS = int(os.environ.get("OPD_DUMP_NUM_SEQS", "2"))
OPD_DUMP_MAX_STEPS = int(os.environ.get("OPD_DUMP_MAX_STEPS", "50"))
_dump_step_counter = 0


def _dump_alignment(step, seq_idx, dump_data):
    """Write alignment debug info to a JSON file.

    Args:
        step: global step counter
        seq_idx: sequence index within the batch
        dump_data: dict containing alignment details
    """
    if not OPD_DUMP_DIR:
        return
    os.makedirs(OPD_DUMP_DIR, exist_ok=True)
    filepath = os.path.join(OPD_DUMP_DIR, f"step_{step:04d}_seq_{seq_idx:03d}.json")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(dump_data, f, ensure_ascii=False, indent=2)


def chunk_list(lst, n_chunks):
    """Split a list into chunks of equal length"""
    size = len(lst) // n_chunks
    for i, start in enumerate(range(0, len(lst), size)):
        if i == n_chunks - 1:
            yield lst[start:]
            return
        else:
            yield lst[start : start + size]


def serialize(data):
    buffer = io.BytesIO()
    torch.save(data, buffer)
    return buffer.getbuffer()


def deserialize(message):
    buffer = io.BytesIO(message)
    return torch.load(buffer)


def check_if_invalid(topk_logps, inputs):
    is_valid = True
    reason = ""
    for x in topk_logps:
        if x.isnan().any():
            is_valid = False
            reason = "nan"
            break
        elif x.isinf().any():
            is_valid = False
            reason = "inf"
            break
        elif (x == 0).any():
            is_valid = False
            reason = "zero"
            break
    if not is_valid:
        if isinstance(inputs, torch.Tensor):
            inputs = inputs.tolist()
        with open("teacher_debug.log", "a") as f:
            f.write("{}\n".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
            f.write(f"{reason}\n")
            f.write(f"{str(inputs)}\n")


class TeacherClient:
    def __init__(
        self,
        server_ip,
        server_port,
        teacher_ckpt_path,
        num_microbatches=1,
        max_tokens=1,
        n_server_workers=1,
        temperature=1,
        only_response=False,
        max_seq_len=None,
    ) -> None:
        self.server_ip = server_ip
        self.server_port = server_port
        # self.server_ip = "29.160.160.136"
        # self.server_port = 15555
        self.num_microbatches = num_microbatches
        self.n_server_workers = n_server_workers
        self.max_tokens = max_tokens
        self.task_queue = queue.Queue()
        self.mutex = threading.Lock() if n_server_workers > 1 else nullcontext()
        self.context = zmq.Context()
        self.temperature = temperature
        self.only_response = only_response
        self.max_seq_len = max_seq_len
        self.large_chunk_threshold = int(os.environ.get('OPD_LARGE_CHUNK_THRESHOLD', '6'))
        self.tokenizer = AutoTokenizer.from_pretrained(teacher_ckpt_path)
        self.student_tokenizer = None
        self._same_tokenizer = None  # cached result of tokenizer comparison
        self._tokenizer_debug_printed = False
        self._run()

    def _is_same_tokenizer(self):
        """Check if teacher and student tokenizers have the same vocabulary. Result is cached."""
        if self._same_tokenizer is None:
            self._same_tokenizer = (self.tokenizer.get_vocab() == self.student_tokenizer.get_vocab())
        return self._same_tokenizer

    def _detect_model_family(self, tokenizer):
        """Detect the model family of a tokenizer. Result is cached.

        Deliberately does NOT use `tokenizer.get_vocab()` membership as
        evidence anywhere in here anymore. Vocab membership only proves a
        token id is *addressable* -- it says nothing about whether the model
        was ever trained to treat it as meaningful, or whether this specific
        checkpoint's real chat format ever writes it. That assumption broke
        twice on OLMo-2 alone: its vocab contains an unused "<|im_end|>" id
        (misclassifying it as "qwen"), and a follow-up fix assumed "<|user|>"/
        "<|assistant|>" were themselves addressable vocab entries -- they are
        NOT (verified against the real tokenizer.json: absent from both the
        base BPE vocab and added_tokens; they're plain sub-word-tokenized text
        that only ever appears inside the chat_template string).

        Two signals are actually trustworthy, and both are used here:
          1. `tokenizer.eos_token` -- the string the checkpoint's own config
             designates as its real stop token. Unambiguous for LLaMA-Instruct
             ("<|eot_id|>"), Qwen-chat ("<|im_end|>"), and DeepSeek
             ("<｜end▁of▁sentence｜>"). Verified against each family's real HF
             config, not assumed.
          2. `tokenizer.chat_template` -- the literal Jinja2 template string
             actually used to render conversations. Checking for a marker as
             literal text *inside this string* (not vocab membership) proves
             the checkpoint's real format uses it. Verified marker literalness
             against each family's real chat_template (LLaMA-Instruct hardcodes
             "<|start_header_id|>"/"<|end_header_id|>"/"<|eot_id|>" as literal
             text but references its leading bos via the "{{ bos_token }}"
             template *variable*, not literal "<|begin_of_text|>" text -- so
             that marker specifically is NOT checked here; Qwen-chat hardcodes
             "<|im_start|>"; DeepSeek hardcodes its fullwidth markers; OLMo/Tulu
             hardcodes "<|user|>"/"<|assistant|>").

        `eos_token` alone can't fully disambiguate: "<|endoftext|>" is a
        generic tiktoken/GPT-2-lineage default shared by OLMo/Tulu-style chat
        checkpoints and unrelated checkpoints from other lineages. For that
        case, chat_template is the tiebreaker.

        A checkpoint with no chat_template AND an ambiguous/generic eos_token
        (a base/non-instruct checkpoint -- e.g. a real OLMo base/mid-train
        teacher) falls back to `_family_from_model_type()`, which reads
        config.json's model_type/architectures instead of guessing from vocab
        lineage. That field is set from the model's actual architecture class
        at conversion/save time, so it's present and reliable regardless of
        instruction-tuning status. Only if that ALSO fails (offline, no
        config found, unrecognized model_type) does this return "unknown".

        Returns:
            str: One of "llama", "qwen", "deepseek", "olmo", or "unknown".
        """
        if not hasattr(self, '_family_cache'):
            self._family_cache = {}
        tok_id = id(tokenizer)
        if tok_id not in self._family_cache:
            eos = tokenizer.eos_token
            chat_template = getattr(tokenizer, "chat_template", None) or ""
            if eos == "<|eot_id|>":
                family = "llama"
            elif eos == "<|im_end|>":
                family = "qwen"
            elif eos == "<｜end▁of▁sentence｜>":
                family = "deepseek"
            elif "<|user|>" in chat_template and "<|assistant|>" in chat_template:
                family = "olmo"
            elif "<|start_header_id|>" in chat_template or "<|eot_id|>" in chat_template:
                family = "llama"
            elif "<|im_start|>" in chat_template:
                family = "qwen"
            elif "<｜begin▁of▁sentence｜>" in chat_template or "<｜Assistant｜>" in chat_template:
                family = "deepseek"
            else:
                # Neither eos_token nor chat_template gave an answer -- most
                # likely a base/template-less checkpoint (exactly the case a
                # missing chat_template on a real OLMo base/mid-train teacher
                # hits). Fall back to config.json's model_type, which is set
                # from the model's actual architecture regardless of
                # instruction-tuning status.
                family = self._family_from_model_type(tokenizer)
            self._family_cache[tok_id] = family
        return self._family_cache[tok_id]

    @staticmethod
    def _family_from_model_type(tokenizer) -> str:
        """Fall back to config.json's model_type/architectures for a checkpoint
        whose tokenizer gave no eos_token/chat_template signal. Best-effort:
        returns "unknown" on any failure (offline, gated repo, no matching
        config alongside a bare tokenizer, exotic model_type) rather than
        raising, since this is already the last-resort branch.
        """
        name_or_path = getattr(tokenizer, "name_or_path", None)
        if not name_or_path:
            return "unknown"
        try:
            config = AutoConfig.from_pretrained(name_or_path, trust_remote_code=True)
        except Exception as exc:  # noqa: BLE001 - best-effort fallback, never fatal here.
            warnings.warn(
                f"[_detect_model_family] Could not load AutoConfig for {name_or_path!r} "
                f"to resolve model_type fallback: {type(exc).__name__}: {exc}"
            )
            return "unknown"
        model_type = str(getattr(config, "model_type", "") or "").lower()
        family = _MODEL_TYPE_TO_FAMILY.get(model_type, "unknown")
        if family == "unknown":
            warnings.warn(
                f"[_detect_model_family] {name_or_path!r} has no eos_token/chat_template "
                f"match and unrecognized model_type={model_type!r}; treating as unknown. "
                f"Add it to _MODEL_TYPE_TO_FAMILY if this checkpoint is a known family."
            )
        return family

    @staticmethod
    def _tokenizer_family_evidence(tokenizer):
        """Return the exact vocabulary markers used by family detection plus OLMo probes."""
        vocab = tokenizer.get_vocab()
        probes = (
            "<|begin_of_text|>",
            "<|eot_id|>",
            "<|im_start|>",
            "<|im_end|>",
            "<｜begin▁of▁sentence｜>",
            "<｜end▁of▁sentence｜>",
            "<|user|>",
            "<|assistant|>",
            "<|system|>",
            "<|endoftext|>",
        )
        return {token: vocab[token] for token in probes if token in vocab}

    @staticmethod
    def _compact_template(template, max_chars=800):
        if not template:
            return "<none>"
        template = str(template).replace("\n", "\\n")
        return template[:max_chars] + ("..." if len(template) > max_chars else "")

    def _print_tokenizer_debug_once(self):
        """Print tokenizer-family detection and selected mapping without changing either."""
        if self._tokenizer_debug_printed:
            return
        enabled = os.environ.get("OPD_TOKENIZER_DEBUG", "0").strip().lower() in {
            "1", "true", "yes", "on"
        }
        if not enabled:
            return
        self._tokenizer_debug_printed = True

        same_tokenizer = self._is_same_tokenizer()
        print("\n=== OPD tokenizer / mapping detection ===", flush=True)
        print(f"[mode] R1_ZERO_MODE={R1_ZERO_MODE}", flush=True)
        for label, tokenizer in (("student", self.student_tokenizer), ("teacher", self.tokenizer)):
            print(f"[{label}] name_or_path={tokenizer.name_or_path}", flush=True)
            print(f"[{label}] class={tokenizer.__class__.__name__}", flush=True)
            print(f"[{label}] detected_family={self._detect_model_family(tokenizer)}", flush=True)
            print(
                f"[{label}] detection_markers="
                f"{json.dumps(self._tokenizer_family_evidence(tokenizer), ensure_ascii=False)}",
                flush=True,
            )
            print(
                f"[{label}] bos={tokenizer.bos_token!r}({tokenizer.bos_token_id}) "
                f"eos={tokenizer.eos_token!r}({tokenizer.eos_token_id}) "
                f"pad={tokenizer.pad_token!r}({tokenizer.pad_token_id})",
                flush=True,
            )
            print(
                f"[{label}] special_tokens_map="
                f"{json.dumps(tokenizer.special_tokens_map, ensure_ascii=False, default=str)}",
                flush=True,
            )
            print(
                f"[{label}] chat_template={self._compact_template(getattr(tokenizer, 'chat_template', None))}",
                flush=True,
            )
            try:
                rendered_probe = tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": "You are a concise assistant."},
                        {"role": "user", "content": "What is 1+1?"},
                    ],
                    add_generation_prompt=True,
                    tokenize=False,
                )
            except Exception as exc:
                rendered_probe = f"<render error: {type(exc).__name__}: {exc}>"
            print(
                f"[{label}] rendered_probe={self._compact_template(rendered_probe)}",
                flush=True,
            )

        print(f"[mapping] vocabularies_identical={same_tokenizer}", flush=True)
        if same_tokenizer:
            print("[mapping] selected_route=same_tokenizer_direct_shift", flush=True)
        else:
            student_family = self._detect_model_family(self.student_tokenizer)
            teacher_family = self._detect_model_family(self.tokenizer)
            print(f"[mapping] selected_route={student_family}_to_{teacher_family}", flush=True)
            try:
                mapping = self._build_chat_template_mapping()
            except NotImplementedError as exc:
                print(f"[mapping] supported=False error={exc}", flush=True)
            else:
                print("[mapping] supported=True replacement_rules:", flush=True)
                for old, new in mapping:
                    print(f"  {old!r} -> {new!r}", flush=True)
        print("=== End OPD tokenizer / mapping detection ===\n", flush=True)

    def _build_chat_template_mapping(self):
        """Build string replacement rules from student chat template to teacher chat template.

        Currently supports:
            - LLaMA 3.x (student) -> Qwen (teacher)
            - LLaMA 3.x (student) -> DeepSeek (teacher)
            - LLaMA 3.x (student) -> OLMo (teacher)
            - Qwen (student) -> DeepSeek (teacher)
            - Qwen (student) -> LLaMA 3.x (teacher) [newly added, not yet validated end-to-end]
            - Qwen (student) -> Qwen (teacher) [same chat template, no mapping needed]
            - DeepSeek (student) -> DeepSeek (teacher) [same chat template, no mapping needed]
            - OLMo (student) -> OLMo (teacher) [same chat template, no mapping needed]

        The mapping is a list of (old_str, new_str) tuples applied in order via str.replace().
        """
        student_family = self._detect_model_family(self.student_tokenizer)
        teacher_family = self._detect_model_family(self.tokenizer)

        if student_family == "llama" and teacher_family == "qwen":
            # LLaMA 3.x -> Qwen
            # LLaMA format:  <|begin_of_text|><|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>
            # Qwen format:   <|im_start|>{role}\n{content}<|im_end|>\n
            return [
                # Must replace compound patterns first (longer -> shorter)
                ("<|begin_of_text|><|start_header_id|>", "<|im_start|>"),
                ("<|start_header_id|>", "<|im_start|>"),
                ("<|end_header_id|>\n\n", "\n"),
                ("<|eot_id|>", "<|im_end|>\n"),
                # Sequence-level tokens
                ("<|begin_of_text|>", ""),
                ("<|end_of_text|>", "<|im_end|>\n"),
            ]
        elif student_family == "llama" and teacher_family == "deepseek":
            # LLaMA 3.x (student) -> DeepSeek (teacher)
            # LLaMA format:  <|begin_of_text|><|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>
            # DeepSeek format: <｜begin▁of▁sentence｜><｜User｜>{content}<｜Assistant｜>{response}<｜end▁of▁sentence｜>
            # With system:     <｜begin▁of▁sentence｜>{system}<｜User｜>{content}<｜Assistant｜>{response}<｜end▁of▁sentence｜>
            return [
                # Handle system role first (longest compound patterns first)
                ("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n", "<｜begin▁of▁sentence｜>"),
                # Handle user role
                ("<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n", "<｜begin▁of▁sentence｜><｜User｜>"),
                ("<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n", "<｜User｜>"),
                # Handle assistant role
                ("<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", "<｜Assistant｜>"),
                # Handle end tokens
                ("<|eot_id|>", "<｜end▁of▁sentence｜>"),
                ("<|begin_of_text|>", "<｜begin▁of▁sentence｜>"),
                ("<|end_of_text|>", "<｜end▁of▁sentence｜>"),
            ]
        elif student_family == "qwen" and teacher_family == "deepseek":
            # Qwen (student) -> DeepSeek (teacher)
            # Qwen format:    <|im_start|>{role}\n{content}<|im_end|>\n
            # DeepSeek format: <｜begin▁of▁sentence｜><｜User｜>{content}<｜Assistant｜>{response}<｜end▁of▁sentence｜>
            # With system:     <｜begin▁of▁sentence｜>{system}<｜User｜>{content}<｜Assistant｜>{response}<｜end▁of▁sentence｜>
            return [
                # Handle system role (if present)
                ("<|im_start|>system\n", "<｜begin▁of▁sentence｜>"),
                # Handle user role (compound patterns first)
                ("<|im_end|>\n<|im_start|>user\n", "<｜User｜>"),
                ("<|im_start|>user\n", "<｜begin▁of▁sentence｜><｜User｜>"),
                # Handle assistant role (compound pattern first)
                ("<|im_end|>\n<|im_start|>assistant\n", "<｜Assistant｜>"),
                # Handle remaining end tokens (with trailing newline first)
                ("<|im_end|>\n", "<｜end▁of▁sentence｜>"),
                ("<|im_end|>", "<｜end▁of▁sentence｜>"),
                ("<|endoftext|>", "<｜end▁of▁sentence｜>"),
            ]
        elif student_family == "qwen" and teacher_family == "llama":
            # Qwen (student) -> LLaMA 3.x (teacher)
            # Qwen format:  <|im_start|>{role}\n{content}<|im_end|>\n
            # LLaMA format: <|begin_of_text|><|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>
            #
            # NOTE: this is newly added and not yet validated end-to-end (no LLaMA teacher
            # checkpoint was available to test against). Before trusting it for real training,
            # inspect a few sequences via OPD_DUMP_DIR / tools/opd_inspect_tokenizer_pair.py and
            # confirm the reconstructed teacher_text round-trips to the same conversation as
            # student_text.
            return [
                # Mid-conversation turns are preceded by the prior turn's "<|im_end|>\n" - match
                # those compound patterns FIRST so the bare "<|im_start|>{role}\n" rules below can
                # only match what's left over: the true first turn (which needs <|begin_of_text|>
                # prepended, mirroring LLaMA's own format).
                ("<|im_end|>\n<|im_start|>system\n", "<|eot_id|><|start_header_id|>system<|end_header_id|>\n\n"),
                ("<|im_end|>\n<|im_start|>user\n", "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"),
                ("<|im_end|>\n<|im_start|>assistant\n", "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"),
                # First-turn (leftover, not preceded by "<|im_end|>\n"):
                ("<|im_start|>system\n", "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"),
                ("<|im_start|>user\n", "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"),
                ("<|im_start|>assistant\n", "<|begin_of_text|><|start_header_id|>assistant<|end_header_id|>\n\n"),
                # Any remaining turn-close marker (e.g. the trailing generation-stop token).
                ("<|im_end|>\n", "<|eot_id|>"),
                ("<|im_end|>", "<|eot_id|>"),
                ("<|endoftext|>", "<|end_of_text|>"),
            ]
        elif student_family == "qwen" and teacher_family == "qwen":
            # Qwen (student) -> Qwen (teacher): same chat template format,
            # no string replacement needed. Re-tokenization handles vocab differences.
            return []
        elif student_family == "deepseek" and teacher_family == "deepseek":
            # DeepSeek (student) -> DeepSeek (teacher): same chat template format
            # (both use <｜User｜>, <｜Assistant｜>, <｜end▁of▁sentence｜>).
            # No string replacement needed. Re-tokenization handles vocab differences.
            return []
        elif student_family == "llama" and teacher_family == "olmo":
            # LLaMA 3.x (student) -> OLMo (teacher)
            # LLaMA format: <|begin_of_text|><|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>
            # OLMo (Tulu) format (verified against allenai/OLMo-2-1124-7B-Instruct
            # tokenizer_config.json chat_template, 2026-09-15):
            #   {bos}<|system|>\n{content}\n<|user|>\n{content}\n<|assistant|>\n{content}{eos}
            # where bos == eos == "<|endoftext|>". Turns are separated by a single "\n"
            # already emitted at the end of the previous turn's content -- there is no
            # blank line between turns like LLaMA's "<|end_header_id|>\n\n".
            return [
                # Must replace compound patterns first (longer -> shorter).
                # First-turn system / user (preceded by <|begin_of_text|>, not <|eot_id|>).
                ("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n", "<|endoftext|><|system|>\n"),
                ("<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n", "<|endoftext|><|user|>\n"),
                # Mid-conversation turns (preceded by the prior turn's <|eot_id|>).
                ("<|eot_id|><|start_header_id|>system<|end_header_id|>\n\n", "\n<|system|>\n"),
                ("<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n", "\n<|user|>\n"),
                ("<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", "\n<|assistant|>\n"),
                # End-of-turn / end-of-sequence markers -> OLMo's single eos token.
                ("<|eot_id|>", "<|endoftext|>\n"),
                ("<|begin_of_text|>", "<|endoftext|>"),
                ("<|end_of_text|>", "<|endoftext|>"),
            ]
        elif student_family == "olmo" and teacher_family == "olmo":
            # OLMo (student) -> OLMo (teacher): same chat template format,
            # no string replacement needed. Re-tokenization handles vocab differences.
            return []
        else:
            raise NotImplementedError(
                f"Unsupported cross-tokenizer pair: student has "
                f"'{student_family}' vocab, "
                f"teacher has '{teacher_family}' vocab. "
                f"Please add a mapping in _build_chat_template_mapping()."
            )

    def _swap_bos_eos_for_teacher(self, student_text: str) -> str:
        """R1_ZERO_MODE only: swap the student tokenizer's own bos/eos strings
        for the teacher tokenizer's, leaving everything else (the raw
        r1_zero prompt text and the model's own response content) untouched.

        Safe to call repeatedly (memoizes the resolved token strings on self).
        """
        if not hasattr(self, '_raw_mode_bos_eos'):
            student_bos = self.student_tokenizer.bos_token or ""
            teacher_bos = self.tokenizer.bos_token or ""
            # Collect every string the student tokenizer might emit to mean
            # "stop" -- eos_token plus any additional special tokens whose
            # content looks like a generic end marker. Most checkpoints only
            # ever use eos_token itself; this is defensive, not load-bearing.
            student_eos_candidates = []
            if self.student_tokenizer.eos_token:
                student_eos_candidates.append(self.student_tokenizer.eos_token)
            for extra in getattr(self.student_tokenizer, "additional_special_tokens", None) or []:
                if extra not in student_eos_candidates and extra != student_bos:
                    student_eos_candidates.append(extra)
            teacher_eos = self.tokenizer.eos_token or ""
            self._raw_mode_bos_eos = (student_bos, teacher_bos, student_eos_candidates, teacher_eos)

        student_bos, teacher_bos, student_eos_candidates, teacher_eos = self._raw_mode_bos_eos
        teacher_text = student_text
        if student_bos and teacher_text.startswith(student_bos):
            teacher_text = teacher_bos + teacher_text[len(student_bos):]
        elif teacher_bos:
            teacher_text = teacher_bos + teacher_text
        if teacher_eos:
            for tok in student_eos_candidates:
                if tok and tok != teacher_eos:
                    teacher_text = teacher_text.replace(tok, teacher_eos)
        return teacher_text

    def retokenize_batch(self, batch):
        """Convert a batch of student token ID sequences to teacher token ID sequences.

        Pipeline: student token IDs -> decode to text (with special tokens) ->
                  string-replace chat template markers -> encode with teacher tokenizer.

        Args:
            batch: list of list[int], each inner list is a sequence of student token IDs.

        Returns:
            list of list[int], each inner list is the re-encoded sequence in teacher token IDs.
        """
        if self._is_same_tokenizer():
            return batch

        if not R1_ZERO_MODE and not hasattr(self, '_template_mapping'):
            self._template_mapping = self._build_chat_template_mapping()

        input_id_list = []

        for seq_idx in range(len(batch)):
            student_text = self.student_tokenizer.decode(batch[seq_idx], skip_special_tokens=False)

            if R1_ZERO_MODE:
                # Raw-prompt mode: the prompt content is identical literal text
                # for both tokenizers (no role markers to translate). The only
                # difference is each tokenizer's own bos-string prefix and its
                # own natural eos string -- swap those directly instead of
                # using the family-based chat-template mapping table, which
                # assumes role markers that don't exist in this mode.
                teacher_text = self._swap_bos_eos_for_teacher(student_text)
            else:
                teacher_text = student_text
                for old_str, new_str in self._template_mapping:
                    teacher_text = teacher_text.replace(old_str, new_str)

                # Strip trailing \n that may arise from template mapping:
                # - LLaMA->Qwen: <|eot_id|> -> <|im_end|>\n leaves a trailing \n
                # - LLaMA->DeepSeek: <|eot_id|> -> <｜end▁of▁sentence｜> (no trailing \n, rstrip harmless)
                # Only strip if the original student text ended with an eos/eot special token
                # (meaning the \n came from mapping, not from the actual content).
                # NOTE: For Qwen->DeepSeek, the mapping handles <|im_end|>\n -> <｜end▁of▁sentence｜>
                # atomically, so no trailing \n remains. For same-template pairs (Qwen->Qwen,
                # DeepSeek->DeepSeek), text is unchanged so no stripping is needed.
                student_family = self._detect_model_family(self.student_tokenizer)
                if student_family == "llama":
                    student_has_eos = any(
                        student_text.endswith(tok)
                        for tok in ["<|eot_id|>", "<|end_of_text|>"]
                    )
                    if student_has_eos:
                        teacher_text = teacher_text.rstrip('\n')
                elif student_family == "qwen":
                    # Qwen->Qwen: no mapping, text unchanged, no trailing \n issue.
                    # Qwen->DeepSeek: <|im_end|>\n -> <｜end▁of▁sentence｜> already clean.
                    # Qwen->LLaMA: <|im_end|>\n -> <|eot_id|> already clean (no trailing \n added).
                    pass
                elif student_family == "deepseek":
                    # DeepSeek->DeepSeek: same chat template format, no conversion,
                    # no trailing \n issue. Text is passed through unchanged.
                    pass

            new_input_id = self.tokenizer(teacher_text, add_special_tokens=False)['input_ids']
            if self.max_seq_len:
                # The teacher API generates at least one token so that vLLM returns
                # prompt log-probabilities.  Keep that generation token inside the
                # configured sequence budget; truncating to max_seq_len itself would
                # make bg_task send max_tokens=0 and abort the whole training run.
                reserved_generation_tokens = max(int(self.max_tokens), 1)
                max_input_len = self.max_seq_len - reserved_generation_tokens
                if max_input_len < 1:
                    raise ValueError(
                        f"max_seq_len ({self.max_seq_len}) must be greater than the "
                        f"reserved teacher generation tokens ({reserved_generation_tokens})"
                    )

            if self.max_seq_len and len(new_input_id) > max_input_len:
                warnings.warn(
                    f"[retokenize] seq {seq_idx}: teacher tokens ({len(new_input_id)}) exceed "
                    f"max input length ({max_input_len}; max_seq_len={self.max_seq_len}), "
                    f"will skip gradient for this sequence. "
                    f"Student tokens: {len(batch[seq_idx])}, inflation ratio: {len(new_input_id)/len(batch[seq_idx]):.2f}x"
                )
                new_input_id = new_input_id[:max_input_len]
            input_id_list.append(new_input_id)

        return input_id_list

    @staticmethod
    def _align_chunks(student_ids, teacher_ids, teacher_logps,
                      student_tokenizer, teacher_tokenizer, debug=False,
                      large_chunk_threshold=6):
        """Align teacher logprobs to student token granularity via chunk-level greedy matching.

        Both student_ids and teacher_ids should be the *response content only*
        (no chat template special tokens on either side).

        Algorithm:
            Maintain two pointers (s_ptr into student_ids, t_ptr into teacher_ids).
            Greedily expand both sides by decoding accumulated tokens to text.
            When the decoded texts match, a chunk boundary is found:
                teacher_logprob_for_chunk = sum(teacher_logps[t_start:t_ptr])
                each student token in the chunk gets teacher_logprob_for_chunk / n_student_tokens

        Args:
            student_ids: list[int], student response token IDs
            teacher_ids: list[int], teacher response token IDs
            teacher_logps: Tensor [len(teacher_ids)], teacher logprob for each teacher response token
            student_tokenizer: student's tokenizer
            teacher_tokenizer: teacher's tokenizer
            debug: bool, if True print alignment visualization
            large_chunk_threshold: int, chunks with n_stu or n_tec exceeding this
                threshold are marked with inf sentinel (unreliable alignment from
                consecutive replacement characters). Default=6.

        Returns:
            (aligned, chunk_ids, chunk_details, teacher_truncated) where:
                aligned: Tensor [len(student_ids)], aligned logprobs for each student token
                chunk_ids: Tensor [len(student_ids)], same ID for student tokens in a synchronized chunk
                chunk_details: list of dicts with per-chunk info (only populated when debug=True)
                teacher_truncated: bool, True if teacher ran out before student
        """
        n_stu = len(student_ids)
        n_tec = len(teacher_ids)
        aligned = torch.zeros(n_stu, dtype=teacher_logps.dtype)
        chunk_ids = torch.full((n_stu,), -1, dtype=torch.float32)

        chunk_details = []  # per-chunk info for debug
        teacher_truncated = False  # True iff teacher tokens ran out while student still had content

        if n_stu == 0 or n_tec == 0:
            return aligned, chunk_ids, chunk_details, teacher_truncated

        s_ptr = 0  # next student token to consume
        t_ptr = 0  # next teacher token to consume
        _chunk_count = 0
        _one_to_one = 0      # chunks where 1 student token <-> 1 teacher token
        _multi_chunks = 0    # chunks where >1 tokens on either side
        _multi_stu_tokens = 0  # student tokens involved in multi-token chunks
        _multi_tec_tokens = 0  # teacher tokens involved in multi-token chunks
        _multi_chunk_details = []  # collect multi-chunk details for deferred printing

        while s_ptr < n_stu and t_ptr < n_tec:
            # Grow a chunk from both sides until decoded texts match
            s_end = s_ptr + 1
            t_end = t_ptr + 1
            matched = False

            while s_end <= n_stu and t_end <= n_tec:
                s_text = unicodedata.normalize('NFC', student_tokenizer.decode(student_ids[s_ptr:s_end], skip_special_tokens=False))
                t_text = unicodedata.normalize('NFC', teacher_tokenizer.decode(teacher_ids[t_ptr:t_end], skip_special_tokens=False))

                if s_text == t_text and not s_text.endswith('\ufffd'):
                    # Chunk boundary found
                    n_stu_tokens = s_end - s_ptr
                    n_tec_tokens = t_end - t_ptr
                    chunk_logp = teacher_logps[t_ptr:t_end].sum()
                    # Mark large chunks as inf sentinel \u2014 these are typically
                    # consecutive U+FFFD (garbled output) runs where the averaged
                    # teacher logprob is unreliable. Treated same as fallback:
                    # advantage = 0 \u2192 no gradient on these positions.
                    if n_stu_tokens > large_chunk_threshold or n_tec_tokens > large_chunk_threshold:
                        aligned[s_ptr:s_end] = float('inf')
                    else:
                        aligned[s_ptr:s_end] = chunk_logp / n_stu_tokens
                        chunk_ids[s_ptr:s_end] = _chunk_count
                    # if debug and not (n_stu_tokens == 1 and (t_end - t_ptr) == 1):
                    #     s_repr = repr(s_text)
                    #     print(f"  [multi-chunk {_chunk_count}] stu[{s_ptr}:{s_end}]({n_stu_tokens}t) <-> tec[{t_ptr}:{t_end}]({t_end-t_ptr}t)  "
                    #           f"text={s_repr[:80]}  logp_sum={chunk_logp.item():.4f} -> avg={aligned[s_ptr].item():.4f}")
                    #     # Print each student token in this chunk
                    #     for si in range(s_ptr, s_end):
                    #         s_tok_text = repr(student_tokenizer.decode([student_ids[si]], skip_special_tokens=False))
                    #         print(f"    stu[{si}] id={student_ids[si]:>6d}  {s_tok_text}")
                    #     # Print each teacher token with its logprob
                    #     for ti in range(t_ptr, t_end):
                    #         t_tok_text = repr(teacher_tokenizer.decode([teacher_ids[ti]], skip_special_tokens=False))
                    #         t_logp = teacher_logps[ti].item()
                    #         print(f"    tec[{ti}] id={teacher_ids[ti]:>6d}  logp={t_logp:.4f}  {t_tok_text}")
                    _chunk_count += 1
                    if debug:
                        chunk_details.append({
                            "chunk_id": _chunk_count - 1,
                            "text": s_text,
                            "s_start": s_ptr, "s_end": s_end,
                            "t_start": t_ptr, "t_end": t_end,
                            "n_stu": n_stu_tokens, "n_tec": t_end - t_ptr,
                            "teacher_logp_sum": chunk_logp.item(),
                            "aligned_avg": aligned[s_ptr].item(),
                            "type": "matched",
                            "teacher_tokens": [
                                (teacher_ids[ti],
                                 teacher_logps[ti].item(),
                                 teacher_tokenizer.decode([teacher_ids[ti]], skip_special_tokens=False))
                                for ti in range(t_ptr, t_end)
                            ],
                            "student_tokens": [
                                (student_ids[si],
                                 student_tokenizer.decode([student_ids[si]], skip_special_tokens=False))
                                for si in range(s_ptr, s_end)
                            ],
                        })
                    if n_stu_tokens == 1 and (t_end - t_ptr) == 1:
                        _one_to_one += 1
                    else:
                        _multi_chunks += 1
                        _multi_stu_tokens += n_stu_tokens
                        _multi_tec_tokens += (t_end - t_ptr)
                        if debug:
                            _multi_chunk_details.append((
                                _chunk_count - 1, s_ptr, s_end, t_ptr, t_end,
                                s_text, chunk_logp.item(), aligned[s_ptr].item(),
                                [(si, student_ids[si], student_tokenizer.decode([student_ids[si]], skip_special_tokens=False)) for si in range(s_ptr, s_end)],
                                [(ti, teacher_ids[ti], teacher_logps[ti].item(), teacher_tokenizer.decode([teacher_ids[ti]], skip_special_tokens=False)) for ti in range(t_ptr, t_end)],
                            ))
                    s_ptr = s_end
                    t_ptr = t_end
                    matched = True
                    break

                # Track previous pointers to detect stuck state
                s_end_prev, t_end_prev = s_end, t_end

                if len(s_text) < len(t_text):
                    if s_end < n_stu:
                        s_end += 1
                    else:
                        t_end += 1
                elif len(s_text) > len(t_text):
                    if t_end < n_tec:
                        t_end += 1
                    else:
                        s_end += 1
                else:
                    s_incomplete = s_text.endswith('\ufffd')
                    t_incomplete = t_text.endswith('\ufffd')
                    if s_incomplete and not t_incomplete:
                        if s_end < n_stu:
                            s_end += 1
                        elif t_end < n_tec:
                            t_end += 1
                    elif t_incomplete and not s_incomplete:
                        if t_end < n_tec:
                            t_end += 1
                        elif s_end < n_stu:
                            s_end += 1
                    else:
                        if s_end < n_stu:
                            s_end += 1
                        if t_end < n_tec:
                            t_end += 1

                # If no pointer advanced, both sides are exhausted — break to fallback
                if s_end == s_end_prev and t_end == t_end_prev:
                    break

            if not matched:
                n_stu_tokens = n_stu - s_ptr
                n_tec_tokens = n_tec - t_ptr
                chunk_logp = teacher_logps[t_ptr:n_tec].sum()
                # Fallback: alignment got stuck. Instead of hard-averaging the
                # remaining teacher logp across student tokens (which injects
                # noisy/meaningless signal), mark these student positions with
                # inf sentinel. core_algos will replace inf with student logp
                # → advantage = 0 → no gradient on unaligned positions.
                if n_stu_tokens > 0:
                    aligned[s_ptr:n_stu] = float('inf')
                if debug:
                    fb_text = student_tokenizer.decode(student_ids[s_ptr:min(s_ptr+20, n_stu)], skip_special_tokens=False)
                    chunk_details.append({
                        "chunk_id": _chunk_count,
                        "text": fb_text + ("..." if n_stu_tokens > 20 else ""),
                        "s_start": s_ptr, "s_end": n_stu,
                        "t_start": t_ptr, "t_end": n_tec,
                        "n_stu": n_stu_tokens, "n_tec": n_tec_tokens,
                        "teacher_logp_sum": chunk_logp.item(),
                        "aligned_avg": float('inf'),  # marked as sentinel
                        "type": "fallback",
                        "teacher_tokens": [
                            (teacher_ids[ti],
                             teacher_logps[ti].item(),
                             teacher_tokenizer.decode([teacher_ids[ti]], skip_special_tokens=False))
                            for ti in range(t_ptr, min(t_ptr + 20, n_tec))
                        ],
                        "student_tokens": [
                            (student_ids[si],
                             student_tokenizer.decode([student_ids[si]], skip_special_tokens=False))
                            for si in range(s_ptr, min(s_ptr + 20, n_stu))
                        ],
                    })
                # if debug:
                #     print(f"  [FALLBACK] stu[{s_ptr}:{n_stu}]({n_stu_tokens}t) <-> tec[{t_ptr}:{n_tec}]({n_tec_tokens}t)  "
                #           f"logp={chunk_logp.item():.4f}")
                #     _fb_s_text = student_tokenizer.decode(student_ids[s_ptr:min(s_ptr+20, n_stu)], skip_special_tokens=False)
                #     _fb_t_text = teacher_tokenizer.decode(teacher_ids[t_ptr:min(t_ptr+20, n_tec)], skip_special_tokens=False)
                #     print(f"  [FALLBACK] stuck at: s_text={repr(_fb_s_text)[:120]}")
                #     print(f"  [FALLBACK] stuck at: t_text={repr(_fb_t_text)[:120]}")
                #     _fb_stu_limit = min(n_stu, s_ptr + 30)
                #     print(f"  [FALLBACK] student tokens [{s_ptr}:{_fb_stu_limit}] (showing first 30):")
                #     for si in range(s_ptr, _fb_stu_limit):
                #         s_tok_text = repr(student_tokenizer.decode([student_ids[si]], skip_special_tokens=False))
                #         print(f"    stu[{si}] id={student_ids[si]:>6d}  {s_tok_text}")
                #     _fb_tec_limit = min(n_tec, t_ptr + 30)
                #     print(f"  [FALLBACK] teacher tokens [{t_ptr}:{_fb_tec_limit}] (showing first 30):")
                #     for ti in range(t_ptr, _fb_tec_limit):
                #         t_tok_text = repr(teacher_tokenizer.decode([teacher_ids[ti]], skip_special_tokens=False))
                #         t_logp = teacher_logps[ti].item()
                #         print(f"    tec[{ti}] id={teacher_ids[ti]:>6d}  logp={t_logp:.4f}  {t_tok_text}")
                #     if n_stu_tokens > 30:
                #         print(f"  [FALLBACK] student tokens (last 10):")
                #         for si in range(max(s_ptr, n_stu - 10), n_stu):
                #             s_tok_text = repr(student_tokenizer.decode([student_ids[si]], skip_special_tokens=False))
                #             print(f"    stu[{si}] id={student_ids[si]:>6d}  {s_tok_text}")
                #     if n_tec_tokens > 30:
                #         print(f"  [FALLBACK] teacher tokens (last 10):")
                #         for ti in range(max(t_ptr, n_tec - 10), n_tec):
                #             t_tok_text = repr(teacher_tokenizer.decode([teacher_ids[ti]], skip_special_tokens=False))
                #             t_logp = teacher_logps[ti].item()
                #             print(f"    tec[{ti}] id={teacher_ids[ti]:>6d}  logp={t_logp:.4f}  {t_tok_text}")
                s_ptr = n_stu
                t_ptr = n_tec
                break

        # Teacher tokens exhausted but student tokens remain (truncated sequences).
        # Mark with inf sentinel → core_algos replaces with student logprob → advantage = 0.
        # Also set teacher_truncated flag so caller can decide to skip whole seq.
        # NOTE: fallback branch sets s_ptr = n_stu explicitly, so reaching here means
        # the while loop ended naturally with teacher exhausted.
        if s_ptr < n_stu:
            aligned[s_ptr:n_stu] = float('inf')
            teacher_truncated = True
            # if debug:
            #     print(f"  [TRUNCATED] stu[{s_ptr}:{n_stu}]({n_stu-s_ptr}t) marked as inf sentinel")

        if debug:
            _matched_stu = _one_to_one + _multi_stu_tokens
            _fallback_stu = n_stu - s_ptr if s_ptr < n_stu else 0  # already set above
            _truncated_stu = n_stu - s_ptr  # tokens marked inf (0 if fully aligned)
            # s_ptr after loop = number of student tokens that were matched or fallback'd
            # but we need to distinguish: matched via chunks vs fallback vs truncated
            print(f"  [SUMMARY] {n_stu} student tokens, {n_tec} teacher tokens")
            print(f"  [STATS] total chunks: {_chunk_count}, "
                  f"1:1 chunks: {_one_to_one} ({_one_to_one} stu tokens), "
                  f"multi-token chunks: {_multi_chunks} ({_multi_stu_tokens} stu tokens, {_multi_tec_tokens} tec tokens)")
            print(f"  [STATS] chunk-matched: {_matched_stu}/{n_stu} stu tokens ({_matched_stu/n_stu*100:.1f}%), "
                  f"of which 1:1: {_one_to_one}/{_matched_stu} ({_one_to_one/_matched_stu*100:.1f}% if matched>0)" if _matched_stu > 0 else
                  f"  [STATS] chunk-matched: 0/{n_stu} stu tokens (0%)")
            if n_stu > _matched_stu:
                _rest = n_stu - _matched_stu
                print(f"  [STATS] NOT chunk-matched: {_rest}/{n_stu} stu tokens ({_rest/n_stu*100:.1f}%) — fallback or truncation")
            # Print multi-chunk details when 1:1 ratio < 90%
            if _chunk_count > 0 and _one_to_one / _chunk_count < 0.9 and _multi_chunk_details:
                # Filter out pure digit cases (1 stu token of digits <-> N tec tokens of single digits)
                _non_digit = []
                for detail in _multi_chunk_details:
                    (cidx, sp, se, tp, te, text, logp, avg, stu_toks, tec_toks) = detail
                    if text.strip().isdigit():
                        continue  # skip pure number multi-chunks
                    _non_digit.append(detail)
                if _non_digit:
                    print(f"  [LOW 1:1] printing {len(_non_digit)} non-digit multi-token chunks (filtered {len(_multi_chunk_details)-len(_non_digit)} digit chunks):")
                    for (cidx, sp, se, tp, te, text, logp, avg, stu_toks, tec_toks) in _non_digit:
                        n_s = se - sp
                        n_t = te - tp
                        print(f"    [multi-chunk {cidx}] stu[{sp}:{se}]({n_s}t) <-> tec[{tp}:{te}]({n_t}t)  "
                              f"text={repr(text)[:80]}  logp_sum={logp:.4f} -> avg={avg:.4f}")
                        for si, sid, stxt in stu_toks:
                            print(f"      stu[{si}] id={sid:>6d}  {repr(stxt)}")
                        for ti, tid, tlogp, ttxt in tec_toks:
                            print(f"      tec[{ti}] id={tid:>6d}  logp={tlogp:.4f}  {repr(ttxt)}")

        return aligned, chunk_ids, chunk_details, teacher_truncated
    def bg_task(self):
        socket = self.context.socket(zmq.REQ)
        socket.connect(f"tcp://{self.server_ip}:{self.server_port}")
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.RCVTIMEO, 1800000)  # 接收超时 30 分钟

        while True:
            futures = []
            inputs = []
            batch = []
            try:
                with self.mutex:
                    for _ in range(self.num_microbatches):
                        future, data = self.task_queue.get()
                        if DEBUG:
                            inputs.append(data)
                        futures.append(future)
                        batch.extend(data.tolist() if isinstance(data, torch.Tensor) else data)

                batch = self.retokenize_batch(batch)
                if self.max_seq_len:
                    # retokenize_batch applies this limit for cross-tokenizer
                    # requests. Apply it again here as a safety net for the
                    # same-tokenizer fast path, which returns before retokenization.
                    reserved_generation_tokens = max(int(self.max_tokens), 1)
                    max_input_len = self.max_seq_len - reserved_generation_tokens
                    if max_input_len < 1:
                        raise ValueError(
                            f"max_seq_len ({self.max_seq_len}) must be greater than the "
                            f"reserved teacher generation tokens ({reserved_generation_tokens})"
                        )
                    batch = [prompt[:max_input_len] for prompt in batch]
                    max_tokens = [
                        min(reserved_generation_tokens, self.max_seq_len - len(prompt))
                        for prompt in batch
                    ]
                    assert all(value >= 1 for value in max_tokens), (
                        f"Teacher request must reserve at least one generation token, got {max_tokens}"
                    )
                    request = {"prompt_token_ids": batch, "max_tokens": max_tokens}
                else:
                    request = {"prompt_token_ids": batch, "max_tokens": self.max_tokens}
                if self.temperature:
                    request["temperature"] = self.temperature
                if self.only_response:
                    request["only_response"] = True

                socket.send(serialize(request))
                raw = socket.recv()
                response = deserialize(raw)

                if isinstance(response, dict) and response.get("status") == "error":
                    reason = response.get("reason", "unknown")
                    err = RuntimeError(f"Teacher error: {reason}")
                    for f in futures:
                        f.set_exception(err)
                    continue

                required = ("responses", "teacher_topk_logprobs", "teacher_topk_indices")
                for k in required:
                    if k not in response:
                        raise RuntimeError(f"Invalid response: missing key '{k}'")

                total = len(response["teacher_topk_logprobs"])
                if self.num_microbatches <= 0 or total % self.num_microbatches != 0:
                    raise RuntimeError(f"Size mismatch: total={total}, num_microbatches={self.num_microbatches}")

                mbs = total // self.num_microbatches
                for i, future in enumerate(futures):
                    s, e = i * mbs, (i + 1) * mbs
                    responses = response["responses"][s:e]
                    teacher_topk_logps = response["teacher_topk_logprobs"][s:e]
                    if DEBUG:
                        check_if_invalid(teacher_topk_logps, inputs[i])
                    teacher_topk_indices = response["teacher_topk_indices"][s:e]
                    future.set_result((responses, teacher_topk_logps, teacher_topk_indices))

            except zmq.Again:
                err = TimeoutError(f"Timeout waiting for server {self.server_ip}:{self.server_port}")
                for f in futures:
                    f.set_exception(err)
                continue
            except Exception as e:
                for f in futures:
                    try:
                        f.set_exception(e)
                    except Exception:
                        pass
                continue

    def _run(self):
        for _ in range(self.n_server_workers):
            threading.Thread(target=self.bg_task, daemon=True).start()

    def submit(self, data):
        future = Future()
        self.task_queue.put((future, data))
        return future

    def __del__(self):
        self.context.destroy()

    def get_teacher_knowledge(self, batch: DataProto, is_async=False, student_tokenizer=None):
        """
        Retrieve teacher model's top-k predictions and log probabilities for knowledge distillation.

        Args:
            batch (DataProto): Input batch containing input_ids and attention_mask
            is_async (bool): Whether to use asynchronous processing

        Returns:
            If is_async=True: SimpleNamespace with get() method to process futures
            If is_async=False: Processed DataProto containing teacher knowledge

        Raises:
            RuntimeError: If teacher model request fails
        """

        assert student_tokenizer is not None, "To get knowledge of teacher, tokenizer of student must be passed"
        self.student_tokenizer = student_tokenizer
        self._print_tokenizer_debug_once()
        input_ids = []
        attention_mask = batch.batch["attention_mask"].to(torch.bool)
        # response_length = batch.meta_info["response_length"]

        for ids, mask in zip(batch.batch["input_ids"], attention_mask, strict=False):
            input_ids.append(ids[mask].tolist())

        all_teacher_topk_logps = []
        all_teacher_topk_indices = []
        responses = []

        batch_size = len(input_ids)
        assert batch_size % self.n_server_workers == 0
        micro_batch_size = batch_size // self.n_server_workers
        futures = []
        tik1 = time.time()
        tok1 = tik1

        def cb(future):
            nonlocal tok1
            tok1 = max(tok1, time.time())

        for i in range(0, batch_size, micro_batch_size):
            fut = self.submit(input_ids[i : i + micro_batch_size])
            fut.add_done_callback(cb)
            futures.append(fut)

        def handle_futures():
            MAX_RETRIES = 3
            retry_count = 0
            while retry_count <= MAX_RETRIES:
                try:
                    for future in futures:
                        response, teacher_topk_logps, teacher_topk_indices = future.result()
                        all_teacher_topk_logps.extend(teacher_topk_logps)
                        all_teacher_topk_indices.extend(teacher_topk_indices)
                        responses.extend(response)
                    break  # success
                except Exception as e:
                    retry_count += 1
                    if retry_count > MAX_RETRIES:
                        raise RuntimeError(f"Teacher request failed after {MAX_RETRIES} retries: {e}") from e
                    print(f"[WARNING] Teacher request failed (attempt {retry_count}/{MAX_RETRIES}): {e}. Retrying...")
                    import time as _time
                    _time.sleep(5)
                    # Clear and resubmit
                    all_teacher_topk_logps.clear()
                    all_teacher_topk_indices.clear()
                    responses.clear()
                    futures.clear()
                    for i in range(0, batch_size, micro_batch_size):
                        fut = self.submit(input_ids[i : i + micro_batch_size])
                        fut.add_done_callback(cb)
                        futures.append(fut)

            tik2 = time.time()
            # teacher_topk_logps = [x.to(params_dtype) for x in all_teacher_topk_logps]
            # teacher_topk_indices = [x.to(params_dtype) for x in all_teacher_topk_indices]
            teacher_topk_logps, teacher_topk_indices = all_teacher_topk_logps, all_teacher_topk_indices

            real_seq_lens = torch.tensor([x.size(0) for x in teacher_topk_logps], dtype=torch.int32)

            topk = teacher_topk_logps[0].size(-1)

            logp_dtype = teacher_topk_logps[0].dtype
            idx_dtype = teacher_topk_indices[0].dtype
            # teacher_knowledge_shape = list(batch.batch["input_ids"].shape) + [topk]
            teacher_knowledge_shape = list(batch.batch["input_ids"].shape)

            global teacher_topk_logps_padded, teacher_topk_indices_padded, teacher_chunk_ids_padded
            if (
                teacher_topk_logps_padded is None
                or teacher_topk_logps_padded.dtype != logp_dtype
                or teacher_topk_logps_padded.shape != torch.Size(teacher_knowledge_shape)
            ):
                teacher_topk_logps_padded = torch.zeros(*teacher_knowledge_shape, dtype=logp_dtype)
            else:
                teacher_topk_logps_padded.zero_()

            if (
                teacher_topk_indices_padded is None
                or teacher_topk_indices_padded.dtype != idx_dtype
                or teacher_topk_indices_padded.shape != torch.Size(teacher_knowledge_shape)
            ):
                teacher_topk_indices_padded = torch.zeros(*teacher_knowledge_shape, dtype=idx_dtype)
            else:
                teacher_topk_indices_padded.zero_()

            if (
                teacher_chunk_ids_padded is None
                or teacher_chunk_ids_padded.shape != torch.Size(teacher_knowledge_shape)
            ):
                teacher_chunk_ids_padded = torch.full(teacher_knowledge_shape, -1.0, dtype=torch.float32)
            else:
                teacher_chunk_ids_padded.fill_(-1.0)

            # NOTE: named distinctly from the outer `batch_size` (get_teacher_knowledge scope) -
            # reusing that name here would make Python treat `batch_size` as local to this whole
            # nested function (handle_futures), which breaks the retry loop above: it reads
            # `batch_size` before this line ever runs, raising UnboundLocalError on any retry.
            local_batch_size = attention_mask.size(0)

            # ---- Dump control ----
            global _dump_step_counter
            should_dump = bool(OPD_DUMP_DIR) and _dump_step_counter < OPD_DUMP_MAX_STEPS
            if should_dump:
                current_dump_step = _dump_step_counter
                _dump_step_counter += 1
                dump_num = min(OPD_DUMP_NUM_SEQS, local_batch_size)
            else:
                dump_num = 0

            if self._is_same_tokenizer():
                # Same tokenizer: direct fill, no alignment needed.
                # teacher_topk_logps[i][k, 0] predicts token at position k+1,
                # so we shift right by 1: fill [1:] logprobs into positions [1:].
                # This way position j holds the logprob predicting token j,
                # and core_algos can use [-R:] directly (no -1 offset).
                for i in range(local_batch_size):
                    valid_pos = attention_mask[i].nonzero(as_tuple=True)[0]
                    logps = teacher_topk_logps[i][:, 0]  # [N], predicts token 1..N
                    # logps[0] predicts token 1, fill at valid_pos[1]
                    # logps[N-1] predicts token N (doesn't exist / last+1), skip it
                    n_fill = min(len(logps) - 1, len(valid_pos) - 1)
                    teacher_topk_logps_padded[i, valid_pos[1:1+n_fill]] = logps[:n_fill]
                    teacher_chunk_ids_padded[i, valid_pos[1:1+n_fill]] = torch.arange(
                        n_fill, dtype=torch.float32
                    )

                    # Dump for same-tokenizer case
                    if i < dump_num:
                        student_tokenizer = self.student_tokenizer
                        stu_ids = input_ids[i]
                        tch_ids = responses[i].tolist()
                        _dump_alignment(current_dump_step, i, {
                            "mode": "same_tokenizer",
                            "student_family": self._detect_model_family(student_tokenizer),
                            "teacher_family": self._detect_model_family(self.tokenizer),
                            "student_text": student_tokenizer.decode(stu_ids, skip_special_tokens=False),
                            "teacher_text": self.tokenizer.decode(tch_ids, skip_special_tokens=False),
                            "student_token_count": len(stu_ids),
                            "teacher_token_count": len(tch_ids),
                            "student_tokens": [
                                {"id": tid, "text": student_tokenizer.decode([tid], skip_special_tokens=False)}
                                for tid in stu_ids[:100]  # first 100 for brevity
                            ],
                            "teacher_logps_sample": logps[:50].tolist(),
                        })
                return torch.cat(
                    [teacher_topk_logps_padded, teacher_chunk_ids_padded.to(teacher_topk_logps_padded.dtype)],
                    dim=-1,
                )

            # ============================================================
            # Cross-tokenizer alignment: align teacher logprobs to student
            # token granularity using chunk-level greedy matching.
            # ============================================================

            # We need:
            #   - input_ids[i]: student token IDs (prompt + response, no padding), list[int]
            #   - responses[i]: teacher token IDs returned by teacher server (prompt + response + 1 generated), Tensor
            #   - teacher_topk_logps[i]: teacher logprobs, shape [teacher_seq_len - 1, topk]
            #     (logprob at position j predicts token at position j+1 in the teacher sequence)

            teacher_tokenizer = self.tokenizer
            student_tokenizer = self.student_tokenizer

            if R1_ZERO_MODE:
                # No chat-template role markers exist to search for in this
                # mode -- response boundaries are derived directly from the
                # batch's own known prompt length (student side) and a
                # teacher-side re-encoding of the student's own prompt slice
                # (teacher side), not from a decoded-text marker search.
                student_family = teacher_family = "raw"
                prompt_length = batch.batch["prompts"].shape[-1]
            else:
                # Detect model families for response boundary markers
                student_family = self._detect_model_family(student_tokenizer)
                teacher_family = self._detect_model_family(teacher_tokenizer)

                # Precompute the response-start markers (as text) for locating response boundaries
                if student_family == "llama":
                    # LLaMA: ...assistant<|end_header_id|>\n\n{response}
                    student_resp_marker = "<|end_header_id|>\n\n"
                elif student_family == "qwen":
                    # Qwen: ...<|im_start|>assistant\n{response}
                    student_resp_marker = "<|im_start|>assistant\n"
                elif student_family == "deepseek":
                    # DeepSeek: ...<｜Assistant｜>{response}
                    student_resp_marker = "<｜Assistant｜>"
                elif student_family == "olmo":
                    # OLMo (Tulu): ...<|assistant|>\n{response}
                    student_resp_marker = "<|assistant|>\n"
                else:
                    raise NotImplementedError(f"Unsupported student model family: {student_family}")

                # Teacher response marker depends on teacher family
                if teacher_family == "qwen":
                    teacher_resp_marker = "<|im_start|>assistant\n"
                elif teacher_family == "deepseek":
                    teacher_resp_marker = "<｜Assistant｜>"
                elif teacher_family == "llama":
                    # LLaMA: ...assistant<|end_header_id|>\n\n{response}
                    teacher_resp_marker = "<|end_header_id|>\n\n"
                elif teacher_family == "olmo":
                    # OLMo (Tulu): ...<|assistant|>\n{response}
                    teacher_resp_marker = "<|assistant|>\n"
                else:
                    raise NotImplementedError(f"Unsupported teacher model family: {teacher_family}")

            for i in range(local_batch_size):
                student_ids = input_ids[i]  # list[int], student prompt+response
                teacher_ids = responses[i].tolist()  # list[int], teacher prompt+response+1 generated token
                teacher_logps = teacher_topk_logps[i][:, 0]  # [teacher_seq_len - 1], top-1 logprob

                # --- Step A: Find response boundaries in both sequences ---

                if R1_ZERO_MODE:
                    # Student: prompt length is already known from the batch
                    # itself -- no marker search needed.
                    student_resp_start_tok = int(attention_mask[i, :prompt_length].sum().item())
                    student_full_text = student_tokenizer.decode(student_ids, skip_special_tokens=False)
                else:
                    # Student: decode full sequence, find last occurrence of response marker
                    student_full_text = student_tokenizer.decode(student_ids, skip_special_tokens=False)
                    student_resp_start_char = student_full_text.rfind(student_resp_marker)
                    if student_resp_start_char == -1:
                        # Fallback: can't find marker, fill zeros
                        warnings.warn(f"[align seq {i}] Cannot find student response marker, filling zeros")
                        continue
                    student_resp_start_char += len(student_resp_marker)

                    # Find the token index where the response content starts in student
                    student_resp_start_tok = 0
                    decoded_so_far = ""
                    for tok_idx in range(len(student_ids)):
                        decoded_so_far = student_tokenizer.decode(student_ids[:tok_idx + 1], skip_special_tokens=False)
                        if len(decoded_so_far) >= student_resp_start_char:
                            student_resp_start_tok = tok_idx + 1
                            break

                # Find the token index where the response content ends in student
                # (exclude trailing special tokens)
                student_resp_end_tok = len(student_ids)
                student_special_end_ids = set()
                if student_tokenizer.eos_token_id is not None:
                    student_special_end_ids.add(student_tokenizer.eos_token_id)
                if student_family == "llama":
                    # LLaMA 3.x uses <|eot_id|> (128009) as end-of-turn, distinct from eos
                    eot_id = student_tokenizer.convert_tokens_to_ids("<|eot_id|>")
                    if isinstance(eot_id, int) and eot_id != student_tokenizer.unk_token_id:
                        student_special_end_ids.add(eot_id)
                elif student_family == "qwen":
                    # Qwen uses <|im_end|> as end-of-turn and <|endoftext|> as eos
                    for special_tok in ["<|im_end|>", "<|endoftext|>"]:
                        tok_id = student_tokenizer.convert_tokens_to_ids(special_tok)
                        if isinstance(tok_id, int) and tok_id != student_tokenizer.unk_token_id:
                            student_special_end_ids.add(tok_id)
                elif student_family in ("deepseek", "raw"):
                    # DeepSeek, and raw r1_zero mode, have no separate end-of-turn
                    # token distinct from eos (already added above via eos_token_id).
                    pass
                elif student_family == "olmo":
                    # OLMo (Tulu) has no separate end-of-turn token distinct from eos;
                    # "<|endoftext|>" is already added above via eos_token_id.
                    pass
                while student_resp_end_tok > student_resp_start_tok and student_ids[student_resp_end_tok - 1] in student_special_end_ids:
                    student_resp_end_tok -= 1

                student_resp_ids = student_ids[student_resp_start_tok:student_resp_end_tok]

                # Teacher generates 1 extra token during prefill; drop it before locating boundaries.
                teacher_ids_no_gen = teacher_ids[:-1]

                if R1_ZERO_MODE:
                    # Re-encode the student's own prompt slice (bos/eos swapped
                    # for the teacher) to get the teacher's prompt token count
                    # directly -- no marker search needed, since the prompt
                    # content is identical literal text on both sides.
                    student_prompt_text = student_tokenizer.decode(
                        student_ids[:student_resp_start_tok], skip_special_tokens=False
                    )
                    teacher_prompt_text = self._swap_bos_eos_for_teacher(student_prompt_text)
                    teacher_resp_start_tok = len(
                        teacher_tokenizer(teacher_prompt_text, add_special_tokens=False)["input_ids"]
                    )
                    # Only needed for the debug dump below (no marker search
                    # happens in this mode), but the dump code references it
                    # unconditionally, so compute it here too.
                    teacher_full_text = teacher_tokenizer.decode(teacher_ids_no_gen, skip_special_tokens=False)
                    if teacher_resp_start_tok >= len(teacher_ids_no_gen):
                        warnings.warn(
                            f"[align seq {i}] R1_ZERO_MODE: teacher prompt re-encoding "
                            f"({teacher_resp_start_tok} tok) covers the whole teacher sequence "
                            f"({len(teacher_ids_no_gen)} tok); skipping"
                        )
                        continue
                else:
                    # Teacher: decode full sequence, find last occurrence of response marker
                    teacher_full_text = teacher_tokenizer.decode(teacher_ids_no_gen, skip_special_tokens=False)
                    teacher_resp_start_char = teacher_full_text.rfind(teacher_resp_marker)
                    if teacher_resp_start_char == -1:
                        warnings.warn(f"[align seq {i}] Cannot find teacher response marker, filling zeros")
                        continue
                    teacher_resp_start_char += len(teacher_resp_marker)

                    # Find the token index where the response content starts in teacher
                    teacher_resp_start_tok = 0
                    decoded_so_far = ""
                    for tok_idx in range(len(teacher_ids_no_gen)):
                        decoded_so_far = teacher_tokenizer.decode(teacher_ids_no_gen[:tok_idx + 1], skip_special_tokens=False)
                        if len(decoded_so_far) >= teacher_resp_start_char:
                            teacher_resp_start_tok = tok_idx + 1
                            break

                # Teacher response ends before trailing special tokens
                teacher_special_end_ids = set()
                if teacher_tokenizer.eos_token_id is not None:
                    teacher_special_end_ids.add(teacher_tokenizer.eos_token_id)
                if teacher_family == "qwen":
                    # Qwen uses <|im_end|> as end-of-turn and <|endoftext|> as eos; both should be stripped
                    for special_tok in ["<|im_end|>", "<|endoftext|>"]:
                        tok_id = teacher_tokenizer.convert_tokens_to_ids(special_tok)
                        if isinstance(tok_id, int) and tok_id != teacher_tokenizer.unk_token_id:
                            teacher_special_end_ids.add(tok_id)
                elif teacher_family in ("deepseek", "raw"):
                    # DeepSeek, and raw r1_zero mode, have no separate end-of-turn
                    # token distinct from eos (already added above via eos_token_id).
                    pass
                elif teacher_family == "llama":
                    # LLaMA 3.x uses <|eot_id|> (128009) as end-of-turn, distinct from eos
                    eot_id = teacher_tokenizer.convert_tokens_to_ids("<|eot_id|>")
                    if isinstance(eot_id, int) and eot_id != teacher_tokenizer.unk_token_id:
                        teacher_special_end_ids.add(eot_id)
                elif teacher_family == "olmo":
                    # OLMo (Tulu) has no separate end-of-turn token distinct from eos;
                    # "<|endoftext|>" is already added above via eos_token_id.
                    pass
                teacher_resp_end_tok = len(teacher_ids_no_gen)
                while teacher_resp_end_tok > teacher_resp_start_tok and teacher_ids_no_gen[teacher_resp_end_tok - 1] in teacher_special_end_ids:
                    teacher_resp_end_tok -= 1

                teacher_resp_ids = teacher_ids_no_gen[teacher_resp_start_tok:teacher_resp_end_tok]

                # Compact runtime audit for the same number of samples shown by
                # the trainer's rollout console. This makes terminal-token
                # alignment visible without printing the full token mapping.
                console_samples = max(0, int(os.environ.get("VERL_CONSOLE_ROLLOUT_SAMPLES", "0")))
                if i < console_samples:
                    student_terminal = None
                    if student_resp_end_tok < len(student_ids):
                        student_terminal_id = int(student_ids[student_resp_end_tok])
                        student_terminal = (
                            student_tokenizer.decode([student_terminal_id], skip_special_tokens=False),
                            student_terminal_id,
                        )

                    teacher_terminal = None
                    if teacher_resp_end_tok < len(teacher_ids_no_gen):
                        teacher_terminal_id = int(teacher_ids_no_gen[teacher_resp_end_tok])
                        teacher_terminal = (
                            teacher_tokenizer.decode([teacher_terminal_id], skip_special_tokens=False),
                            teacher_terminal_id,
                        )

                    teacher_eos_logp_idx = teacher_resp_end_tok - 1
                    teacher_eos_logp = (
                        float(teacher_logps[teacher_eos_logp_idx])
                        if 0 <= teacher_eos_logp_idx < len(teacher_logps)
                        else None
                    )
                    mapping_available = (
                        student_terminal is not None
                        and teacher_terminal is not None
                        and teacher_eos_logp is not None
                    )

                    def _terminal_display(terminal):
                        return "NONE" if terminal is None else f"{terminal[0]!r}({terminal[1]})"

                    logp_display = "NONE" if teacher_eos_logp is None else f"{teacher_eos_logp:.6g}"
                    print(
                        f"[OPD EOS mapping] seq={i} available={mapping_available} "
                        f"student={_terminal_display(student_terminal)} "
                        f"teacher={_terminal_display(teacher_terminal)} "
                        f"teacher_logp={logp_display}",
                        flush=True,
                    )

                # teacher_logps[j] = logprob of predicting token at position j+1
                # So for teacher token at position k, its logprob is teacher_logps[k-1]
                # For response tokens [teacher_resp_start_tok, teacher_resp_end_tok),
                # their logprobs are teacher_logps[teacher_resp_start_tok - 1 : teacher_resp_end_tok - 1]
                assert teacher_resp_start_tok > 0, (
                    f"[align seq {i}] teacher_resp_start_tok is 0, which would cause "
                    f"negative index in logprob slicing. Teacher prompt is empty or marker not found."
                )
                teacher_resp_logps = teacher_logps[teacher_resp_start_tok - 1 : teacher_resp_end_tok - 1]

                # --- Step B: Chunk-level greedy alignment on response text ---
                _do_dump_this_seq = (i < dump_num)
                aligned_logps, aligned_chunk_ids, seq_chunk_details, teacher_truncated = self._align_chunks(
                    student_resp_ids, teacher_resp_ids, teacher_resp_logps,
                    student_tokenizer, teacher_tokenizer, debug=_do_dump_this_seq,
                    large_chunk_threshold=self.large_chunk_threshold
                )

                # --- Dump alignment details ---
                if _do_dump_this_seq:
                    _dump_alignment(current_dump_step, i, {
                        "mode": "cross_tokenizer",
                        "student_family": student_family,
                        "teacher_family": teacher_family,
                        "is_same_tokenizer": False,
                        "template_mapping": getattr(self, '_template_mapping', None),
                        "teacher_truncated": teacher_truncated,
                        # Full text with chat template
                        "student_full_text": student_full_text,
                        "teacher_full_text": teacher_full_text,
                        # Response boundaries
                        "student_resp_start_tok": student_resp_start_tok,
                        "student_resp_end_tok": student_resp_end_tok,
                        "teacher_resp_start_tok": teacher_resp_start_tok,
                        "teacher_resp_end_tok": teacher_resp_end_tok,
                        # Explicit terminal-token audit. Token IDs are expected
                        # to differ across tokenizer families; the important
                        # invariant is that teacher_eos_logp_index predicts the
                        # teacher terminal token and is assigned to the first
                        # student terminal-token position.
                        "eos_mapping": {
                            "student_first_terminal_position": (
                                student_resp_end_tok if student_resp_end_tok < len(student_ids) else None
                            ),
                            "student_terminal_tokens": [
                                {
                                    "position": idx,
                                    "id": int(student_ids[idx]),
                                    "text": student_tokenizer.decode(
                                        [student_ids[idx]], skip_special_tokens=False
                                    ),
                                }
                                for idx in range(student_resp_end_tok, len(student_ids))
                            ],
                            "teacher_first_terminal_position": (
                                teacher_resp_end_tok
                                if teacher_resp_end_tok < len(teacher_ids_no_gen)
                                else None
                            ),
                            "teacher_terminal_tokens": [
                                {
                                    "position": idx,
                                    "id": int(teacher_ids_no_gen[idx]),
                                    "text": teacher_tokenizer.decode(
                                        [teacher_ids_no_gen[idx]], skip_special_tokens=False
                                    ),
                                }
                                for idx in range(teacher_resp_end_tok, len(teacher_ids_no_gen))
                            ],
                            "teacher_eos_logp_index": (
                                teacher_resp_end_tok - 1
                                if teacher_resp_end_tok > 0
                                and teacher_resp_end_tok - 1 < len(teacher_logps)
                                else None
                            ),
                            "teacher_eos_logp": (
                                float(teacher_logps[teacher_resp_end_tok - 1])
                                if teacher_resp_end_tok > 0
                                and teacher_resp_end_tok - 1 < len(teacher_logps)
                                else None
                            ),
                            "mapping_available": bool(
                                student_resp_end_tok < len(student_ids)
                                and teacher_resp_end_tok < len(teacher_ids_no_gen)
                                and teacher_resp_end_tok > 0
                                and teacher_resp_end_tok - 1 < len(teacher_logps)
                            ),
                        },
                        # Response text
                        "student_resp_text": student_tokenizer.decode(student_resp_ids, skip_special_tokens=False),
                        "teacher_resp_text": teacher_tokenizer.decode(teacher_resp_ids, skip_special_tokens=False),
                        # Token counts
                        "student_resp_token_count": len(student_resp_ids),
                        "teacher_resp_token_count": len(teacher_resp_ids),
                        # Student response tokens (id + decoded text)
                        "student_resp_tokens": [
                            {"pos": idx, "id": int(tid), "text": student_tokenizer.decode([tid], skip_special_tokens=False)}
                            for idx, tid in enumerate(student_resp_ids)
                        ],
                        # Teacher response tokens (id + decoded text + logprob)
                        "teacher_resp_tokens": [
                            {
                                "pos": idx,
                                "id": int(teacher_resp_ids[idx]),
                                "text": teacher_tokenizer.decode([teacher_resp_ids[idx]], skip_special_tokens=False),
                                "logprob": float(teacher_resp_logps[idx]) if idx < len(teacher_resp_logps) else None,
                            }
                            for idx in range(len(teacher_resp_ids))
                        ],
                        # Chunk alignment details
                        "chunk_alignment": seq_chunk_details,
                        # Final aligned logprobs for student tokens
                        "aligned_logps": aligned_logps.tolist(),
                        # Stats
                        "stats": {
                            "n_inf_positions": int((aligned_logps == float('inf')).sum()),
                            "n_valid_positions": int((aligned_logps != float('inf')).sum()) - int((aligned_logps == 0).sum()),
                            "n_zero_positions": int((aligned_logps == 0).sum()),
                            "mean_valid_logp": float(aligned_logps[
                                (aligned_logps != float('inf')) & (aligned_logps != 0)
                            ].mean()) if ((aligned_logps != float('inf')) & (aligned_logps != 0)).any() else None,
                        },
                    })

                # --- Step C: Check if teacher was truncated (overlong sequence) ---
                # teacher_truncated=True means teacher tokens ran out before covering
                # the full student response. Skip the whole sequence: fill row with inf
                # → advantage = 0 → no gradient anywhere on this seq.
                # Fallback-induced inf holes (mid-sequence) are handled token-wise in
                # core_algos (torch.where), so we DON'T blow away the whole seq here.
                if teacher_truncated:
                    teacher_topk_logps_padded[i, :] = float('inf')
                    teacher_chunk_ids_padded[i, :] = -1.0
                    warnings.warn(
                        f"[align seq {i}] teacher truncated: skipping entire sequence "
                        f"(stu_resp={len(student_resp_ids)}t, tec_resp={len(teacher_resp_ids)}t)"
                    )
                    continue

                # --- Step D: Fill into the padded tensor ---
                # The padded tensor has shape [bs, student_padded_seq_len].
                # attention_mask[i] marks valid positions (prompt + response).
                #
                # After the same-tokenizer shift-right fix, core_algos uses:
                #   teacher_log_probs = advantages[..., -R:]
                # where R = response_length (including special tokens).
                # Position j holds the logprob predicting token j, no offset needed.
                # So aligned_logps[0] fills at resp_start, aligned_logps[-1] at resp_end-1.
                # Special token positions (eos/eot) get inf sentinel.
                valid_positions = attention_mask[i].nonzero(as_tuple=True)[0]

                # Direct fill: aligned_logps[k] → position resp_start + k
                fill_start = student_resp_start_tok
                fill_positions = valid_positions[fill_start:fill_start + len(aligned_logps)]
                teacher_topk_logps_padded[i, fill_positions] = aligned_logps
                teacher_chunk_ids_padded[i, fill_positions] = aligned_chunk_ids

                # Fill special token positions.
                # For non-truncated sequences: teacher has an eos logprob that tells the
                # student when to stop. Map teacher's eos logprob to student's first eos
                # position so the student learns the stopping signal.
                # Remaining special tokens (if any) get inf sentinel → advantage = 0.
                # For truncated sequences (no eos): sentinel range is empty, nothing to do.
                sentinel_start = student_resp_end_tok
                sentinel_end = len(student_ids)
                n_special = sentinel_end - sentinel_start

                if n_special > 0:
                    # Teacher's eos logprob: predicts eos at position teacher_resp_end_tok,
                    # so its logprob is at teacher_logps[teacher_resp_end_tok - 1]
                    teacher_eos_logp_idx = teacher_resp_end_tok - 1
                    if teacher_eos_logp_idx < len(teacher_logps):
                        teacher_eos_logp = teacher_logps[teacher_eos_logp_idx]
                    else:
                        teacher_eos_logp = float('inf')

                    # First special token gets teacher's eos logprob
                    first_special_pos = valid_positions[sentinel_start]
                    teacher_topk_logps_padded[i, first_special_pos] = teacher_eos_logp
                    teacher_chunk_ids_padded[i, first_special_pos] = (
                        aligned_chunk_ids.max().item() + 1 if len(aligned_chunk_ids) > 0 else 0
                    )

                    # Remaining special tokens (if any) get inf sentinel
                    if n_special > 1:
                        remaining_positions = valid_positions[sentinel_start + 1:sentinel_end]
                        teacher_topk_logps_padded[i, remaining_positions] = float('inf')
                        teacher_chunk_ids_padded[i, remaining_positions] = -1.0

            return torch.cat(
                [teacher_topk_logps_padded, teacher_chunk_ids_padded.to(teacher_topk_logps_padded.dtype)],
                dim=-1,
            )

        if is_async:
            return SimpleNamespace(get=handle_futures)
        else:
            return handle_futures()


@register("opd")
class OPDRewardManager(AbstractRewardManager):
    """The reward manager."""

    def __init__(
        self,
        tokenizer,
        num_examine,
        compute_score=None,
        reward_fn_key="data_source",
        max_resp_len=None,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.reward_fn_key = reward_fn_key
        self.max_resp_len = max_resp_len
        self.teacher_client = TeacherClient(
            os.environ['TEACHER_SERVER_IP'], int(os.environ['TEACHER_SERVER_PORT']),
            n_server_workers=int(os.environ['TEACHER_N_WORKERS']),
            teacher_ckpt_path=os.environ['TEACHER_CKPT_PATH'],
            max_seq_len=int(os.environ.get('TEACHER_MAX_SEQ_LEN', 0)) or None,
        )

    def __call__(self, data: DataProto, return_dict: bool = False):
        """We will expand this function gradually based on the available datasets"""
        # breakpoint()
        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
                reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
                return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        is_validate = data.meta_info.get("validate", False)

        # For Evaluation
        if is_validate:
            import concurrent.futures

            reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)

            already_print_data_sources = {}

            # Pre-decode all samples
            decoded_samples = []
            for i in range(len(data)):
                data_item = data[i]  # DataProtoItem

                prompt_ids = data_item.batch["prompts"]
                prompt_length = prompt_ids.shape[-1]
                valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
                valid_prompt_ids = prompt_ids[-valid_prompt_length:]

                response_ids = data_item.batch["responses"]
                valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
                valid_response_ids = response_ids[:valid_response_length]

                prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
                response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

                ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
                data_source = data_item.non_tensor_batch[self.reward_fn_key]
                extra_info = data_item.non_tensor_batch.get("extra_info", {})
                num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
                rollout_reward_scores = data_item.non_tensor_batch.get("reward_scores", {})
                extra_info["num_turns"] = num_turns
                extra_info["rollout_reward_scores"] = rollout_reward_scores

                decoded_samples.append({
                    "idx": i,
                    "prompt_str": prompt_str,
                    "response_str": response_str,
                    "ground_truth": ground_truth,
                    "data_source": data_source,
                    "extra_info": extra_info,
                    "valid_response_length": int(valid_response_length),
                })

            # Score all samples in parallel using ThreadPoolExecutor
            # (subprocess-based scoring in livecodebench already isolates execution,
            #  so threads are fine here — they just launch/wait on subprocesses)
            NUM_EVAL_WORKERS = 16

            def _score_one(sample):
                return default_compute_score(
                    data_source=sample["data_source"],
                    solution_str=sample["response_str"],
                    ground_truth=sample["ground_truth"],
                    extra_info=sample["extra_info"],
                )

            with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_EVAL_WORKERS) as executor:
                scores = list(executor.map(_score_one, decoded_samples))

            # Collect results
            for sample, score in zip(decoded_samples, scores):
                i = sample["idx"]
                valid_response_length = sample["valid_response_length"]

                if isinstance(score, dict):
                    reward = score["score"]
                    for key, value in score.items():
                        reward_extra_info[key].append(value)
                else:
                    reward = score
                    reward_extra_info["score"].append(reward)
                    reward_extra_info["acc"].append(reward > 0)
                    reward_extra_info["pred"].append(None)

                reward_tensor[i, valid_response_length - 1] = reward

                data_source = sample["data_source"]
                if data_source not in already_print_data_sources:
                    already_print_data_sources[data_source] = 0

                if already_print_data_sources[data_source] < self.num_examine:
                    already_print_data_sources[data_source] += 1
                    print("[prompt]", sample["prompt_str"][:500])
                    print("[response]", sample["response_str"][:500])
                    gt_str = str(sample["ground_truth"])
                    print("[ground_truth]", gt_str[:200] + ("..." if len(gt_str) > 200 else ""))
                    if isinstance(score, dict):
                        for key, value in score.items():
                            print(f"[{key}]", value)
                    else:
                        print("[score]", score)
            reward = reward_tensor
        else:
            reward = self.teacher_client.get_teacher_knowledge(data, False, self.tokenizer)
        # breakpoint()

        if return_dict:
            return {
                "reward_tensor": reward,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward
