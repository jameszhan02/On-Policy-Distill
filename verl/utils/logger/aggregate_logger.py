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
"""
A Ray logger will receive logging info from different processes.
"""

import datetime
import logging
import math
import numbers
import os
import pprint

import torch


_METRIC_DESCRIPTIONS = {
    "actor/pg_loss": "OPD policy loss used for backprop",
    "actor/entropy": "student token-distribution entropy",
    "actor/grad_norm": "gradient norm before clipping",
    "actor/lr": "actor learning rate",
    "actor/ppo_kl": "current-vs-old policy KL proxy",
    "actor/pg_clipfrac": "fraction clipped by PPO bound",
    "actor/pg_clipfrac_lower": "fraction clipped by lower dual bound",
    "actor/opd_inf_tokens": "teacher/alignment tokens skipped by OPD",
    "actor/opd_inf_ratio": "fraction of OPD response tokens skipped",
    "critic/score/mean": "teacher response log-prob sum; length-dependent",
    "critic/mean_token_score/mean": "teacher response log-prob per valid token",
    "critic/advantages/mean": "raw teacher signal before OPD loss transform",
    "response_length/mean": "mean generated response tokens",
    "response_length/clip_ratio": "fraction reaching response-length limit",
    "response/aborted_ratio": "fraction of empty or aborted responses",
    "perf/throughput": "processed prompt + response tokens per second",
    "perf/max_memory_allocated_gb": "peak GPU memory actively allocated",
    "perf/max_memory_reserved_gb": "peak GPU memory reserved by allocator",
    "perf/cpu_memory_used_gb": "host memory used by trainer process",
    "timing_s/step": "end-to-end seconds for this training step",
}


_SECTION_RULES = (
    ("Optimization", ("actor/",)),
    ("Teacher / OPD signal", ("critic/",)),
    ("Sequence lengths", ("response_length/", "response_length_non_aborted/", "response/", "prompt_length/")),
    ("Performance", ("perf/",)),
    ("Timing (seconds)", ("timing_s/",)),
    ("Timing per token", ("timing_per_token_ms/",)),
    ("Validation", ("val-", "val/", "test/")),
)


# Deliberately small console view for interactive training diagnosis. Detailed
# metrics are still sent unchanged to structured logging backends.
_DEBUG_METRICS = (
    ("actor/pg_loss", "OPD loss"),
    ("actor/grad_norm", "grad norm"),
    ("actor/lr", "learning rate"),
    ("actor/entropy", "student entropy"),
    ("critic/mean_token_score/mean", "teacher logp/token"),
    ("actor/opd_inf_ratio", "OPD skipped"),
    ("response_length/mean", "response tokens mean"),
    ("response_length/clip_ratio", "responses clipped"),
    ("response/aborted_ratio", "responses aborted"),
    ("perf/max_memory_allocated_gb", "GPU allocated GB"),
    ("timing_s/step", "step seconds"),
    ("perf/throughput", "tokens/second"),
)

_DEBUG_PERCENT_METRICS = {
    "actor/opd_inf_ratio",
    "response_length/clip_ratio",
    "response/aborted_ratio",
}


def _format_metric_value(value):
    """Format scalar metrics compactly while retaining useful precision."""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, numbers.Integral):
        return f"{int(value)}"
    if isinstance(value, numbers.Real):
        value = float(value)
        if value == 0:
            return "0"
        if not math.isfinite(value):
            return str(value)
        magnitude = abs(value)
        if magnitude >= 1_000_000 or magnitude < 1e-4:
            return f"{value:.3e}"
        if magnitude >= 1_000:
            return f"{value:,.1f}"
        return f"{value:.6g}"
    return pprint.pformat(value)


def _metric_section(key):
    for section, prefixes in _SECTION_RULES:
        if key.startswith(prefixes):
            return section
    return "Other metrics"


def format_debug_metrics_table(data: dict, step) -> str:
    """Render only metrics that directly help diagnose OPD training."""
    epoch = data.get("training/epoch")
    logged_step = data.get("training/global_step", step)
    title = f"Debug metrics | step {int(logged_step)}"
    if isinstance(epoch, numbers.Number):
        title += f" | epoch {_format_metric_value(epoch)}"

    rows = []
    for key, label in _DEBUG_METRICS:
        value = data.get(key)
        if not isinstance(value, numbers.Number):
            continue
        rendered = (
            f"{float(value):.1%}" if key in _DEBUG_PERCENT_METRICS else _format_metric_value(value)
        )
        rows.append((label, rendered))

    # Validation metric names contain their dataset name, so include matching
    # accuracy values dynamically instead of hard-coding one dataset path.
    for key, value in sorted(data.items()):
        if (
            key.startswith(("val-", "val/", "test/"))
            and "/acc/" in key
            and isinstance(value, numbers.Number)
        ):
            rows.append((key, _format_metric_value(value)))

    if not rows:
        return f"\n=== {title} ===\n(no selected debug metrics)"

    metric_width = max(len("Metric"), *(len(label) for label, _ in rows))
    value_width = max(len("Value"), *(len(value) for _, value in rows))
    separator = f"+-{'-' * metric_width}-+-{'-' * value_width}-+"
    lines = [
        "",
        f"=== {title} ===",
        separator,
        f"| {'Metric':<{metric_width}} | {'Value':>{value_width}} |",
        separator,
    ]
    lines.extend(f"| {label:<{metric_width}} | {value:>{value_width}} |" for label, value in rows)
    lines.append(separator)

    response_clip_ratio = data.get("response_length/clip_ratio")
    opd_inf_ratio = data.get("actor/opd_inf_ratio")
    aborted_ratio = data.get("response/aborted_ratio")
    alerts = []
    if isinstance(response_clip_ratio, numbers.Real) and response_clip_ratio >= 0.25:
        alerts.append(f"response clipping {float(response_clip_ratio):.1%}")
    if isinstance(opd_inf_ratio, numbers.Real) and opd_inf_ratio >= 0.05:
        alerts.append(f"OPD skipped tokens {float(opd_inf_ratio):.1%}")
    if isinstance(aborted_ratio, numbers.Real) and aborted_ratio > 0:
        alerts.append(f"aborted responses {float(aborted_ratio):.1%}")
    if alerts:
        lines.append("! " + " | ".join(alerts))
    return "\n".join(lines)


def format_metrics_table(data: dict, step) -> str:
    """Render scalar trainer metrics as compact, grouped console tables.

    Structured logging backends still receive the original metric dictionary;
    this function changes console presentation only.
    """
    scalar_metrics = {key: value for key, value in data.items() if isinstance(value, numbers.Number)}
    epoch = scalar_metrics.pop("training/epoch", None)
    logged_step = scalar_metrics.pop("training/global_step", step)
    title = f"Training step {int(logged_step)}"
    if epoch is not None:
        title += f" | epoch {_format_metric_value(epoch)}"

    grouped = {}
    for key, value in scalar_metrics.items():
        grouped.setdefault(_metric_section(key), []).append((key, value))

    lines = ["", f"=== {title} ==="]
    section_order = [section for section, _ in _SECTION_RULES] + ["Other metrics"]
    for section in section_order:
        rows = grouped.get(section)
        if not rows:
            continue
        rows.sort(key=lambda item: item[0])
        rendered = [
            (key, _format_metric_value(value), _METRIC_DESCRIPTIONS.get(key, "")) for key, value in rows
        ]
        metric_width = max(len("Metric"), *(len(row[0]) for row in rendered))
        value_width = max(len("Value"), *(len(row[1]) for row in rendered))
        meaning_width = 52
        separator = f"+-{'-' * metric_width}-+-{'-' * value_width}-+-{'-' * meaning_width}-+"
        lines.extend(
            [
                "",
                f"[{section}]",
                separator,
                f"| {'Metric':<{metric_width}} | {'Value':>{value_width}} | {'Meaning':<{meaning_width}} |",
                separator,
            ]
        )
        for metric, value, meaning in rendered:
            lines.append(
                f"| {metric:<{metric_width}} | {value:>{value_width}} | "
                f"{meaning[:meaning_width]:<{meaning_width}} |"
            )
        lines.append(separator)

    alerts = []
    response_clip_ratio = scalar_metrics.get("response_length/clip_ratio")
    if isinstance(response_clip_ratio, numbers.Real) and response_clip_ratio >= 0.25:
        alerts.append(
            f"response clipping is {float(response_clip_ratio):.1%}; generations are frequently hitting the token limit"
        )
    opd_inf_ratio = scalar_metrics.get("actor/opd_inf_ratio")
    if isinstance(opd_inf_ratio, numbers.Real) and opd_inf_ratio >= 0.05:
        alerts.append(f"OPD skipped-token ratio is {float(opd_inf_ratio):.1%}; inspect tokenizer alignment")
    aborted_ratio = scalar_metrics.get("response/aborted_ratio")
    if isinstance(aborted_ratio, numbers.Real) and aborted_ratio > 0:
        alerts.append(f"aborted-response ratio is {float(aborted_ratio):.1%}")
    if alerts:
        lines.extend(["", "[Attention]", *(f"! {alert}" for alert in alerts)])
    return "\n".join(lines)


def concat_dict_to_str(dict: dict, step):
    output = [f"step:{step}"]
    for k, v in dict.items():
        if isinstance(v, numbers.Number):
            output.append(f"{k}:{pprint.pformat(v)}")
    output_str = " - ".join(output)
    return output_str


class LocalLogger:
    """
    A local logger that logs messages to the console.

    Args:
        print_to_console (bool): Whether to print to the console.
    """

    def __init__(self, print_to_console=True):
        self.print_to_console = print_to_console
        self.console_mode = os.environ.get("VERL_CONSOLE_LOG_MODE", "full").strip().lower()

    def flush(self):
        pass

    def log(self, data, step):
        if self.print_to_console:
            if self.console_mode in {"debug", "compact"}:
                output = format_debug_metrics_table(data, step=step)
            else:
                output = format_metrics_table(data, step=step)
            print(output, flush=True)


class DecoratorLoggerBase:
    """
    Base class for all decorators that log messages.

    Args:
        role (str): The role (the name) of the logger.
        logger (logging.Logger): The logger instance to use for logging.
        level (int): The logging level.
        rank (int): The rank of the process.
        log_only_rank_0 (bool): If True, only log for rank 0.
    """

    def __init__(
        self, role: str, logger: logging.Logger = None, level=logging.DEBUG, rank: int = 0, log_only_rank_0: bool = True
    ):
        self.role = role
        self.logger = logger
        self.level = level
        self.rank = rank
        self.log_only_rank_0 = log_only_rank_0
        self.logging_function = self.log_by_logging
        if logger is None:
            self.logging_function = self.log_by_print

    def log_by_print(self, log_str):
        if not self.log_only_rank_0 or self.rank == 0:
            print(f"{self.role} {log_str}", flush=True)

    def log_by_logging(self, log_str):
        if self.logger is None:
            raise ValueError("Logger is not initialized")
        if not self.log_only_rank_0 or self.rank == 0:
            self.logger.log(self.level, f"{self.role} {log_str}")


def print_rank_0(message):
    """If distributed is initialized, print only on rank 0."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message, flush=True)
    else:
        print(message, flush=True)


def print_with_rank(message: str, rank: int = 0, log_only_rank_0: bool = False):
    """_summary_
    Print a message with rank information.
    This function prints the message only if `log_only_rank_0` is False or if the rank is 0.

    Args:
        message (str): _description_
        rank (int, optional): _description_. Defaults to 0.
        log_only_rank_0 (bool, optional): _description_. Defaults to False.
    """
    if not log_only_rank_0 or rank == 0:
        print(f"[Rank {rank}] {message}", flush=True)


def print_with_rank_and_timer(message: str, rank: int = 0, log_only_rank_0: bool = False):
    """_summary_
    Print a message with rank information and a timestamp.
    This function prints the message only if `log_only_rank_0` is False or if the rank is 0.

    Args:
        message (str): _description_
        rank (int, optional): _description_. Defaults to 0.
        log_only_rank_0 (bool, optional): _description_. Defaults to False.
    """
    now = datetime.datetime.now()
    message = f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] [Rank {rank}] {message}"
    if not log_only_rank_0 or rank == 0:
        print(message, flush=True)


def log_with_rank(message: str, rank, logger: logging.Logger, level=logging.INFO, log_only_rank_0: bool = False):
    """_summary_
    Log a message with rank information using a logger.
    This function logs the message only if `log_only_rank_0` is False or if the rank is 0.
    Args:
        message (str): The message to log.
        rank (int): The rank of the process.
        logger (logging.Logger): The logger instance to use for logging.
        level (int, optional): The logging level. Defaults to logging.INFO.
        log_only_rank_0 (bool, optional): If True, only log for rank 0. Defaults to False.
    """
    if not log_only_rank_0 or rank == 0:
        logger.log(level, f"[Rank {rank}] {message}")
