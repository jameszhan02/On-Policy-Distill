#!/usr/bin/env python3
"""A minimal tensor-based playground for the verl-style OPD data flow.

This is an engineering demo, not a faithful large-model trainer. It keeps the
same component boundaries as the repo:

Dataset -> RolloutWorker -> OPDRewardManager -> TeacherClient -> Advantage
-> ActorWorker.update_actor -> Checkpoint

It uses PyTorch tensors for the important training payloads:

- `responses`: token IDs, shape [batch, response_len]
- `response_mask`: valid-token mask, shape [batch, response_len]
- `old_log_probs`: student old-policy logprobs, shape [batch, response_len]
- `token_level_scores`: teacher aligned logprobs, shape [batch, response_len]
- `advantages`: OPD signal passed into the policy loss, shape [batch, response_len]

Run:
    python3 playground/mini_verl_opd_flow.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import json

import torch
import torch.nn.functional as F


VOCAB = {
    "<pad>": 0,
    "Reasoning:": 1,
    "one": 2,
    "plus": 3,
    "two": 4,
    "is": 5,
    "####": 6,
    "2": 7,
    "five": 8,
    "5": 9,
    "unsure.": 10,
    "0": 11,
}
ID_TO_TOKEN = {idx: token for token, idx in VOCAB.items()}
VOCAB_SIZE = len(VOCAB)


@dataclass
class DataProto:
    """Small stand-in for `verl.DataProto`.

    Real `DataProto` carries tensor fields in `batch`, Python/object metadata in
    `non_tensor_batch`, and step-level flags in `meta_info`.
    """

    batch: dict[str, Any]
    non_tensor_batch: dict[str, Any] = field(default_factory=dict)
    meta_info: dict[str, Any] = field(default_factory=dict)

    def union(self, other: "DataProto") -> "DataProto":
        return DataProto(
            batch={**self.batch, **other.batch},
            non_tensor_batch={**self.non_tensor_batch, **other.non_tensor_batch},
            meta_info={**self.meta_info, **other.meta_info},
        )


def decode(token_ids: torch.Tensor) -> str:
    return " ".join(ID_TO_TOKEN[int(idx)] for idx in token_ids if int(idx) != VOCAB["<pad>"])


class SmokeDataset:
    """Tiny prompt table.

    Real location:
        `verl/utils/dataset/rl_dataset.py`
    """

    def __init__(self) -> None:
        self.rows = [
            {"prompt": "What is 1+1?", "ground_truth": "2"},
            {"prompt": "What is 2+3?", "ground_truth": "5"},
            {"prompt": "What is 4-1?", "ground_truth": "3"},
        ]

    def __iter__(self):
        for row_id, row in enumerate(self.rows):
            yield DataProto(
                batch={"prompts": [row["prompt"]]},
                non_tensor_batch={
                    "uid": [f"row-{row_id}"],
                    "data_source": ["openai/gsm8k"],
                    "reward_model": [{"ground_truth": row["ground_truth"]}],
                },
            )


class RolloutWorker:
    """Student rollout worker.

    Real call:
        `actor_rollout_wg.generate_sequences()`

    Real implementation area:
        `verl/workers/rollout/vllm_rollout/`
    """

    def generate_sequences(self, batch: DataProto) -> DataProto:
        prompt = batch.batch["prompts"][0]
        if "1+1" in prompt:
            ids = [1, 2, 3, 2, 5, 4, 6, 7]
        elif "2+3" in prompt:
            ids = [1, 4, 3, 4, 5, 8, 6, 9]
        else:
            ids = [1, 10, 6, 11]

        responses = torch.tensor([ids], dtype=torch.long)
        response_mask = torch.ones_like(responses, dtype=torch.float32)
        return DataProto(
            batch={
                "responses": responses,
                "response_mask": response_mask,
                "response_text": [decode(responses[0])],
            },
            meta_info={"rollout_engine": "toy_vllm"},
        )


class TeacherClient:
    """Tiny online teacher.

    Real teacher service:
        `recipe/gkd/teacher/proxy.py`
        `recipe/gkd/teacher/worker.py`
        `recipe/gkd/teacher/vllm_engine_v019.py`
    """

    def get_logprobs(self, responses: torch.Tensor) -> torch.Tensor:
        # Teacher aligned logprobs. Less negative means teacher considers a token
        # more likely under its policy.
        teacher_log_probs = torch.full(responses.shape, -0.8, dtype=torch.float32)
        teacher_log_probs[responses == VOCAB["####"]] = -0.05
        teacher_log_probs[responses == VOCAB["2"]] = -0.05
        teacher_log_probs[responses == VOCAB["5"]] = -0.05
        teacher_log_probs[responses == VOCAB["0"]] = -2.5
        return teacher_log_probs


class OPDRewardManager:
    """Converts teacher service output into the OPD signal.

    Real implementation:
        `verl/workers/reward_manager/opd.py`
    """

    def __init__(self, teacher_client: TeacherClient) -> None:
        self.teacher_client = teacher_client

    def __call__(self, batch: DataProto) -> DataProto:
        teacher_log_probs = self.teacher_client.get_logprobs(batch.batch["responses"])
        chunk_ids = torch.arange(teacher_log_probs.shape[1], dtype=torch.float32).unsqueeze(0)
        return DataProto(
            batch={
                "token_level_scores": teacher_log_probs,
                "opd_chunk_ids": chunk_ids,
            },
            meta_info={"reward_manager": "opd"},
        )


class ActorWorker:
    """Tiny trainable student actor.

    Real implementation area:
        `verl/workers/fsdp_workers.py`
        `verl/workers/actor/dp_actor.py`
        `verl/trainer/ppo/core_algos.py::compute_policy_loss_opd`
    """

    def __init__(self) -> None:
        self.step = 0
        # One trainable logits vector shared by every token position. This is
        # tiny, but enough to demonstrate backward and optimizer update.
        self.logits = torch.nn.Parameter(torch.zeros(VOCAB_SIZE))
        self.optimizer = torch.optim.SGD([self.logits], lr=0.1)

    def compute_old_log_probs(self, batch: DataProto) -> DataProto:
        with torch.no_grad():
            log_probs_all = F.log_softmax(self.logits, dim=-1)
            old_log_probs = log_probs_all[batch.batch["responses"]]
        return DataProto(batch={"old_log_probs": old_log_probs})

    def update_actor(self, batch: DataProto) -> DataProto:
        self.step += 1
        responses = batch.batch["responses"]
        response_mask = batch.batch["response_mask"]
        teacher_log_probs = batch.batch["advantages"]
        old_log_probs = batch.batch["old_log_probs"]

        log_probs_all = F.log_softmax(self.logits, dim=-1)
        current_log_probs = log_probs_all[responses]

        # Minimal OPD-like objective:
        #   advantage = teacher_logp - old_student_logp
        #   loss = - current_student_logp * advantage
        #
        # This mirrors the engineering idea that teacher signal is carried in
        # `advantages` and consumed by actor update. Production code adds
        # PPO-style clipping, chunk assignment, and masking details.
        advantages = teacher_log_probs - old_log_probs
        token_losses = -current_log_probs * advantages.detach()
        loss = (token_losses * response_mask).sum() / response_mask.sum().clamp_min(1.0)

        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = self.logits.grad.detach().norm().item()
        self.optimizer.step()

        inf_ratio = torch.isinf(teacher_log_probs).float().mean().item()
        return DataProto(
            batch={"current_log_probs": current_log_probs.detach()},
            meta_info={
                "metrics": {
                    "actor/pg_loss": float(loss.detach()),
                    "actor/grad_norm": grad_norm,
                    "actor/opd_inf_ratio": inf_ratio,
                    "training/global_step": self.step,
                }
            },
        )


def compute_advantage(batch: DataProto) -> DataProto:
    """Stand-in for `compute_advantage(..., adv_estimator=opd)`.

    In this OPD repo, the "advantage" field is used as a carrier for aligned
    teacher logprobs and chunk metadata. The policy loss interprets it later.
    """

    return DataProto(batch={"advantages": batch.batch["token_level_scores"]})


def validate(batch: DataProto) -> dict[str, float]:
    response = batch.batch["response_text"][0]
    ground_truth = batch.non_tensor_batch["reward_model"][0]["ground_truth"]
    score = 1.0 if f"#### {ground_truth}" in response else 0.0
    return {"val-core/openai/gsm8k/acc/mean@1": score}


def save_checkpoint(path: Path, actor: ActorWorker) -> None:
    path.mkdir(parents=True, exist_ok=True)
    payload = {
        "global_step": actor.step,
        "format": "toy_checkpoint_not_hf",
        "logits": actor.logits.detach().tolist(),
    }
    (path / "actor_state.json").write_text(json.dumps(payload, indent=2))


def main() -> None:
    torch.manual_seed(0)
    dataset = SmokeDataset()
    rollout = RolloutWorker()
    reward_manager = OPDRewardManager(TeacherClient())
    actor = ActorWorker()

    print("mini tensor-based verl-style OPD flow")
    for batch in dataset:
        print("\n--- new train step ---")
        print("prompt:", batch.batch["prompts"][0])

        gen_output = rollout.generate_sequences(batch)
        batch = batch.union(gen_output)
        print("responses tensor:", batch.batch["responses"])
        print("response text:", batch.batch["response_text"][0])

        reward_output = reward_manager(batch)
        batch = batch.union(reward_output)
        print("teacher logprobs:", batch.batch["token_level_scores"])

        old_logprob_output = actor.compute_old_log_probs(batch)
        batch = batch.union(old_logprob_output)
        print("old student logprobs:", batch.batch["old_log_probs"])

        adv_output = compute_advantage(batch)
        batch = batch.union(adv_output)

        actor_output = actor.update_actor(batch)
        print("metrics:", actor_output.meta_info["metrics"])

    val_metrics = validate(batch)
    print("\nvalidation:", val_metrics)

    ckpt_path = Path("playground/outputs/mini_verl_opd_flow/global_step_3/actor")
    save_checkpoint(ckpt_path, actor)
    print("saved:", ckpt_path)


if __name__ == "__main__":
    main()
