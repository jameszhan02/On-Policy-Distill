# OPD Engineering Playground

This folder is for learning the engineering flow behind this repo without
starting Ray, vLLM, FSDP, or real model training.

The goal is not to re-explain OPD theory. The goal is to understand how a
training framework wires the pieces together.

## Demo 1: Mini verl-style OPD Flow

Run:

```bash
python3 playground/mini_verl_opd_flow.py
```

This script mirrors the shape of the production flow:

```text
SmokeDataset
-> RolloutWorker.generate_sequences()
-> OPDRewardManager.__call__()
-> TeacherClient.get_logprobs()
-> ActorWorker.compute_old_log_probs()
-> compute_advantage()
-> ActorWorker.update_actor()
-> validate()
-> save_checkpoint()
```

It uses PyTorch tensors for the core training payloads, but keeps the model,
teacher, and rollout logic tiny enough to read in one sitting.

Tensor fields shown in the demo:

- `responses`: generated token IDs, shape `[batch, response_len]`
- `response_mask`: valid response-token mask
- `token_level_scores`: aligned teacher logprobs
- `old_log_probs`: old student logprobs
- `advantages`: OPD signal handed to actor update
- `current_log_probs`: current student logprobs used by the toy loss

## Mapping Toy Components To Real Code

| Playground | Real repo location | Role |
| --- | --- | --- |
| `SmokeDataset` | `verl/utils/dataset/rl_dataset.py` | Loads parquet rows and tokenizes prompts |
| `DataProto` | `verl/protocol.py` | Batch container passed between workers |
| `RolloutWorker.generate_sequences()` | `verl/workers/rollout/vllm_rollout/` | Student vLLM response generation |
| `TeacherClient` | `verl/workers/reward_manager/opd.py` and `recipe/gkd/teacher/` | Online teacher logprob service |
| `OPDRewardManager` | `verl/workers/reward_manager/opd.py` | Converts teacher output into OPD training signal |
| `compute_advantage()` | `verl/trainer/ppo/ray_trainer.py` + `verl/trainer/ppo/core_algos.py` | Prepares advantage tensor for actor update |
| `ActorWorker.update_actor()` | `verl/workers/fsdp_workers.py` + `verl/workers/actor/dp_actor.py` | Student forward/backward/optimizer step |
| `save_checkpoint()` | `verl/utils/checkpoint/` and worker checkpoint managers | Saves sharded training checkpoints |

## Production Flow To Read Next

After running the toy script, read this path in order:

```text
cross_distill_smoke_1gpu.sh
verl/trainer/main_ppo.py
verl/trainer/ppo/ray_trainer.py
verl/workers/reward_manager/opd.py
verl/trainer/ppo/core_algos.py
verl/workers/fsdp_workers.py
recipe/gkd/teacher/worker.py
recipe/gkd/teacher/vllm_engine_v019.py
```

Key production call chain:

```text
python3 -m verl.trainer.main_ppo
-> main_ppo.main(config)
-> run_ppo(config)
-> ray.get(TaskRunner.run.remote(config))
-> TaskRunner.run()
-> RayPPOTrainer.init_workers()
-> RayPPOTrainer.fit()
```

Inside `RayPPOTrainer.fit()`:

```text
dataloader batch
-> actor_rollout_wg.generate_sequences()
-> compute_reward(batch, self.reward_fn)
-> OPDRewardManager.__call__()
-> actor_rollout_wg.compute_log_prob()
-> compute_advantage(..., adv_estimator=opd)
-> actor_rollout_wg.update_actor()
-> optional _validate()
-> optional checkpoint save
```

## What To Modify When Adding A New Algorithm Piece

Common extension points:

- New reward signal:
  - `verl/workers/reward_manager/`
  - register with `@register("name")`
  - use `reward_model.reward_manager=name`

- New advantage estimator:
  - `verl/trainer/ppo/core_algos.py`
  - register with `@register_adv_est(...)`
  - use `algorithm.adv_estimator=name`

- New policy loss:
  - `verl/trainer/ppo/core_algos.py`
  - register with `@register_policy_loss("name")`
  - use `actor_rollout_ref.actor.policy_loss.loss_mode=name`

- New rollout behavior:
  - `verl/workers/rollout/`
  - controlled by `actor_rollout_ref.rollout.*`

- New student update behavior:
  - `verl/workers/fsdp_workers.py`
  - `verl/workers/actor/dp_actor.py`

For this OPD repo, the main algorithm-specific changes are:

```text
verl/workers/reward_manager/opd.py
verl/trainer/ppo/core_algos.py
recipe/gkd/teacher/
verl/trainer/main_ppo.py env forwarding for TEACHER_* and OPD_*
```
