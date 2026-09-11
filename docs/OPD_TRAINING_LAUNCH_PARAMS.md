# OPD Training Launch Params

This documents the `python3 -m verl.trainer.main_ppo ...` launch command used to
start an OPD training run, grouped by what each override actually controls.
Params are Hydra overrides onto `verl`'s PPO trainer config (this run uses the
FSDP actor path — `fsdp_config` / `ulysses_sequence_parallel_size` — not the
`megatron` path described in `recipe/gkd/config/on_policy_distill_trainer.yaml`).

## Data / batching

| Param | Meaning |
|---|---|
| `data.train_files` / `data.val_files` | Parquet dataset paths for train/val. |
| `data.prompt_key=prompt` | Which column in the parquet holds the prompt text. |
| `data.truncation='left'` | If a prompt exceeds `max_prompt_length`, cut from the left (keep the tail) instead of erroring or right-truncating. |
| `data.shuffle` / `data.seed` | Shuffle the dataset each epoch; RNG seed for reproducibility. |
| `data.dataloader_num_workers=0` | PyTorch DataLoader worker processes (0 = load in main process — useful for debugging/smoke, slower for real runs). |
| `data.filter_overlong_prompts_workers=1` | Parallelism for the (separate) pass that filters out prompts longer than `max_prompt_length`. |
| `data.max_prompt_length` / `data.max_response_length` | Hard caps (in tokens) on prompt and generated-response length — sizes both the rollout engine and the token budgets below. |
| `data.train_batch_size=${train_prompt_bsz}` | Number of **prompts** pulled per training iteration (before `rollout.n` expands each into multiple samples). |

## Rollout (generation) sampling

| Param | Meaning |
|---|---|
| `actor_rollout_ref.rollout.n=${n_resp_per_prompt}` | How many responses to sample per prompt (group size for the advantage estimator — GRPO/OPD-style methods need >1 to form relative comparisons). |
| `actor_rollout_ref.rollout.temperature/top_p/top_k` | Sampling params for training rollouts. |
| `actor_rollout_ref.rollout.val_kwargs.*` | Separate (usually more deterministic/greedy) sampling config used only during validation — its own temperature/top_p/top_k/`do_sample`/`n`/`max_tokens=128` (short, since val just needs a quick quality signal). |

## Algorithm / OPD-specific loss

| Param | Meaning |
|---|---|
| `algorithm.adv_estimator=${adv_estimator}` | Which advantage estimator to use (here presumably `opd`). |
| `algorithm.use_kl_in_reward` / `algorithm.kl_ctrl.kl_coef` | Whether to fold a KL-to-reference penalty directly into the *reward* (as opposed to a separate loss term), and its coefficient. |
| `actor.use_kl_loss` / `actor.kl_loss_coef` | Whether to instead add KL-to-reference as an explicit term in the *policy loss*, and its weight. (These two KL knobs are alternative/complementary ways of constraining drift from the reference model — check you actually want both, or you're double-penalizing.) |
| `actor.clip_ratio_low/high` | PPO-style clipping bounds on the probability ratio (asymmetric low/high, standard PPO-clip trick). |
| `actor.clip_ratio_c=10.0` | Dual-clip PPO's extra lower bound (Ye et al., dual-clip PPO paper) — caps how negative the surrogate loss can go when the ratio is very off-policy, must be `>1.0`. |
| `actor.policy_loss.loss_mode="opd"` | Selects the OPD policy-loss implementation (`verl/trainer/ppo/core_algos.py`) instead of vanilla PPO loss. |
| `actor.policy_loss.opd_loss_max_clamp=${opd_loss_max_clamp}` | Per-token clamp on the OPD advantage, which is `(teacher_logprob/student_logprob ratio − 1) × student_logprob` per chunk — without a clamp, a token where teacher and student log-probs diverge a lot (e.g. student ~-30) can produce a huge advantage that dominates the gradient for that single token. Set to `null` to disable. |
| `actor.entropy_coeff=0` | Weight on an entropy bonus in the loss (0 = disabled — no exploration bonus, common in distillation since you want to match the teacher, not explore). |
| `actor.grad_clip=1.0` | Global gradient-norm clipping. |
| `actor.loss_agg_mode=${loss_agg_mode}` | How per-token losses are reduced to a scalar: `token-mean`, `seq-mean-token-sum`, `seq-mean-token-mean`, or `seq-mean-token-sum-norm` — changes whether long sequences get proportionally more or equal weight vs. short ones. |
| `reward_model.reward_manager=opd` / `reward_model.enable=False` | Use the OPD reward manager (talks to the teacher server, does token alignment) instead of a learned reward model; reward model itself disabled since OPD gets its signal from the teacher, not an RM. |

## Model / sequence-length engineering

| Param | Meaning |
|---|---|
| `actor_rollout_ref.model.use_remove_padding=True` | Packs variable-length sequences without padding (removes wasted compute/memory on pad tokens) — standard efficiency flag, requires the model's attention impl to support it. |
| `+actor_rollout_ref.model.override_config.max_position_embeddings=512` | Hydra "add new key" override (`+`) forcing the underlying HF config's `max_position_embeddings` to 512 — i.e. constraining the model's positional-encoding range, presumably matching a short smoke-test context. |
| `+actor_rollout_ref.model.override_config.attn_implementation=sdpa` | Forces PyTorch's native scaled-dot-product-attention kernel instead of e.g. flash-attention or eager — portability over max speed. |
| `actor_rollout_ref.model.enable_gradient_checkpointing=True` | Recomputes activations during backward instead of storing them — the big activation-memory lever (see the memory-sizing discussion). |

## Batch-size / token-budget controls

| Param | Meaning |
|---|---|
| `actor.use_dynamic_bsz` / `ref.log_prob_use_dynamic_bsz` / `rollout.log_prob_use_dynamic_bsz` | Switch actor-update, reference-logprob, and rollout-logprob computation from fixed-sample micro-batches to token-budget packing — set together so all three stages use the same batching strategy. |
| `actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len}` | Token budget per GPU per micro-batch during the actor's PPO update (only used when dynamic bsz is on). |
| `ref.log_prob_max_token_len_per_gpu` / `rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len}` | Same token-budget idea, but for computing reference-model and rollout-model log-probs (inference-only forward passes — no optimizer state, so this can usually be set higher than the actor's). |
| `actor.ppo_mini_batch_size=${train_prompt_mini_bsz}` | Splits each `train_batch_size` batch into mini-batches for multiple gradient-update sub-steps per rollout (a PPO staple — allows several optimizer steps per collected batch of experience). |

## Parallelism / memory sharding (FSDP side, actor + ref)

| Param | Meaning |
|---|---|
| `actor.fsdp_config.param_offload` / `optimizer_offload=${offload}` | Move FSDP-sharded parameters/optimizer states to CPU between uses — trades GPU memory for host RAM + PCIe traffic. |
| `actor.fsdp_config.fsdp_size` / `ref.fsdp_config.fsdp_size=${fsdp_size}` | Size of the FSDP sharding group (how many GPUs share-shard one model replica) for actor and reference model respectively. |
| `actor.ulysses_sequence_parallel_size` / `ref.ulysses_sequence_parallel_size=${sp_size}` | Ulysses-style sequence parallelism degree — shards the sequence dimension across GPUs (the FSDP-path equivalent of Megatron's context parallel), reduces per-GPU activation memory for long sequences. |
| `ref.fsdp_config.param_offload=${offload}` | Same offload idea, applied to the frozen reference model. |

## Rollout engine (vLLM) sizing

Same family of knobs as the teacher-server vLLM process — see the teacher
server docs for the underlying vLLM semantics.

| Param | Meaning |
|---|---|
| `rollout.name=vllm`, `rollout.mode=sync` | Use vLLM as the generation backend, synchronous (blocking) rollout mode rather than async/overlapped. |
| `rollout.gpu_memory_utilization=0.25` | Fraction of GPU memory vLLM claims for the student's own rollout engine. |
| `rollout.tensor_model_parallel_size=${gen_tp}` | TP degree for the rollout engine specifically (can differ from the training TP/FSDP degree). |
| `rollout.enable_chunked_prefill=True` | Splits long prefills into chunks interleaved with decode steps — better scheduling throughput. |
| `rollout.max_num_batched_tokens=512` | Token budget per scheduler step. |
| `rollout.max_num_seqs=1` | Caps concurrent sequences in the engine to 1 — very conservative, consistent with a smoke-test-sized setup. |

## Trainer / bookkeeping

| Param | Meaning |
|---|---|
| `actor_rollout_ref.nccl_timeout=72000` | NCCL collective-op timeout in seconds (72000s = 20h) — generously long, avoids spurious timeout kills on slow/long steps. |
| `trainer.logger='["console"]'` | Log only to stdout (no wandb/tensorboard). |
| `trainer.project_name` / `trainer.experiment_name` | Run identifiers for logging/checkpoint naming. |
| `trainer.n_gpus_per_node` / `trainer.nnodes` | Cluster shape for Ray. |
| `trainer.val_before_train=False` | Skip the initial validation pass before training starts. |
| `trainer.test_freq=5` / `trainer.save_freq=5` | Run validation / save a checkpoint every 5 steps. |
| `trainer.total_epochs=1` / `trainer.total_training_steps=5` | Stop after whichever limit hits first — `total_training_steps=5` makes this clearly a short smoke run regardless of epoch count. |
| `trainer.default_local_dir="${CKPTS_DIR}"` | Checkpoint output directory. |
| `trainer.resume_mode=auto` | Auto-detect and resume from the latest checkpoint in that dir if one exists. |
| `trainer.log_val_generations=1` | Log 1 sample generation per validation pass (for eyeballing quality). |
