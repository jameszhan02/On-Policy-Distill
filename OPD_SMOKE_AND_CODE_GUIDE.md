# OPD Smoke Test And Code Guide

This guide documents the current smoke-test workflow for this repo and where
the main OPD implementation lives.

The current smoke test has been run successfully through 5 training steps,
validation, OPD loss, and checkpoint save using:

- student: `Qwen2.5-0.5B-Instruct`
- teacher: `Qwen2.5-0.5B-Instruct`
- GPU: one 24 GB RTX 4090

## Mental Model

There are three separate pieces:

1. Teacher service
   - Runs a teacher model with vLLM.
   - Does not train.
   - Receives tokenized prompts/responses and returns teacher logprobs.

2. Student training process
   - Runs `verl.trainer.main_ppo`.
   - Loads the student model.
   - Uses vLLM rollout to generate student responses.
   - Calls the teacher service through the OPD reward manager.
   - Computes OPD loss and updates the student.

3. Ray
   - Manages the local/distributed worker processes used by `verl`.
   - Even one-GPU smoke tests use Ray.

High-level flow:

```text
train.parquet prompt
-> student rollout generates response
-> OPD reward manager sends prompt + response to teacher server
-> teacher server returns teacher top-k logprobs
-> reward manager aligns teacher token logprobs to student tokens
-> OPD advantage/loss updates student
-> optional validation
-> checkpoint save
```

## Smoke Data

Generate smoke data from the repo root:

```bash
python3 examples/data_preprocess/create_opd_smoke_data.py --output-dir data/smoke
```

This creates:

```text
data/smoke/train.parquet
data/smoke/val.parquet
```

The parquet files contain a tiny GSM8K-like table:

```text
prompt
data_source
reward_model
extra_info
```

Example row:

```python
{
    "prompt": "What is 1+1? Give the final answer in the form #### number.",
    "data_source": "openai/gsm8k",
    "reward_model": {"style": "rule", "ground_truth": "2"},
    "extra_info": {"split": "smoke", "index": 0},
}
```

Why these fields matter:

- `prompt`: model input.
- `data_source`: tells `default_compute_score()` which scorer to use.
- `reward_model.ground_truth`: used during validation.
- `extra_info`: optional metadata passed to scoring.

For OPD training itself, prompts are enough. For validation, this repo's
`OPDRewardManager` expects `reward_model.ground_truth`.

Inspect the data:

```bash
python3 - <<'PY'
import pandas as pd

df = pd.read_parquet("data/smoke/train.parquet")
print(df.head())
print(df.dtypes)
PY
```

## Model Paths

Use local checkpoint directories on the server when possible.

Current smoke model path:

```text
/data/shared_ckpt/opd_smoke/Qwen2.5-0.5B-Instruct
```

Use it as both:

```bash
MODEL_PATH=/data/shared_ckpt/opd_smoke/Qwen2.5-0.5B-Instruct
TEACHER_CKPT_PATH=/data/shared_ckpt/opd_smoke/Qwen2.5-0.5B-Instruct
```

`MODEL_PATH` is the student model.

`TEACHER_CKPT_PATH` is used in two places:

- teacher server loads teacher model weights
- student training loads teacher tokenizer for OPD retokenization/alignment

## Start Teacher vLLM

Open a shell or tmux session and keep it running:

```bash
cd /data/shengzhan/On-Policy-Distill
source .venv/bin/activate

cd recipe/gkd/teacher

TEACHER_CKPT_PATH=/data/shared_ckpt/opd_smoke/Qwen2.5-0.5B-Instruct \
TEACHER_GPU_MEMORY_UTILIZATION=0.12 \
TEACHER_MAX_MODEL_LEN=256 \
TEACHER_MAX_NUM_BATCHED_TOKENS=256 \
TEACHER_ENFORCE_EAGER=0 \
bash start_server_smoke_1gpu.sh
```

Watch logs:

```bash
tail -f worker.log
```

Teacher is ready when vLLM finishes loading and the worker remains alive. Useful
signals include:

```text
Model loading took ...
Available KV cache memory: ...
Supported_tasks: ['generate']
worker started...
```

Check processes:

```bash
ps -ef | grep -E "proxy.py|worker.py|VLLM::EngineCore" | grep -v grep
```

Stop teacher service:

```bash
pkill -f "proxy.py"
pkill -f "worker.py"
pkill -f "VLLM::EngineCore"
```

Be careful with `pkill -f "VLLM::EngineCore"` on shared servers if other jobs
from the same user are running vLLM.

## Teacher Parameters

Main knobs in `recipe/gkd/teacher/start_server_smoke_1gpu.sh`:

```bash
TEACHER_CKPT_PATH
TEACHER_GPU_MEMORY_UTILIZATION
TEACHER_MAX_MODEL_LEN
TEACHER_MAX_NUM_BATCHED_TOKENS
TEACHER_ENFORCE_EAGER
TEACHER_TP_SIZE
TEACHER_N_LOGPROBS
PROXY_FRONTEND_PORT
PROXY_BACKEND_PORT
```

Meanings:

- `TEACHER_CKPT_PATH`: local teacher checkpoint or HF model ID.
- `TEACHER_GPU_MEMORY_UTILIZATION`: vLLM GPU memory planning fraction. `0.12`
  means roughly 12 percent.
- `TEACHER_MAX_MODEL_LEN`: maximum teacher prompt + generated length.
- `TEACHER_MAX_NUM_BATCHED_TOKENS`: maximum vLLM batched token budget.
- `TEACHER_ENFORCE_EAGER`: `1` disables CUDA graph capture, sometimes lower
  memory but can also be higher depending on vLLM/runtime behavior.
- `TEACHER_TP_SIZE`: teacher tensor parallel size. Use `1` on one 4090.
- `TEACHER_N_LOGPROBS`: number of teacher logprobs returned. Smoke uses `1`.
- `PROXY_FRONTEND_PORT`: port the student connects to, default `15555`.
- `PROXY_BACKEND_PORT`: port workers connect to, default `15556`.

Files:

- `recipe/gkd/teacher/start_server_smoke_1gpu.sh`
- `recipe/gkd/teacher/worker.py`
- `recipe/gkd/teacher/proxy.py`
- `recipe/gkd/teacher/vllm_engine_v019.py`

## Start Student Training

Open another shell. Keep teacher running.

```bash
cd /data/shengzhan/On-Policy-Distill
source .venv/bin/activate

MODEL_PATH=/data/shared_ckpt/opd_smoke/Qwen2.5-0.5B-Instruct \
TEACHER_CKPT_PATH=/data/shared_ckpt/opd_smoke/Qwen2.5-0.5B-Instruct \
bash cross_distill_smoke_1gpu.sh
```

The smoke script will start Ray automatically if Ray is not already running:

```bash
ray status || ray start --head --num-gpus=1 --include-dashboard=false
```

Stop Ray:

```bash
ray stop
```

## Student Smoke Parameters

The smoke training script is:

```text
cross_distill_smoke_1gpu.sh
```

Important shell variables:

```bash
RAY_DATA_HOME=${RAY_DATA_HOME:-"${PWD}/data"}
MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen2.5-0.5B-Instruct"}
TEACHER_CKPT_PATH=${TEACHER_CKPT_PATH:-"Qwen/Qwen2.5-0.5B-Instruct"}
CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/smoke_ckpts/${project_name}/${exp_name}"}
TRAIN_FILE=${TRAIN_FILE:-"${RAY_DATA_HOME}/smoke/train.parquet"}
TEST_FILE=${TEST_FILE:-"${RAY_DATA_HOME}/smoke/val.parquet"}
```

Meanings:

- `PWD`: shell's current directory.
- `RAY_DATA_HOME`: smoke workspace data root. Defaults to repo `data/`.
- `TRAIN_FILE`: parquet prompts for training.
- `TEST_FILE`: parquet prompts for validation.
- `CKPTS_DIR`: output directory for checkpoints.

Current one-GPU settings:

```bash
NNODES=1
NGPUS_PER_NODE=1
max_prompt_length=256
max_response_length=128
train_prompt_bsz=1
n_resp_per_prompt=1
train_prompt_mini_bsz=1
sp_size=1
gen_tp=1
fsdp_size=1
offload=True
actor_ppo_max_token_len=512
infer_ppo_max_token_len=512
```

Important Hydra overrides passed to `main_ppo`:

```bash
data.train_files="${TRAIN_FILE}"
data.val_files="${TEST_FILE}"
data.prompt_key=prompt
data.dataloader_num_workers=0
data.max_prompt_length=256
data.max_response_length=128
data.train_batch_size=1

algorithm.adv_estimator=opd
reward_model.reward_manager=opd
actor_rollout_ref.actor.policy_loss.loss_mode=opd

actor_rollout_ref.model.path="${MODEL_PATH}"
actor_rollout_ref.model.override_config.attn_implementation=sdpa
actor_rollout_ref.model.use_remove_padding=True

actor_rollout_ref.rollout.name=vllm
actor_rollout_ref.rollout.mode=sync
actor_rollout_ref.rollout.gpu_memory_utilization=0.25
actor_rollout_ref.rollout.max_num_batched_tokens=512
actor_rollout_ref.rollout.max_num_seqs=1

trainer.total_training_steps=5
trainer.total_epochs=1
trainer.test_freq=5
trainer.save_freq=5
trainer.logger='["console"]'
trainer.default_local_dir="${CKPTS_DIR}"
```

Notes:

- `max_num_seqs=1` means student rollout vLLM handles one sequence at a time.
- `max_num_batched_tokens=512` limits rollout token scheduling budget.
- `attn_implementation=sdpa` avoids requiring FlashAttention2 for model
  attention.
- `use_remove_padding=True` still uses `flash_attn.bert_padding`; if
  `flash_attn` is not installed, change it to `False` for smoke tests.
- `test_freq=5` runs validation at step 5.
- `save_freq=5` saves checkpoint at step 5.

## Expected Successful Output

Training logs should reach:

```text
Training Progress: 100%|...| 5/5
```

Useful metrics:

```text
actor/pg_loss
actor/grad_norm
actor/opd_inf_tokens
actor/opd_inf_ratio
critic/score/mean
critic/advantages/mean
timing_s/reward
timing_s/update_actor
timing_s/testing
timing_s/save_checkpoint
val-core/openai/gsm8k/acc/mean@1
```

For smoke test, `val acc = 0.0` is fine. Five steps on toy prompts is not an
effectiveness test.

`actor/opd_inf_ratio=0.0` is a good sign for same-tokenizer smoke because no
tokens were skipped by OPD alignment.

## Check Outputs

Alignment dumps:

```bash
ls /tmp/opd_dumps
```

Checkpoint directory:

```bash
find data/smoke_ckpts/ON_POLICY_DISTILL/OPD_SMOKE_1GPU/global_step_5 -maxdepth 4 -type f | sort | head -50
```

Expected structure is usually:

```text
global_step_5/
  actor/
    model_world_size_1_rank_0.pt
    optim_world_size_1_rank_0.pt
    extra_state_world_size_1_rank_0.pt
    fsdp_config.json
    huggingface/
      config.json
      tokenizer.json
      tokenizer_config.json
      ...
```

The `actor/huggingface` directory may contain config/tokenizer files but not a
full `model.safetensors`. The train-time checkpoint is normally verl/FSDP shard
format.

## Convert FSDP Checkpoint To HuggingFace

For the current smoke script, backend is FSDP, so use:

```bash
cd /data/shengzhan/On-Policy-Distill
source .venv/bin/activate

python3 -m verl.model_merger merge \
  --backend fsdp \
  --local_dir data/smoke_ckpts/ON_POLICY_DISTILL/OPD_SMOKE_1GPU/global_step_5/actor \
  --target_dir data/smoke_ckpts/ON_POLICY_DISTILL/OPD_SMOKE_1GPU/global_step_5/actor_hf_merged \
  --trust-remote-code
```

After merge, check:

```bash
find data/smoke_ckpts/ON_POLICY_DISTILL/OPD_SMOKE_1GPU/global_step_5/actor_hf_merged -maxdepth 1 -type f | sort
```

You want to see model weights, usually:

```text
model.safetensors
```

or sharded weights:

```text
model-00001-of-000xx.safetensors
model.safetensors.index.json
```

Test HF loading:

```bash
python3 - <<'PY'
from transformers import AutoModelForCausalLM, AutoTokenizer

path = "data/smoke_ckpts/ON_POLICY_DISTILL/OPD_SMOKE_1GPU/global_step_5/actor_hf_merged"
tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, device_map="cpu")
print(type(tok).__name__)
print(type(model).__name__)
PY
```

## Main Code Path

For a small engineering demo of the same flow without Ray/vLLM/FSDP, see:

```bash
python3 playground/mini_verl_opd_flow.py
```

Read:

```text
playground/README.md
playground/mini_verl_opd_flow.py
```

### 1. Bash Entry

File:

```text
cross_distill_smoke_1gpu.sh
```

Responsibilities:

- define smoke-scale hyperparameters
- define paths
- export teacher environment variables
- generate smoke data if missing
- start Ray if missing
- call `python3 -m verl.trainer.main_ppo`

### 2. Python PPO Entry

File:

```text
verl/trainer/main_ppo.py
```

Important parts:

- Hydra entrypoint: `main(config)`
- runtime env forwarding for `TEACHER_*` and `OPD_*`
- Ray init: `ray.init(address="auto", runtime_env={...})`
- `TaskRunner.run()`
- builds tokenizer/datasets/reward manager/trainer
- calls `trainer.fit()`

### 3. Training Loop

File:

```text
verl/trainer/ppo/ray_trainer.py
```

Important flow inside `RayPPOTrainer.fit()`:

```text
load batch from dataloader
_get_gen_batch()
actor_rollout_wg.generate_sequences()
compute_response_mask()
compute_reward(batch, self.reward_fn)
actor_rollout_wg.compute_log_prob()
compute_advantage(..., adv_estimator=opd)
actor_rollout_wg.update_actor()
_validate() if enabled
save checkpoint if enabled
```

### 4. OPD Reward Manager

File:

```text
verl/workers/reward_manager/opd.py
```

Important objects:

- `OPDRewardManager`
- `TeacherClient`
- `TeacherClient.retokenize_batch()`
- `TeacherClient._detect_model_family()`
- `TeacherClient._build_chat_template_mapping()`
- `TeacherClient._align_chunks()`
- `TeacherClient.get_teacher_knowledge()`

Training mode:

```text
reward = teacher_client.get_teacher_knowledge(data, False, student_tokenizer)
```

Validation mode:

```text
default_compute_score(data_source, response_str, ground_truth, extra_info)
```

### 5. OPD Advantage And Loss

File:

```text
verl/trainer/ppo/core_algos.py
```

Important functions:

- registered OPD advantage estimator
- `compute_policy_loss_opd()`

The reward manager passes teacher logprobs/chunk IDs through the batch. The OPD
loss interprets those values as teacher-side token/chunk targets and updates the
student using PPO-style clipping.

### 6. Student Worker

File:

```text
verl/workers/fsdp_workers.py
```

Important functions:

- `_build_model_optimizer()`
- `_build_rollout()`
- `compute_log_prob()`
- `update_actor()`

This is where the student model, optimizer, and rollout engine are created.

### 7. Teacher vLLM Service

Files:

```text
recipe/gkd/teacher/start_server_smoke_1gpu.sh
recipe/gkd/teacher/proxy.py
recipe/gkd/teacher/worker.py
recipe/gkd/teacher/vllm_engine_v019.py
```

Responsibilities:

- `proxy.py`: ZeroMQ frontend/backend proxy.
- `worker.py`: receives requests and calls teacher engine.
- `vllm_engine_v019.py`: wraps vLLM and extracts top-k logprobs.

## Cross-Tokenizer Work

Same-tokenizer smoke uses:

```text
Qwen -> Qwen
```

For custom heterogeneous models, inspect tokenizers:

```bash
python3 tools/opd_inspect_tokenizer_pair.py \
  --student /path/to/student \
  --teacher /path/to/teacher
```

Existing mapping logic:

```text
verl/workers/reward_manager/opd.py
  TeacherClient._detect_model_family()
  TeacherClient._build_chat_template_mapping()
```

Currently supported template mappings:

```text
LLaMA -> Qwen
LLaMA -> DeepSeek
Qwen -> DeepSeek
Qwen -> LLaMA    (added 2026-08-31; NOT yet validated against a real LLaMA teacher -
                  verify via OPD_DUMP_DIR / tools/opd_inspect_tokenizer_pair.py before
                  trusting it for real training)
Qwen -> Qwen
DeepSeek -> DeepSeek
```

Note: LLaMA had only ever been wired up as a *student* family (chat-template mapping,
response-marker detection, and end-of-turn token stripping are three separate branch
points in `verl/workers/reward_manager/opd.py`). Adding a new "X -> LLaMA (teacher)"
pair means checking all three, not just `_build_chat_template_mapping()`.

If your pair is unsupported:

- add a family detector in `_detect_model_family()`
- add template replacement rules in `_build_chat_template_mapping()`
- run with `OPD_DUMP_DIR=/tmp/opd_dumps`
- inspect fallback/`inf` ratio and response alignment

## Common Issues Seen During Smoke Bring-Up

### Ray Not Running

Error:

```text
Could not find any running Ray instance
```

Fix:

```bash
ray start --head --num-gpus=1 --include-dashboard=false
```

The smoke script now auto-starts Ray by default.

### Qwen2Tokenizer Missing `all_special_tokens_extended`

Error:

```text
AttributeError: Qwen2Tokenizer has no attribute all_special_tokens_extended
```

Reason:

```text
vLLM child process expects a tokenizer attribute missing from this transformers/tokenizer combination.
```

Local fix:

```text
sitecustomize.py
```

and `start_server_smoke_1gpu.sh` adds repo root to `PYTHONPATH`.

### FlashAttention Missing

Error:

```text
ModuleNotFoundError: No module named 'flash_attn'
```

Two separate causes exist:

1. Model attention implementation asks for FlashAttention2.
   - Smoke script uses `attn_implementation=sdpa`.

2. Remove-padding path asks for `flash_attn.bert_padding`.
   - Install `flash-attn`, or set:

```bash
actor_rollout_ref.model.use_remove_padding=False
```

### vLLM `max_num_batched_tokens` vs `max_num_seqs`

Error:

```text
max_num_batched_tokens (512) must be greater than or equal to max_num_seqs (1024)
```

Fix:

```bash
actor_rollout_ref.rollout.max_num_seqs=1
```

### Validation Missing Ground Truth

Error:

```text
KeyError: 'reward_model'
```

Reason:

```text
validation path expects reward_model.ground_truth.
```

Fix:

```text
examples/data_preprocess/create_opd_smoke_data.py
```

now creates GSM8K-like validation rows with `reward_model.ground_truth`.

## Smoke vs Real Training (`cross_distill.sh`)

`cross_distill_smoke_1gpu.sh` is a shrunk-down version of the real training
script, `cross_distill.sh` (1 GPU / 5 steps / batch size 1, vs. 16 GPUs / 500
steps / batch size 128). Full diff, with which differences actually change
what the model learns vs. which are pure scale/plumbing:

| Param | Smoke | Real (`cross_distill.sh`) | Effect |
|---|---|---|---|
| `train_prompt_bsz` / `ppo_mini_batch_size` | 1 | 128 | **Accuracy-relevant** - see below |
| `total_training_steps` / `total_epochs` | 5 / 1 | 500 / 10 | **Accuracy-relevant** - the lever for "did distillation actually happen" |
| `max_response_length` | 128 | 16384 | **Accuracy-relevant** - caps how long the student is allowed to reason |
| `val_kwargs.max_tokens` | 128 | 31744 | **Accuracy-relevant** - same cap, applied at validation |
| `optim.lr_warmup_steps` | 1 | 10 | Minor training-stability knob |
| `data.max_prompt_length` | 256 | 1024 | Accuracy-relevant only if real prompts are long |
| `TEACHER_MAX_SEQ_LEN` | 512 | 30720 | Must scale with response length or the teacher truncates |
| `val_kwargs.n` | 1 | 4 | Statistical reliability of the *reported* accuracy, not the model itself |
| `NNODES`/`NGPUS_PER_NODE` | 1/1 | 2/8 | Pure scale, no accuracy effect |
| `sp_size`, `gen_tp`, `fsdp_size` | 1, 1, 1 | 2, 2, 8 | Pure parallelism/memory, no accuracy effect |
| `gpu_memory_utilization` | 0.25 | 0.90 | Pure memory budget for vLLM KV cache |
| `actor_ppo_max_token_len`/`infer_ppo_max_token_len`/`max_num_batched_tokens` | hardcoded small | computed from prompt+response length | Dynamic-batching token budget, no accuracy effect (just needs to be >= your longest sequence) |
| `test_freq`/`save_freq` | 5/5 | 25/500 | Logging/checkpoint cadence only |
| `logger` | console | console+wandb | Tracking only |
| `attn_implementation=sdpa` override | present | absent (uses model default, presumably flash-attn) | Speed/memory, numerically near-identical, not an accuracy difference |
| auto-generate data / auto-start Ray | present | absent | Convenience for iterating locally; real run assumes data + cluster already exist |

Notes on the accuracy-relevant ones:

- **`total_training_steps`/`total_epochs` is the big one.** Every other
  parameter being "correct" doesn't matter if the model only takes 5 gradient
  steps - that's a plumbing check, not a learning signal. 500 steps x 10
  epochs is what's actually intended to pull the student's distribution
  toward the teacher's.
- **`max_response_length`/`val_kwargs.max_tokens` matters a lot for
  GSM8K-style scoring specifically**, because the scorer (`gsm8k.py`,
  `method="strict"`) requires the literal `#### number` at the end of the
  response. If the cap is too small, a response gets truncated before it
  reaches that marker and scores 0 - not because the model is wrong, but
  because generation never got to finish. Too-short a cap manufactures
  artificially low accuracy independent of how well distillation is working.
- **`train_prompt_bsz`/`ppo_mini_batch_size` matters for gradient/reward
  noise, but with a catch:** both scripts set `ppo_mini_batch_size` equal to
  `train_prompt_bsz` (1==1 in smoke, 128==128 in real). That means even the
  500-step real config still does exactly one gradient step per rollout
  batch - no split into multiple mini-batches (and presumably a single PPO
  epoch, since `ppo_epochs` isn't overridden either). So `actor/ppo_kl` and
  `actor/pg_clipfrac` will be trivially `0.0` in the real run too, for the
  same structural reason as the smoke test: the ratio never gets a chance to
  move away from 1 within an update. To make PPO's clipping/KL bookkeeping
  meaningful at scale, set `ppo_mini_batch_size < train_prompt_bsz` (e.g. 32
  instead of 128) and/or `ppo_epochs > 1` - as written, that mechanism is
  inert in both configs.
- Everything else in the table (GPU/node count, `sp_size`/`gen_tp`/`fsdp_size`,
  `gpu_memory_utilization`, the token-length-budget knobs, `test_freq`/
  `save_freq`, `logger`) is scale and infrastructure, not learning dynamics -
  it determines how fast the run goes and whether it fits in memory, not what
  the model ends up learning. Getting these "wrong" typically means an OOM or
  a slow run, not a silently different trained model.

If the goal is still "does this model pair run end to end," keep the
smoke-scale numbers. The moment the goal is "is the student getting closer to
the teacher," bump `total_training_steps`/`total_epochs`,
`max_response_length`/`val_kwargs.max_tokens`, and `train_prompt_bsz` first -
roughly in that order of impact.

## Next Steps After Smoke

1. Run Qwen -> Qwen smoke to confirm local environment.
2. Run an existing supported cross-tokenizer pair, such as LLaMA -> Qwen.
3. Inspect `OPD_DUMP_DIR` outputs.
4. Add mapping for your real tokenizer pair.
5. Build a real prompt set for the behavior you want to distill.
6. Increase sequence length gradually.
7. Increase batch size gradually.
8. Move teacher to another GPU/server if training needs more memory.
9. Merge saved checkpoint to HuggingFace format.
10. Evaluate with a held-out dataset or `lm-evaluation-harness`.
