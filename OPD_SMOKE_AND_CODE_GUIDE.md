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

TEACHER_CKPT_PATH=/data/shared_ckpt/opd_teacher \
TEACHER_GPU_MEMORY_UTILIZATION=0.17 \
TEACHER_MAX_MODEL_LEN=1024 \
TEACHER_MAX_NUM_BATCHED_TOKENS=1024 \
TEACHER_ENFORCE_EAGER=0 \
bash start_server_smoke_1gpu.sh
```

```bash
tail -f worker.log
```

```bash
pkill -f "proxy.py"
pkill -f "worker.py"
pkill -f "VLLM::EngineCore"
```


```bash
ray stop
ray start --head --num-gpus=1 --include-dashboard=false
ray status   # confirm it now shows a live, local cluster
```

```bash
TRAIN_FILE=/data/shengzhan/On-Policy-Distill/data/gsm8k/train.parquet \
TEST_FILE=/data/shengzhan/On-Policy-Distill/data/gsm8k/test.parquet \
MODEL_PATH=/data/shared_ckpt/Llama-3.2-1B-Instruct \
TEACHER_CKPT_PATH=/data/shared_ckpt/opd_teacher \
bash cross_distill_smoke_1gpu.sh
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



Verify it actually stopped:

```bash
ps -ef | grep -E "proxy.py|worker.py|VLLM::EngineCore" | grep -v grep
ss -ltn | grep -E '15555|15556'
```

Both should come back empty.

`cs-tai-srv02` is a **shared server** — `pkill -f "VLLM::EngineCore"` matches by
process name only, not by owner, so it can kill another user's vLLM job too.
Scope it to yourself, or kill specific PIDs instead:

```bash
pkill -u "$(whoami)" -f "VLLM::EngineCore"     # only your own processes
# or:
ps -ef | grep VLLM::EngineCore | grep -v grep  # find the PID(s) that are yours
kill -9 <pid>
```

You don't need to `ray stop` between teacher restarts — the student training
script auto-starts Ray only if it isn't already running, so leaving Ray up
across teacher stop/start cycles is fine and saves a re-init. See "Stop Ray"
below for when you do want to tear the whole thing down.

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

## Training On Real GSM8K (Instead Of The 16-Prompt Smoke Set)

Generate the real dataset (needs internet access from the training box):

```bash
mkdir -p data/gsm8k
python3 examples/data_preprocess/gsm8k.py --local_save_dir data/gsm8k
```

This downloads/caches `openai/gsm8k` from HF Hub and writes:

```text
data/gsm8k/train.parquet   # 7,473 rows
data/gsm8k/test.parquet    # 1,319 rows
```

in the same schema `cross_distill_smoke_1gpu.sh` already expects — no other
script changes needed. `mkdir -p` first: `to_parquet()` does not create the
output directory itself and fails with `FileNotFoundError` if it's missing.

If the training box has no internet access, or you already have this data
(or a variant of it) elsewhere as raw `{"question": ..., "answer": ...}`
JSONL (the un-preprocessed HF `openai/gsm8k` shape), convert it directly
instead of downloading:

```bash
python3 - <<'PY'
import json
import re
import pandas as pd

def extract_solution(solution_str):
    solution = re.search(r"#### (\-?[0-9\.\,]+)", solution_str)
    assert solution is not None, f"no #### marker in: {solution_str!r}"
    return solution.group(0).split("#### ")[1].replace(",", "")

instruction_following = 'Let\'s think step by step and output the final answer after "####".'

def convert(jsonl_path, out_parquet_path, split):
    rows = []
    with open(jsonl_path) as f:
        for idx, line in enumerate(f):
            ex = json.loads(line)
            question_raw = ex["question"]
            answer_raw = ex["answer"]
            rows.append({
                "data_source": "openai/gsm8k",
                "prompt": [{"role": "user", "content": question_raw + " " + instruction_following}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": extract_solution(answer_raw)},
                "extra_info": {"split": split, "index": idx, "answer": answer_raw, "question": question_raw},
            })
    df = pd.DataFrame(rows)
    df.to_parquet(out_parquet_path)
    print(f"wrote {len(df)} rows -> {out_parquet_path}")

# EDIT THESE FOUR PATHS to match your source jsonl / this repo's data dir:
convert("/path/to/train.jsonl", "data/gsm8k/train.parquet", "train")
convert("/path/to/test.jsonl",  "data/gsm8k/test.parquet",  "test")
PY
```

This mirrors `examples/data_preprocess/gsm8k.py`'s exact extraction/schema
logic (`extract_solution()`'s regex, the `prompt`/`data_source`/
`reward_model`/`extra_info` shape), just reading from a local JSONL file
instead of going through `datasets.load_dataset(...)`.

Point training at the real data — `TRAIN_FILE`/`TEST_FILE` are already
env-overridable, no script edit needed for this part:

```bash
TRAIN_FILE=/data/shengzhan/On-Policy-Distill/data/gsm8k/train.parquet \
TEST_FILE=/data/shengzhan/On-Policy-Distill/data/gsm8k/test.parquet \
MODEL_PATH=/data/shared_ckpt/Llama-3.2-1B-Instruct \
TEACHER_CKPT_PATH=/data/shared_ckpt/opd_teacher \
bash cross_distill_smoke_1gpu.sh
```

### Setting Total Steps For A Real Run

The smoke defaults (`total_epochs=1` + `total_training_steps=5`) are fine for
the 16-prompt plumbing check, much too short to see any real learning on
7,473 real rows. Edit `cross_distill_smoke_1gpu.sh` directly near the bottom.
Two options:

**A. Go by epochs (auto-derives steps):** delete the
`trainer.total_training_steps=5 \` line entirely, keep only:

```bash
    trainer.total_epochs=<N> \
```

Per `verl/trainer/ppo/ray_trainer.py`: `total_training_steps` defaults to
`None`, and when it is `None` the trainer computes
`total_training_steps = len(train_dataloader) * total_epochs` itself — i.e.
exactly `N` full passes over the training file. If `trainer.total_training_steps`
is explicitly set to anything, it silently overrides this and hard-caps the
run there regardless of `total_epochs` — this is why leaving the line in at
`5` caps every run at 5 steps no matter how high `total_epochs` is set.

**B. Cap at an exact step count** (for a shorter first checkpoint, not a full
epoch):

```bash
    trainer.total_epochs=1 \
    trainer.total_training_steps=1000 \
```

`total_epochs=1` just needs to stay big enough that its own natural cap
(`len(train_dataloader)` steps) doesn't cut the run off before your
`total_training_steps` value is reached.

Picking a number — from an observed real run at `train_prompt_bsz=1`,
`timing_s/step ≈ 3.7-3.9s`:

| Target                                                        | Approx. wall-clock |
| ------------------------------------------------------------- | ------------------ |
| 1,000 steps (Option B, first "does accuracy move" checkpoint) | ~1 hour            |
| 1 full epoch = 7,473 steps (Option A, `total_epochs=1`)       | ~8 hours           |
| 3 full epochs                                                 | ~24 hours          |

Start with a short Option B run (~1000 steps) first and check whether
`val-core/openai/gsm8k/acc/mean@1` moves off `0.0`, before committing to a
multi-hour full-epoch run. Also bump `max_response_length`/
`val_kwargs.max_tokens` up from the smoke default of `128` once on real
data — real GSM8K reasoning chains need more room, or responses keep getting
truncated before the `#### number` marker regardless of step count.

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

The saved `global_step_*/actor` directory is a verl/FSDP checkpoint. Convert it
to a standard Hugging Face full-weight directory with the repository's built-in
model merger:

```bash
cd /data/shengzhan/On-Policy-Distill
source .venv/bin/activate

RUN_DIR=data/smoke_ckpts/ON_POLICY_DISTILL/OPD_SMOKE_1GPU_Llama-3.2-1B-Instruct_to_opd_teacher
export STEP=5000

# data/shengzhan/On-Policy-Distill/data/smoke_ckpts/ON_POLICY_DISTILL/OPD_SMOKE_1GPU_Llama-3.2-1B-Instruct_to_opd_teacher/global_step_3000/actor_hf_merged

python3 -m verl.model_merger merge \
  --backend fsdp \
  --local_dir "${RUN_DIR}/global_step_${STEP}/actor" \
  --target_dir "${RUN_DIR}/global_step_${STEP}/actor_hf_merged"
```

Set `STEP` to the checkpoint to convert. For example, to convert step 2000:

```bash
export STEP=2000

python3 -m verl.model_merger merge \
  --backend fsdp \
  --local_dir "${RUN_DIR}/global_step_${STEP}/actor" \
  --target_dir "${RUN_DIR}/global_step_${STEP}/actor_hf_merged"
```

Check the converted Hugging Face directory:

```bash
find "${RUN_DIR}/global_step_${STEP}/actor_hf_merged" \
  -maxdepth 1 -type f -printf '%f\n' | sort
```

Expected files include `config.json`, tokenizer files, and either one weight
file:

```text
model.safetensors
```

or standard Hugging Face weight shards:

```text
model-00001-of-00002.safetensors
model-00002-of-00002.safetensors
model.safetensors.index.json
```

Verify the converted model with Transformers:

```bash
python3 - <<'PY'
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

run_dir = "data/smoke_ckpts/ON_POLICY_DISTILL/OPD_SMOKE_1GPU_Llama-3.2-1B-Instruct_to_opd_teacher"
step = os.environ.get("STEP", "5000")
path = f"{run_dir}/global_step_{step}/actor_hf_merged"

tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(
    path,
    torch_dtype="auto",
    low_cpu_mem_usage=True,
)

print("model:", type(model).__name__)
print("parameters:", f"{model.num_parameters():,}")
print("tokenizer size:", len(tokenizer))
print("HF merge/load: OK")
PY
```

The usable Hugging Face model is now in:

```text
${RUN_DIR}/global_step_${STEP}/actor_hf_merged
```

If a model requires custom Hugging Face code, append `--trust-remote-code` to
the merge command. Do not delete the original `actor/` checkpoint until the
converted model passes the load test.

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

Note: LLaMA had only ever been wired up as a _student_ family (chat-template mapping,
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

The smoke script now auto-starts Ray by default. `ray stop`/`ray start`/`ray
status` are machine-scoped, not directory-scoped — run them from anywhere,
as long as you're in a shell with `.venv` activated.

### Ray Connects To A Stale/Dead Cluster Address

Error:

```text
Failed to connect to the default Ray cluster address at <ip>:6379. This is
most likely due to a previous Ray instance that has since crashed.
...
ConnectionError: Failed to connect to Ray cluster at <ip>:6379
```

Cause: the smoke script's auto-start check is
`ray status >/dev/null 2>&1 || ray start --head ...`. `ray status` can find a
**stale local address record** from a previous Ray head that has since died
(crashed, killed, or — on a shared server — possibly another session
entirely) and report that as fine, so the script skips starting a fresh
head. Then `ray.init(address="auto")` inside `main_ppo.py` tries to actually
connect to that dead address and fails after its retry timeout.

Fix — Ray's own error message tells you exactly what to do:

```bash
ray stop
ray start --head --num-gpus=1 --include-dashboard=false
ray status   # confirm it now shows a live, local cluster
```

```python
python3 - <<'PY'
import pandas as pd
df = pd.read_parquet("data/gsm8k/test.parquet")
df.sample(n=150, random_state=42).to_parquet("data/gsm8k/val_small.parquet")
PY
```

Then re-run training. On a shared server, also sanity-check the cluster you
just started is actually yours, not colliding with another user's (Ray's
default temp dir `/tmp/ray` and default GCS port `6379` aren't user-scoped):

```bash
ps -ef -o user,pid,cmd | grep -E "raylet|gcs_server" | grep -v grep
```

### `ray status` Shows No GPU Under `Total Usage`

Symptom: `ray status` runs and shows an active node with no errors, but the
`Total Usage:` section is empty — no `CPU`/`GPU`/`memory` lines at all,
instead of something like `0.0/1.0 GPU`. Ray is "up" but has zero resources
registered, so it can never actually schedule the rollout/training work.

Get a definitive answer (more reliable than eyeballing `ray status`):

```bash
python3 -c "import ray; ray.init(address='auto'); print(ray.cluster_resources())"
```

If `GPU` isn't a key in the printed dict, this is confirmed.

Two likely causes:

1. Ray was started without `--num-gpus=1` (autodetection can fail silently
   on some setups). Fix:
   ```bash
   ray stop
   ray start --head --num-gpus=1 --include-dashboard=false
   ray status   # should now show "0.0/1.0 GPU" under Total Usage
   ```
2. `CUDA_VISIBLE_DEVICES` was empty/unset in the shell that ran `ray start`:
   ```bash
   echo $CUDA_VISIBLE_DEVICES
   nvidia-smi                    # sanity check the GPU is visible from this shell
   unset CUDA_VISIBLE_DEVICES    # if it was set to an empty string
   ```

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

### `wait proxy server ready...` Loop Never Ends

Symptom: `start_server_smoke_1gpu.sh` prints `wait proxy server ready at
localhost:15556...` forever and never reaches `teacher proxy is ready`.

Old cause (fixed 2026-09-11): `wait_server_ready()` used to shell out to
`telnet`, which is not installed by default on many minimal Linux images.
When `telnet` is missing, the command silently fails (its "not found" error
goes through the same `2> /dev/null` meant for its own stderr), `grep`/`wc -l`
see empty input, and the readiness check reports "not ready" forever — even
if `proxy.py` started and bound its ports just fine within the first second.
`start_server_smoke_1gpu.sh` now checks readiness with bash's built-in
`/dev/tcp/<host>/<port>` instead, so it no longer depends on an external
`telnet` binary.

If you still see this loop with the current script, it means the check is
now telling the truth: nothing is actually listening. Check
`recipe/gkd/teacher/proxy.log` immediately (don't wait) — see the next entry
and "Where To Find Logs" below.

### `ModuleNotFoundError: No module named 'zmq'` In `proxy.log`

Cause: `proxy.py`'s `import zmq` (from the `pyzmq` package) ran under a
Python interpreter that doesn't have `pyzmq` installed — almost always
because `.venv` wasn't activated in the shell/tmux pane you launched the
teacher script from. Activation is per-shell-session state; it does not
persist across new panes/SSH logins, and activating a different project's
venv in the same shell fully deactivates this one first (it's a switch, not
a stack) — you need to re-`source` it every time you come back.

Fix:

```bash
source /data/shengzhan/On-Policy-Distill/.venv/bin/activate
which python3                                  # should be inside .venv/bin/
python3 -c "import zmq; print(zmq.__file__)"   # should succeed standalone
```

If `python3` really is `.venv/bin/python3` and `import zmq` still fails,
`pyzmq` itself is missing from that venv — install it directly:

```bash
pip install pyzmq==27.1.0
```

`requirements.txt` also lists a second, unrelated package named `zmq==0.0.0`
(a PyPI name-squat/placeholder package, not the real ZeroMQ bindings — the
real one is `pyzmq`, which provides the importable `zmq` module). Having
both listed is a `requirements.txt` mistake, not something you need to fix
to unblock training, but if reinstalling `pyzmq` alone doesn't resolve the
import, rule out clobbering with:

```bash
pip uninstall zmq -y
pip install --force-reinstall pyzmq==27.1.0
```

## Where To Find Logs

Nothing here writes to one central log file — check the piece you care about:

- **Teacher proxy**: `recipe/gkd/teacher/proxy.log` — ZMQ proxy startup /
  errors (`"proxy is running..."` on success).
- **Teacher worker**: `recipe/gkd/teacher/worker.log` — vLLM engine load,
  KV cache sizing, request handling. Both are written relative to wherever
  you `cd`'d before running `start_server_smoke_1gpu.sh`.
- **Student training console**: the smoke config sets
  `trainer.logger='["console"]'`, and `cross_distill_smoke_1gpu.sh` runs
  `main_ppo` in the foreground — training metrics/progress only print to
  your terminal's stdout, nothing is saved unless you redirect it yourself:
  ```bash
  bash cross_distill_smoke_1gpu.sh 2>&1 | tee train_run.log
  ```
- **Hydra's own log + resolved config**: auto-created relative to wherever
  you launched `main_ppo` from (repo root, for the smoke script):
  ```text
  outputs/<date>/<time>/main_ppo.log        # logger.info()-level messages
  outputs/<date>/<time>/.hydra/config.yaml  # fully-resolved config actually used
  outputs/<date>/<time>/.hydra/overrides.yaml
  ```
- **Ray's internal logs**: `/tmp/ray/session_latest/logs/` — per-actor/worker
  stdout+stderr, `raylet.err`, `gcs_server.err`. Check here first if an actor
  dies silently or gets OOM-killed without a clean Python traceback reaching
  your terminal.
- **OPD alignment dumps** (not logs, but same "where did it go" category):
  `/tmp/opd_dumps/` (`OPD_DUMP_DIR`) — teacher/student token alignment
  artifacts, see "Cross-Tokenizer Work" above.

Quick way to watch the teacher side live while iterating:

```bash
tail -f recipe/gkd/teacher/proxy.log recipe/gkd/teacher/worker.log
```

## Smoke vs Real Training (`cross_distill.sh`)

`cross_distill_smoke_1gpu.sh` is a shrunk-down version of the real training
script, `cross_distill.sh` (1 GPU / 5 steps / batch size 1, vs. 16 GPUs / 500
steps / batch size 128). Full diff, with which differences actually change
what the model learns vs. which are pure scale/plumbing:

| Param                                                                        | Smoke           | Real (`cross_distill.sh`)                          | Effect                                                                                        |
| ---------------------------------------------------------------------------- | --------------- | -------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| `train_prompt_bsz` / `ppo_mini_batch_size`                                   | 1               | 128                                                | **Accuracy-relevant** - see below                                                             |
| `total_training_steps` / `total_epochs`                                      | 5 / 1           | 500 / 10                                           | **Accuracy-relevant** - the lever for "did distillation actually happen"                      |
| `max_response_length`                                                        | 128             | 16384                                              | **Accuracy-relevant** - caps how long the student is allowed to reason                        |
| `val_kwargs.max_tokens`                                                      | 128             | 31744                                              | **Accuracy-relevant** - same cap, applied at validation                                       |
| `optim.lr_warmup_steps`                                                      | 1               | 10                                                 | Minor training-stability knob                                                                 |
| `data.max_prompt_length`                                                     | 256             | 1024                                               | Accuracy-relevant only if real prompts are long                                               |
| `TEACHER_MAX_SEQ_LEN`                                                        | 512             | 30720                                              | Must scale with response length or the teacher truncates                                      |
| `val_kwargs.n`                                                               | 1               | 4                                                  | Statistical reliability of the _reported_ accuracy, not the model itself                      |
| `NNODES`/`NGPUS_PER_NODE`                                                    | 1/1             | 2/8                                                | Pure scale, no accuracy effect                                                                |
| `sp_size`, `gen_tp`, `fsdp_size`                                             | 1, 1, 1         | 2, 2, 8                                            | Pure parallelism/memory, no accuracy effect                                                   |
| `gpu_memory_utilization`                                                     | 0.25            | 0.90                                               | Pure memory budget for vLLM KV cache                                                          |
| `actor_ppo_max_token_len`/`infer_ppo_max_token_len`/`max_num_batched_tokens` | hardcoded small | computed from prompt+response length               | Dynamic-batching token budget, no accuracy effect (just needs to be >= your longest sequence) |
| `test_freq`/`save_freq`                                                      | 5/5             | 25/500                                             | Logging/checkpoint cadence only                                                               |
| `logger`                                                                     | console         | console+wandb                                      | Tracking only                                                                                 |
| `attn_implementation=sdpa` override                                          | present         | absent (uses model default, presumably flash-attn) | Speed/memory, numerically near-identical, not an accuracy difference                          |
| auto-generate data / auto-start Ray                                          | present         | absent                                             | Convenience for iterating locally; real run assumes data + cluster already exist              |

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
