#!/usr/bin/env bash
set -xeuo pipefail

# One-GPU OPD smoke test.
#
# PWD is a shell variable set by your terminal. It means the current working
# directory. If you run this script from the repo root, PWD is this repo path.
#
# RAY_DATA_HOME is this repo's workspace/data root for the smoke test. The
# script defaults it to "${PWD}/data", but you can override it:
#   RAY_DATA_HOME=/some/other/dir bash cross_distill_smoke_1gpu.sh
#
# MODEL_PATH is the student checkpoint. TEACHER_CKPT_PATH is the teacher
# checkpoint. They can be HuggingFace model IDs or local checkpoint paths.

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${REPO_ROOT}"

# Keep the driver, Ray head, and Ray workers on exactly the same Python
# environment. Reusing a Ray head started by a different Python is a common
# cause of workers staying alive without ever registering.
if [[ -z "${PYTHON_BIN:-}" ]]; then
    if [[ -x "${REPO_ROOT}/.venv/bin/python3" ]]; then
        PYTHON_BIN="${REPO_ROOT}/.venv/bin/python3"
    else
        PYTHON_BIN=$(command -v python3 || true)
    fi
fi
if [[ -z "${PYTHON_BIN}" || ! -x "${PYTHON_BIN}" ]]; then
    echo "No Python executable found. Set PYTHON_BIN or create ${REPO_ROOT}/.venv." >&2
    exit 1
fi

if [[ -z "${RAY_BIN:-}" ]]; then
    candidate_ray="$(dirname "${PYTHON_BIN}")/ray"
    if [[ -x "${candidate_ray}" ]]; then
        RAY_BIN="${candidate_ray}"
    else
        RAY_BIN=$(command -v ray || true)
    fi
fi
if [[ -z "${RAY_BIN}" || ! -x "${RAY_BIN}" ]]; then
    echo "Ray executable not found next to ${PYTHON_BIN}; set RAY_BIN explicitly." >&2
    exit 1
fi

project_name="ON_POLICY_DISTILL"

adv_estimator="opd"

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=${USE_KL_LOSS:-False}
kl_loss_coef=${KL_LOSS_COEF:-0.0}

clip_ratio_low=0.2
clip_ratio_high=0.28
opd_loss_max_clamp=${OPD_LOSS_MAX_CLAMP:-2.0}

max_prompt_length=${MAX_PROMPT_LENGTH:-256}
max_response_length=${MAX_RESPONSE_LENGTH:-640}
student_max_seq_len=${STUDENT_MAX_SEQ_LEN:-896}

if (( student_max_seq_len < max_prompt_length + max_response_length )); then
    echo "STUDENT_MAX_SEQ_LEN (${student_max_seq_len}) must cover prompt + response" \
        "($((max_prompt_length + max_response_length)))" >&2
    exit 1
fi

loss_agg_mode="token-mean"

# OPD is especially noisy at tiny batch sizes. Keep the PPO mini-batch equal
# to the rollout batch so each rollout produces one genuinely on-policy
# optimizer step. Dynamic token batching still splits the forward/backward
# work into memory-safe micro-batches on the single GPU.
train_prompt_bsz=${TRAIN_PROMPT_BATCH_SIZE:-32}
n_resp_per_prompt=${N_RESP_PER_PROMPT:-1}
train_prompt_mini_bsz=${PPO_MINI_BATCH_SIZE:-${train_prompt_bsz}}

if (( train_prompt_bsz % train_prompt_mini_bsz != 0 )); then
    echo "TRAIN_PROMPT_BATCH_SIZE must be divisible by PPO_MINI_BATCH_SIZE" >&2
    exit 1
fi
if (( train_prompt_mini_bsz != train_prompt_bsz )); then
    echo "Warning: PPO_MINI_BATCH_SIZE != TRAIN_PROMPT_BATCH_SIZE; OPD will perform multiple updates per rollout" >&2
fi

# Debug defaults: run a short, observable experiment before committing to a
# full training run. All values can be overridden from the environment.
total_epochs=${TOTAL_EPOCHS:-10}
total_training_steps=${TOTAL_TRAINING_STEPS:-1000}
test_freq=${TEST_FREQ:-200}
save_freq=${SAVE_FREQ:-200}
val_before_train=${VAL_BEFORE_TRAIN:-True}
# Was 2e-7: the repo's own real-scale config (cross_distill.sh) pairs lr=1e-6
# with train_prompt_bsz=128 -- a ~5x LR bump for a ~16x batch bump (sub-linear
# scaling, not full linear). At train_prompt_bsz going from 8 (an env override
# some prior run used, below even this script's own default) back up toward
# this script's own default of 32 (a 4x increase), a proportionate but
# conservative bump is ~2-3x, not the full 5x -- start here and watch
# grad_norm/entropy/OPD loss for instability before pushing further.
actor_lr=${ACTOR_LR:-5e-7}
actor_lr_warmup_steps=${ACTOR_LR_WARMUP_STEPS:-50}

NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-1}

RAY_DATA_HOME=${RAY_DATA_HOME:-"${PWD}/data"}
MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen2.5-0.5B-Instruct"}
TEACHER_CKPT_PATH=${TEACHER_CKPT_PATH:-"Qwen/Qwen2.5-0.5B-Instruct"}

# exp_name (and therefore CKPTS_DIR/resume_mode=auto below) is derived from the
# student/teacher model paths so that switching model pairs gets its own checkpoint
# dir instead of silently resuming from a previous, unrelated pair's checkpoint.
# Override EXP_NAME directly if you want a fixed name regardless of model paths.
student_tag=$(basename "${MODEL_PATH}")
teacher_tag=$(basename "${TEACHER_CKPT_PATH}")
exp_name=${EXP_NAME:-"9_rep_32"}

CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/smoke_ckpts/${project_name}/${exp_name}"}

# R1_ZERO_MODE: train on the r1_zero <think>/<answer> protocol (matching what
# lgsm8k_eval.py's grader actually checks) instead of this repo's own
# "#### number" instruction. Set R1_ZERO_MODE=1 to turn this on. This changes
# three things together, gated on the same flag: (1) default train/test files
# point at the r1_zero-prompted GSM8K data instead of the "####" smoke set,
# (2) that data gets auto-generated via gsm8k_r1_zero.py if missing, (3) a raw
# passthrough chat_template override is added below so prompts render as
# literal "{bos}{r1_zero text}" with no Llama/Qwen/OLMo role markers baked in
# -- matching the raw-prompt format R1-Zero-style teacher training/eval uses,
# instead of wrapping it inside this tokenizer's own native chat template.
R1_ZERO_MODE=${R1_ZERO_MODE:-0}
export R1_ZERO_MODE
if [[ "${R1_ZERO_MODE}" == "1" ]]; then
    TRAIN_FILE=${TRAIN_FILE:-"${RAY_DATA_HOME}/gsm8k_r1zero/train.parquet"}
    TEST_FILE=${TEST_FILE:-"${RAY_DATA_HOME}/gsm8k_r1zero/test.parquet"}
else
    TRAIN_FILE=${TRAIN_FILE:-"${RAY_DATA_HOME}/smoke/train.parquet"}
    TEST_FILE=${TEST_FILE:-"${RAY_DATA_HOME}/smoke/val.parquet"}
fi
ROLLOUT_DATA_DIR=${ROLLOUT_DATA_DIR:-"${CKPTS_DIR}/rollouts"}
VALIDATION_DATA_DIR=${VALIDATION_DATA_DIR:-"${CKPTS_DIR}/validation"}

export TEACHER_SERVER_IP=${TEACHER_SERVER_IP:-"127.0.0.1"}
export TEACHER_SERVER_PORT=${TEACHER_SERVER_PORT:-"15555"}
export TEACHER_N_WORKERS=${TEACHER_N_WORKERS:-"1"}
export TEACHER_CKPT_PATH
export TEACHER_MAX_SEQ_LEN=${TEACHER_MAX_SEQ_LEN:-"1280"}
export HYDRA_FULL_ERROR=1

# Reduce CUDA allocator fragmentation when the teacher (a separate process,
# often already resident on the same GPU) and this training process each run
# their own PyTorch/vLLM allocator.
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-"expandable_segments:True"}

export OPD_DUMP_DIR=${OPD_DUMP_DIR:-"/tmp/opd_dumps"}
export OPD_DUMP_NUM_SEQS=${OPD_DUMP_NUM_SEQS:-"1"}
export OPD_DUMP_MAX_STEPS=${OPD_DUMP_MAX_STEPS:-"3"}
export OPD_TOKENIZER_DEBUG=${OPD_TOKENIZER_DEBUG:-"1"}

# Keep the terminal useful during debugging: show one student rollout and only
# the high-signal metrics each step. Full rollouts remain in ROLLOUT_DATA_DIR.
export VERL_CONSOLE_LOG_MODE=${VERL_CONSOLE_LOG_MODE:-"debug"}
export VERL_CONSOLE_ROLLOUT_SAMPLES=${VERL_CONSOLE_ROLLOUT_SAMPLES:-"1"}
export VERL_CONSOLE_ROLLOUT_MAX_CHARS=${VERL_CONSOLE_ROLLOUT_MAX_CHARS:-"2000"}

temperature=${TEMPERATURE:-1.0}
top_p=${TOP_P:-1.0}
top_k=${TOP_K:--1}

val_top_p=1.0
val_top_k=-1
val_temperature=0.0

sp_size=1
use_dynamic_bsz=True
actor_ppo_max_token_len=${ACTOR_PPO_MAX_TOKEN_LEN:-${student_max_seq_len}}
infer_ppo_max_token_len=${INFER_PPO_MAX_TOKEN_LEN:-${student_max_seq_len}}
offload=True
gen_tp=1
fsdp_size=1
actor_model_dtype=${ACTOR_MODEL_DTYPE:-float32}

# Was hardcoded to 1: vLLM generated rollouts one sequence at a time
# regardless of train_prompt_bsz, so raising the batch size mostly bought
# wall-clock time, not throughput. A small concurrency bump lets several
# sequences generate together -- watch for vLLM OOM (this shares the GPU with
# the teacher process per the smoke-test setup) and lower this first, before
# touching train_prompt_bsz or gpu_memory_utilization, if it OOMs.
max_num_seqs=${MAX_NUM_SEQS:-4}
rollout_enforce_eager=${ROLLOUT_ENFORCE_EAGER:-True}
# vLLM's batched-token scheduling budget must cover max_num_seqs sequences
# concurrently, not just one -- scale it with max_num_seqs instead of leaving
# it pinned at a single sequence's length.
max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS:-$((max_num_seqs * student_max_seq_len))}

if [[ ! -f "${TRAIN_FILE}" || ! -f "${TEST_FILE}" ]]; then
    if [[ "${R1_ZERO_MODE}" == "1" ]]; then
        "${PYTHON_BIN}" examples/data_preprocess/gsm8k_r1_zero.py --local_save_dir "${RAY_DATA_HOME}/gsm8k_r1zero"
    else
        "${PYTHON_BIN}" examples/data_preprocess/create_opd_smoke_data.py --output-dir "${RAY_DATA_HOME}/smoke"
    fi
fi

# Raw passthrough chat_template: renders a single-user-turn `messages` list as
# literally "{bos_token}{content}" -- no role headers, no <|eot_id|>/<|im_end|>
# etc. baked into the PROMPT side by either tokenizer's own native template.
#
# The leading/trailing \" are load-bearing, not decorative: Hydra's own CLI
# override grammar tries to parse any value starting with "{" as a structured
# dict literal, and chokes on Jinja's "{{" with "no viable alternative at
# input '{{ '". Wrapping the value in literal double-quote characters (not
# just shell quoting -- verified against hydra's actual OverridesParser)
# forces Hydra to treat it as an opaque string instead. Single quotes for the
# dict key inside are safe precisely because they're not Hydra's quote
# character.
r1_zero_chat_template="\"{{ bos_token }}{{ messages[0]['content'] }}\""
EXTRA_HYDRA_ARGS=()
if [[ "${R1_ZERO_MODE}" == "1" ]]; then
    # "+" (not plain "=") is required: apply_chat_template_kwargs starts as an
    # empty dict in the base config (struct mode), so this is adding a new key,
    # not overriding an existing one -- Hydra's own error message for the
    # plain "=" form says so directly.
    EXTRA_HYDRA_ARGS+=("+data.apply_chat_template_kwargs.chat_template=${r1_zero_chat_template}")
fi

if [[ "${AUTO_START_RAY:-1}" == "1" ]]; then
    "${RAY_BIN}" status >/dev/null 2>&1 || \
        "${RAY_BIN}" start --head --num-gpus="${NGPUS_PER_NODE}" --include-dashboard=false
fi

"${PYTHON_BIN}" -m verl.trainer.main_ppo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.shuffle=True \
    data.seed=44 \
    data.dataloader_num_workers=0 \
    data.filter_overlong_prompts_workers=1 \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_prompt_bsz} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.policy_loss.opd_loss_max_clamp=${opd_loss_max_clamp} \
    actor_rollout_ref.model.use_remove_padding=True \
    +actor_rollout_ref.model.override_config.max_position_embeddings=${student_max_seq_len} \
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.actor.policy_loss.loss_mode="opd" \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=sync \
    actor_rollout_ref.rollout.enforce_eager=${rollout_enforce_eager} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=${actor_lr} \
    actor_rollout_ref.actor.optim.lr_warmup_steps=${actor_lr_warmup_steps} \
    actor_rollout_ref.actor.optim.weight_decay=0.01 \
    actor_rollout_ref.actor.optim.optimizer=AdamW8bit \
    actor_rollout_ref.actor.optim.optimizer_impl=bitsandbytes.optim \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.model_dtype=${actor_model_dtype} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.nccl_timeout=72000 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=${max_num_batched_tokens} \
    actor_rollout_ref.rollout.max_num_seqs=${max_num_seqs} \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${val_temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${val_top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=False \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.max_tokens=${max_response_length} \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=${fsdp_size} \
    actor_rollout_ref.ref.fsdp_config.fsdp_size=${fsdp_size} \
    reward_model.reward_manager=opd \
    reward_model.enable=False \
    trainer.logger='["console"]' \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
    trainer.nnodes="${NNODES}" \
    trainer.val_before_train=${val_before_train} \
    trainer.test_freq=${test_freq} \
    trainer.save_freq=${save_freq} \
    trainer.total_epochs=${total_epochs} \
    trainer.total_training_steps=${total_training_steps} \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.rollout_data_dir="${ROLLOUT_DATA_DIR}" \
    trainer.validation_data_dir="${VALIDATION_DATA_DIR}" \
    trainer.resume_mode=auto \
    trainer.log_val_generations=1 \
    ${EXTRA_HYDRA_ARGS[@]+"${EXTRA_HYDRA_ARGS[@]}"}
