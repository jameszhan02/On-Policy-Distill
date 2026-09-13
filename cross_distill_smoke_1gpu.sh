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

project_name="ON_POLICY_DISTILL"

adv_estimator="opd"

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

clip_ratio_low=0.2
clip_ratio_high=0.28
opd_loss_max_clamp=${OPD_LOSS_MAX_CLAMP:-2.0}

max_prompt_length=256
max_response_length=512

loss_agg_mode="token-mean"

train_prompt_bsz=4
n_resp_per_prompt=1
train_prompt_mini_bsz=2

# Debug defaults: run a short, observable experiment before committing to a
# full training run. All values can be overridden from the environment.
total_epochs=${TOTAL_EPOCHS:-10}
total_training_steps=${TOTAL_TRAINING_STEPS:-300}
test_freq=${TEST_FREQ:-50}
save_freq=${SAVE_FREQ:-50}
val_before_train=${VAL_BEFORE_TRAIN:-True}
actor_lr=${ACTOR_LR:-2e-7}
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
exp_name=${EXP_NAME:-"OPD_DEBUG_1GPU_${student_tag}_to_${teacher_tag}"}

CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/smoke_ckpts/${project_name}/${exp_name}"}
TRAIN_FILE=${TRAIN_FILE:-"${RAY_DATA_HOME}/smoke/train.parquet"}
TEST_FILE=${TEST_FILE:-"${RAY_DATA_HOME}/smoke/val.parquet"}
ROLLOUT_DATA_DIR=${ROLLOUT_DATA_DIR:-"${CKPTS_DIR}/rollouts"}
VALIDATION_DATA_DIR=${VALIDATION_DATA_DIR:-"${CKPTS_DIR}/validation"}

export TEACHER_SERVER_IP=${TEACHER_SERVER_IP:-"127.0.0.1"}
export TEACHER_SERVER_PORT=${TEACHER_SERVER_PORT:-"15555"}
export TEACHER_N_WORKERS=${TEACHER_N_WORKERS:-"1"}
export TEACHER_CKPT_PATH
export TEACHER_MAX_SEQ_LEN=${TEACHER_MAX_SEQ_LEN:-"1024"}
export HYDRA_FULL_ERROR=1

# Reduce CUDA allocator fragmentation when the teacher (a separate process,
# often already resident on the same GPU) and this training process each run
# their own PyTorch/vLLM allocator.
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-"expandable_segments:True"}

export OPD_DUMP_DIR=${OPD_DUMP_DIR:-"/tmp/opd_dumps"}
export OPD_DUMP_NUM_SEQS=${OPD_DUMP_NUM_SEQS:-"1"}
export OPD_DUMP_MAX_STEPS=${OPD_DUMP_MAX_STEPS:-"3"}

temperature=1.0
top_p=1.0
top_k=-1

val_top_p=1.0
val_top_k=-1
val_temperature=0.0

sp_size=1
use_dynamic_bsz=True
actor_ppo_max_token_len=768
infer_ppo_max_token_len=768
offload=True
gen_tp=1
fsdp_size=1

if [[ ! -f "${TRAIN_FILE}" || ! -f "${TEST_FILE}" ]]; then
    python3 examples/data_preprocess/create_opd_smoke_data.py --output-dir "${RAY_DATA_HOME}/smoke"
fi

if [[ "${AUTO_START_RAY:-1}" == "1" ]]; then
    ray status >/dev/null 2>&1 || ray start --head --num-gpus="${NGPUS_PER_NODE}" --include-dashboard=false
fi

python3 -m verl.trainer.main_ppo \
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
    +actor_rollout_ref.model.override_config.max_position_embeddings=768 \
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
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.nccl_timeout=72000 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=768 \
    actor_rollout_ref.rollout.max_num_seqs=1 \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${val_temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${val_top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=False \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.max_tokens=512 \
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
    trainer.log_val_generations=1
