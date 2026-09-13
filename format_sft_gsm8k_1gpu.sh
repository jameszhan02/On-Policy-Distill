#!/usr/bin/env bash
set -euo pipefail

# Short span-selective SFT warmup for reliable GSM8K answer formatting.
# All reasoning stays in context, while loss is applied only to the exact
# `#### <number>` answer span and the terminating special token.

RAY_DATA_HOME=${RAY_DATA_HOME:-"${PWD}/data"}
MODEL_PATH=${MODEL_PATH:-"meta-llama/Llama-3.2-1B-Instruct"}

model_tag=$(basename "${MODEL_PATH}")
SFT_OUTPUT_DIR=${SFT_OUTPUT_DIR:-"${RAY_DATA_HOME}/format_sft_ckpts/${model_tag}_gsm8k_format_span"}
SFT_DATA_DIR=${SFT_DATA_DIR:-"${RAY_DATA_HOME}/gsm8k"}
SFT_TRAIN_FILE=${SFT_TRAIN_FILE:-"${SFT_DATA_DIR}/train.parquet"}
SFT_VAL_FILE=${SFT_VAL_FILE:-"${SFT_DATA_DIR}/test.parquet"}

SFT_TRAIN_SAMPLES=${SFT_TRAIN_SAMPLES:-512}
SFT_VAL_SAMPLES=${SFT_VAL_SAMPLES:-128}
SFT_TRAIN_BATCH_SIZE=${SFT_TRAIN_BATCH_SIZE:-8}
SFT_MICRO_BATCH_SIZE=${SFT_MICRO_BATCH_SIZE:-1}
SFT_MAX_LENGTH=${SFT_MAX_LENGTH:-896}
SFT_LR=${SFT_LR:-5e-6}
SFT_EPOCHS=${SFT_EPOCHS:-1}
SFT_SEED=${SFT_SEED:-44}
SFT_FORMAT_LOSS_MODE=${SFT_FORMAT_LOSS_MODE:-format}
SFT_MODEL_DTYPE=${SFT_MODEL_DTYPE:-bf16}
SFT_OPTIMIZER=${SFT_OPTIMIZER:-AdamW8bit}
SFT_OPTIMIZER_IMPL=${SFT_OPTIMIZER_IMPL:-bitsandbytes.optim}
SFT_FSDP_STRATEGY=${SFT_FSDP_STRATEGY:-fsdp}

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-"expandable_segments:True"}
export HYDRA_FULL_ERROR=${HYDRA_FULL_ERROR:-1}

if (( SFT_TRAIN_BATCH_SIZE % SFT_MICRO_BATCH_SIZE != 0 )); then
    echo "SFT_TRAIN_BATCH_SIZE must be divisible by SFT_MICRO_BATCH_SIZE" >&2
    exit 1
fi

if [[ "${SFT_FORMAT_LOSS_MODE}" != "format" && "${SFT_FORMAT_LOSS_MODE}" != "full" ]]; then
    echo "SFT_FORMAT_LOSS_MODE must be 'format' or 'full'" >&2
    exit 1
fi

if [[ ! -f "${SFT_TRAIN_FILE}" || ! -f "${SFT_VAL_FILE}" ]]; then
    echo "GSM8K parquet files not found; preparing them in ${SFT_DATA_DIR}"
    python3 examples/data_preprocess/gsm8k.py --local_save_dir "${SFT_DATA_DIR}"
fi

echo
echo "=== GSM8K format SFT ==="
echo "Base model:       ${MODEL_PATH}"
echo "Train data:       ${SFT_TRAIN_FILE}"
echo "Train samples:    ${SFT_TRAIN_SAMPLES}"
echo "Global batch:     ${SFT_TRAIN_BATCH_SIZE}"
echo "Micro batch:      ${SFT_MICRO_BATCH_SIZE}"
echo "Max length:       ${SFT_MAX_LENGTH}"
echo "Learning rate:    ${SFT_LR}"
echo "Format loss mode: ${SFT_FORMAT_LOSS_MODE}"
echo "Model dtype:      ${SFT_MODEL_DTYPE}"
echo "Optimizer:        ${SFT_OPTIMIZER} (${SFT_OPTIMIZER_IMPL})"
echo "FSDP strategy:    ${SFT_FSDP_STRATEGY}"
echo "Epochs:           ${SFT_EPOCHS}"
echo "Output directory: ${SFT_OUTPUT_DIR}"
echo

torchrun --standalone --nnodes=1 --nproc-per-node=1 \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files="${SFT_TRAIN_FILE}" \
    data.val_files="${SFT_VAL_FILE}" \
    data.prompt_key=extra_info \
    data.response_key=extra_info \
    'data.prompt_dict_keys=["question"]' \
    'data.response_dict_keys=["answer"]' \
    data.custom_cls.path=pkg://verl.utils.dataset.format_sft_dataset \
    data.custom_cls.name=FormatSFTDataset \
    +data.format_loss_mode="${SFT_FORMAT_LOSS_MODE}" \
    data.train_max_samples="${SFT_TRAIN_SAMPLES}" \
    data.val_max_samples="${SFT_VAL_SAMPLES}" \
    +data.shuffle=True \
    +data.seed="${SFT_SEED}" \
    data.train_batch_size="${SFT_TRAIN_BATCH_SIZE}" \
    data.micro_batch_size_per_gpu="${SFT_MICRO_BATCH_SIZE}" \
    data.max_length="${SFT_MAX_LENGTH}" \
    data.truncation=left \
    optim.lr="${SFT_LR}" \
    optim.optimizer="${SFT_OPTIMIZER}" \
    optim.optimizer_impl="${SFT_OPTIMIZER_IMPL}" \
    optim.lr_warmup_steps_ratio=0.1 \
    optim.weight_decay=0.01 \
    optim.clip_grad=1.0 \
    model.partial_pretrain="${MODEL_PATH}" \
    model.strategy="${SFT_FSDP_STRATEGY}" \
    model.fsdp_config.model_dtype="${SFT_MODEL_DTYPE}" \
    model.enable_gradient_checkpointing=True \
    trainer.default_local_dir="${SFT_OUTPUT_DIR}" \
    trainer.project_name=ON_POLICY_DISTILL \
    trainer.experiment_name="${model_tag}_gsm8k_format_sft" \
    trainer.logger=console \
    trainer.total_epochs="${SFT_EPOCHS}" \
    trainer.test_freq=-1 \
    trainer.save_freq=-1 \
    trainer.resume_mode=disable \
    trainer.max_ckpt_to_keep=1 \
    'trainer.checkpoint.save_contents=["hf_model"]' \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=1 \
    ulysses_sequence_parallel_size=1 \
    use_remove_padding=True

latest_checkpoint=$(find "${SFT_OUTPUT_DIR}" -maxdepth 1 -type d -name 'global_step_*' -print | sort -V | tail -n 1)
if [[ -z "${latest_checkpoint}" || ! -d "${latest_checkpoint}/huggingface" ]]; then
    echo "SFT completed, but no HuggingFace checkpoint was found under ${SFT_OUTPUT_DIR}" >&2
    exit 1
fi

echo
echo "=== Format SFT complete ==="
echo "HuggingFace model: ${latest_checkpoint}/huggingface"
echo
echo "Use it for OPD with:"
printf 'MODEL_PATH=%q EXP_NAME=OPD_AFTER_FORMAT_SFT bash cross_distill_smoke_1gpu.sh\n' \
    "${latest_checkpoint}/huggingface"
