#!/usr/bin/env bash
set -xeuo pipefail

# One-GPU teacher server for OPD smoke tests.
#
# TEACHER_CKPT_PATH can be a HuggingFace model ID or a local checkpoint path:
#   TEACHER_CKPT_PATH=Qwen/Qwen2.5-0.5B-Instruct bash start_server_smoke_1gpu.sh

export PROXY_FRONTEND_PORT=${PROXY_FRONTEND_PORT:-15555}
export PROXY_BACKEND_PORT=${PROXY_BACKEND_PORT:-15556}
export VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD:-spawn}

BACKEND=${BACKEND:-vllm}
CKPT_PATH=${TEACHER_CKPT_PATH:-"Qwen/Qwen2.5-0.5B-Instruct"}
TP_SIZE=${TEACHER_TP_SIZE:-1}
N_LOGPROBS=${TEACHER_N_LOGPROBS:-1}
GPU_MEMORY_UTILIZATION=${TEACHER_GPU_MEMORY_UTILIZATION:-0.25}
MAX_NUM_BATCHED_TOKENS=${TEACHER_MAX_NUM_BATCHED_TOKENS:-512}
MAX_MODEL_LEN=${TEACHER_MAX_MODEL_LEN:-512}

wait_server_ready() {
    server=$1
    ip=$2
    port=$3
    while true; do
        echo "wait ${server} server ready at ${ip}:${port}..."
        result=$(echo -e "\n" | telnet "${ip}" "${port}" 2> /dev/null | grep Connected | wc -l)
        if [ "${result}" -eq 1 ]; then
            break
        fi
        sleep 1
    done
}

ps -ef | grep "python proxy.py" | grep -v grep | awk -F ' ' '{print $2}' | xargs -r kill -9
ps -ef | grep "python worker.py" | grep -v grep | awk -F ' ' '{print $2}' | xargs -r kill -9

nohup python3 proxy.py &> proxy.log &

wait_server_ready proxy localhost "${PROXY_BACKEND_PORT}"

echo "teacher proxy is ready"

nohup python3 worker.py \
    --backend "${BACKEND}" \
    --tp-size "${TP_SIZE}" \
    --n-logprobs "${N_LOGPROBS}" \
    --ckpt-path "${CKPT_PATH}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    &> worker.log &

echo "teacher worker started"
