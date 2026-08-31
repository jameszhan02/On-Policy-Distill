# OPD Custom Experiment TODO

This document tracks the work needed to use this repo for a custom
cross-tokenizer OPD experiment, plus a one-GPU 4090 smoke test.

## Scope Judgment

The change size is moderate, not huge, if the goal is to run OPD on two
HuggingFace causal LM checkpoints.

Most required changes are configuration and validation:

- Point training and teacher scripts to your own student/teacher checkpoints.
- Prepare a small parquet dataset with the expected prompt column.
- Shrink batch size, sequence length, and parallelism for a 24 GB 4090.
- Verify teacher server connectivity.
- Verify tokenizer/chat-template conversion and alignment quality.

The only likely code change is tokenizer-pair support in
`verl/workers/reward_manager/opd.py`, unless your model pair is already covered
by the existing LLaMA/Qwen/DeepSeek mappings.

## Required Decisions

- [ ] Choose student checkpoint.
- [ ] Choose teacher checkpoint.
- [ ] Confirm both checkpoints are HuggingFace-compatible local paths or model
      IDs.
- [ ] Identify student tokenizer family and chat template.
- [ ] Identify teacher tokenizer family and chat template.
- [ ] Decide whether to use the repo's string-replacement template mapping or
      replace it with message-level retokenization.
- [ ] Decide first smoke-test pair:
      - same-tokenizer pair for pipeline validation, or
      - real cross-tokenizer pair for immediate alignment validation.

## Files Most Likely To Change

- [ ] `cross_distill.sh`
      - student model path
      - train/validation parquet paths
      - checkpoint output path
      - batch sizes
      - sequence lengths
      - GPU/node counts
      - FSDP/vLLM parallelism
      - logger settings
- [ ] `recipe/gkd/teacher/start_server.sh`
      - teacher checkpoint path
      - vLLM tensor parallel size
      - vLLM memory fraction
      - max model length
      - max batched tokens
- [ ] `verl/workers/reward_manager/opd.py`
      - only if the student/teacher tokenizer pair is not supported
      - add or replace chat-template conversion logic
      - optionally improve alignment debug output
- [ ] Dataset generation script or notebook
      - create parquet with the prompt field expected by `data.prompt_key`

## One-GPU 4090 Smoke Test Plan

Goal: validate the full OPD pipeline, not model quality.

Use these helper scripts:

- `examples/data_preprocess/create_opd_smoke_data.py`
- `recipe/gkd/teacher/start_server_smoke_1gpu.sh`
- `cross_distill_smoke_1gpu.sh`

Pipeline to verify:

1. Student rollout generates responses.
2. Teacher server receives retokenized requests.
3. Teacher returns top-k logprobs and token IDs.
4. OPD reward manager aligns teacher tokens back to student tokens.
5. OPD policy loss runs without NaN/inf instability.
6. Optimizer completes a few steps.
7. Checkpoint save works.

## Smoke Test Model Constraints

- [ ] Use very small models first.
- [ ] Prefer student <= 0.5B-1B.
- [ ] Prefer teacher <= 0.5B-1.5B.
- [ ] Avoid 7B teacher + 7B student on one 24 GB card.
- [ ] Use `--tp-size 1` for teacher server.
- [ ] Use `gen_tp=1`, `fsdp_size=1`, `sp_size=1` for training.

Suggested first pass:

- student: small Qwen or LLaMA-family checkpoint
- teacher: small Qwen or DeepSeek/Qwen-family checkpoint
- dataset: 16 short prompts
- max prompt length: 256
- max response length: 128
- train batch size: 1
- total training steps: 5

## Smoke Test `cross_distill.sh` TODO

- [ ] Set single-node/single-GPU mode:

```bash
NNODES=1
NGPUS_PER_NODE=1
```

- [ ] Shrink sequence lengths:

```bash
max_prompt_length=256
max_response_length=128
```

- [ ] Shrink batch sizes:

```bash
train_prompt_bsz=1
n_resp_per_prompt=1
train_prompt_mini_bsz=1
```

- [ ] Disable large parallelism:

```bash
sp_size=1
gen_tp=1
fsdp_size=1
offload=True
```

- [ ] Shrink token budgets:

```bash
actor_ppo_max_token_len=512
infer_ppo_max_token_len=512
```

- [ ] Reduce rollout memory:

```bash
actor_rollout_ref.rollout.gpu_memory_utilization=0.25
actor_rollout_ref.rollout.max_num_batched_tokens=512
actor_rollout_ref.rollout.tensor_model_parallel_size=1
```

- [ ] Disable validation pressure:

```bash
trainer.val_before_train=False
trainer.test_freq=100000
actor_rollout_ref.rollout.val_kwargs.n=1
actor_rollout_ref.rollout.val_kwargs.max_tokens=128
```

- [ ] Use console logging first:

```bash
trainer.logger='["console"]'
```

- [ ] Keep run short:

```bash
trainer.total_epochs=1
trainer.total_training_steps=5
trainer.save_freq=5
```

## Smoke Test Teacher Server TODO

- [ ] Use `recipe/gkd/teacher/start_server_smoke_1gpu.sh`.
- [ ] Set teacher checkpoint:

```bash
export TEACHER_CKPT_PATH="/path/to/tiny-teacher"
```

- [ ] Use single-GPU vLLM:

```bash
--tp-size 1
```

- [ ] Use top-1 logprobs first:

```bash
--n-logprobs 1
```

- [ ] Shrink vLLM memory and context:

```bash
--gpu-memory-utilization 0.25
--max-num-batched-tokens 512
--max-model-len 512
```

- [ ] Export matching training-side teacher limits:

```bash
export TEACHER_SERVER_IP=127.0.0.1
export TEACHER_SERVER_PORT=15555
export TEACHER_N_WORKERS=1
export TEACHER_MAX_SEQ_LEN=512
```

Start command:

```bash
cd recipe/gkd/teacher
TEACHER_CKPT_PATH=Qwen/Qwen2.5-0.5B-Instruct bash start_server_smoke_1gpu.sh
```

## Dataset TODO

- [ ] Create a tiny parquet file with a `prompt` column.
- [ ] Start with 16 short prompts.
- [ ] Avoid long chain-of-thought prompts for the first smoke test.
- [ ] Confirm `data.prompt_key=prompt` matches the parquet column.
- [ ] Create a tiny validation parquet or point `TEST_FILE` to the same small
      file for smoke testing.

Example prompt contents:

```text
What is 1+1?
Solve: 3 + 5 =
Write one sentence about gravity.
Answer briefly: what is water?
```

Create smoke data:

```bash
python3 examples/data_preprocess/create_opd_smoke_data.py --output-dir data/smoke
```

Run smoke training from the repo root:

```bash
MODEL_PATH=Qwen/Qwen2.5-0.5B-Instruct \
TEACHER_CKPT_PATH=Qwen/Qwen2.5-0.5B-Instruct \
bash cross_distill_smoke_1gpu.sh
```

## Tokenizer Pair TODO

- [ ] Print or inspect each tokenizer's `chat_template`.
- [ ] Print special tokens for both tokenizers.
- [ ] Check whether `_detect_model_family()` classifies both tokenizers
      correctly.
- [ ] If unsupported, add mapping in `_build_chat_template_mapping()` or replace
      the approach with message-level retokenization.
- [ ] Enable alignment dumps for the first cross-tokenizer run:

```bash
export OPD_DUMP_DIR=/tmp/opd_dumps
export OPD_DUMP_NUM_SEQS=1
export OPD_DUMP_MAX_STEPS=3
```

- [ ] Inspect dumps for:
      - teacher text matching student semantic content
      - low fallback ratio
      - low `inf` sentinel ratio
      - correct response-token region

## How To Identify Tokenizer Family And Template

Use the local inspection helper:

```bash
python tools/opd_inspect_tokenizer_pair.py \
  --student /path/to/student/checkpoint \
  --teacher /path/to/teacher/checkpoint
```

The script prints, for both student and teacher:

- tokenizer class
- repo heuristic family: `llama`, `qwen`, `deepseek`, or `unknown`
- BOS/EOS/PAD tokens
- `special_tokens_map`
- relevant added special tokens
- raw `chat_template`
- rendered sample chat text
- first rendered token IDs and decoded token strings

Use this output to answer:

- What exact string marks the start of a chat?
- What exact string marks `system`, `user`, and `assistant` roles?
- What exact string terminates one turn?
- Does `apply_chat_template(..., add_generation_prompt=True)` end with the
  teacher's expected assistant prefix?
- Does the student decoded text contain special-token strings that can be
  transformed into the teacher rendered text by deterministic replacement?

## How To Add A Mapping

Mappings live in `TeacherClient._build_chat_template_mapping()` inside
`verl/workers/reward_manager/opd.py`.

Add a new branch for your pair:

```python
elif student_family == "your_student_family" and teacher_family == "your_teacher_family":
    return [
        ("student_long_compound_marker", "teacher_marker"),
        ("student_short_marker", "teacher_marker"),
    ]
```

Rules are applied in order with `str.replace()`, so put longer compound strings
before shorter substrings.

After adding a mapping:

- [ ] Run the inspection script again and compare rendered formats.
- [ ] Run a tiny OPD smoke test with `OPD_DUMP_DIR` enabled.
- [ ] Check that `teacher_text` has the teacher's chat format.
- [ ] Check that response text is semantically unchanged.
- [ ] Check that fallback/`inf` alignment ratio is low.

## Bring-Up Order

- [ ] Install dependencies in an isolated environment.
- [ ] Start teacher server with tiny teacher.
- [ ] Run teacher server standalone test if available.
- [ ] Create tiny parquet dataset.
- [ ] Run same-tokenizer smoke test if possible.
- [ ] Run cross-tokenizer smoke test with alignment dumps.
- [ ] Fix template mapping/alignment issues.
- [ ] Run 5 OPD training steps.
- [ ] Confirm checkpoint save.
- [ ] Increase sequence length gradually.
- [ ] Increase batch size gradually.
- [ ] Move to real model pair.

## Success Criteria For Smoke Test

- [ ] Teacher server starts without CUDA OOM.
- [ ] Training starts without Ray/vLLM placement errors.
- [ ] Teacher requests return without timeout.
- [ ] OPD loss executes.
- [ ] No NaN in loss or logprob metrics.
- [ ] Alignment dump is mostly meaningful.
- [ ] At least one checkpoint is written.

## Known Risks

- Single 4090 may not fit both teacher vLLM and student training unless both
  models are very small.
- vLLM and actor training sharing one GPU may fragment memory.
- Unsupported chat templates can produce plausible-looking but wrong teacher
  logprobs.
- High fallback/`inf` ratio means the OPD signal is mostly skipped.
- Long responses make teacher server memory and latency much worse.
- 7B-scale experiments should be treated as multi-GPU or remote-teacher work.
