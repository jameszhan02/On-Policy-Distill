# OPD Engineering Learning Path — Checkpoint Todolist

This file is your TA-authored roadmap. Each stage has:
- **Read** — files to read before writing any code
- **Checkpoint** — what you build yourself (type it, don't paste)
- **Key questions** — things to think about while reading

Work through stages in order. The goal is to understand *why* each framework
choice (Ray, verl, vLLM) exists, not just how to call its API.

---

## Stage 0 — Run The Existing Demo (Warmup)

**Estimated time:** 30 min

### Read
- `playground/mini_verl_opd_flow.py` — read every line, top to bottom
- `playground/README.md` — look at the mapping table

### Do
```bash
python3 playground/mini_verl_opd_flow.py
```

### Key questions to answer before moving on
- [ ] What is `DataProto`? What three dicts does it hold?
- [ ] What does `.union()` do? Why is it called instead of mutating in place?
- [ ] Where does the "advantage" come from in this toy? Is it a real advantage?
- [ ] What shape is `responses`? What shape is `token_level_scores`?

### Checkpoint 0 — nothing to write yet
Just run the script, trace the data flow on paper, answer the questions above.

---

## Stage 1 — Understand DataProto (The Envelope)

**Why:** Every worker in verl passes data through `DataProto`. If you don't
understand this struct, the rest of the code is noise.

### Read
1. `verl/protocol.py` — search for `class DataProto`. Read the docstring and
   the field definitions. You do NOT need to read every method now.
   Focus on: `batch`, `non_tensor_batch`, `meta_info`, `.union()`, `.select()`.

### Key questions
- [ ] In the real `DataProto`, how does `.pop()` differ from `.select()`?
- [ ] Why is `non_tensor_batch` a separate dict from `batch`?
  (Hint: tensors need to be moved to GPU; Python objects don't.)
- [ ] What does `meta_info` typically hold?

### Checkpoint 1 — Toy DataProto
Create `playground/toy_dataproto.py`. Implement:
- A `DataProto` dataclass with `batch`, `non_tensor_batch`, `meta_info`
- A `.union(other)` method — merge two DataProtos (other wins on key conflicts)
- A `.select(keys)` method — return a new DataProto with only those keys in batch
- A small `__repr__` so printing shows key names and tensor shapes

**TA hint:** `.select()` should not copy tensor data, just filter the dict.

---

## Stage 2 — Why Ray? (The Distributed Glue)

**Why:** verl runs workers on multiple GPUs/machines. Ray is the framework that
lets Python code call functions on remote processes as if they were local.

### Read
1. `examples/ray/` — look at any `.py` file there for a minimal Ray example
2. `verl/trainer/ppo/ray_trainer.py` — find the section that calls
   `WorkerGroup.generate_sequences.remote()`. Read ~50 lines around it.
3. Search for `@ray.remote` in the verl codebase:
   ```
   grep -r "@ray.remote" verl/ --include="*.py" -l
   ```
   Pick one file and look at how the class is decorated.

### Key questions
- [ ] What does `@ray.remote` do to a class?
- [ ] What does `.remote()` return — a result or a future?
- [ ] What does `ray.get(...)` do? When does it block?
- [ ] Why would you want rollout and training to be on *separate* Ray actors?

### Checkpoint 2 — Toy Ray Pipeline (no GPU needed)
Create `playground/toy_ray_pipeline.py`. Implement:

```python
import ray

@ray.remote
class RolloutActor:
    def generate(self, prompt: str) -> str:
        # pretend model: reverse the prompt string
        return prompt[::-1]

@ray.remote
class TrainActor:
    def update(self, response: str) -> dict:
        # pretend loss: length of response
        return {"loss": len(response)}

def main():
    ray.init()
    rollout = RolloutActor.remote()
    trainer = TrainActor.remote()

    prompts = ["hello", "world", "ray"]
    for p in prompts:
        # YOUR CODE: call generate, then update, collect metrics
        # remember: .remote() returns a future, ray.get() unwraps it
        pass

    ray.shutdown()

if __name__ == "__main__":
    main()
```

Fill in the `for` loop. Print the metrics. Then add a second rollout actor and
send half the prompts to each — observe that they run concurrently.

**TA hint:** `ray.get([future1, future2])` unwraps a list of futures at once.

---

## Stage 3 — WorkerGroup (How verl Wraps Ray)

**Why:** verl doesn't call `actor.remote()` directly everywhere. It wraps
groups of Ray actors into a `WorkerGroup` that broadcasts or scatters calls.

### Read
1. `verl/single_controller/` — look for the WorkerGroup class definition
2. `verl/trainer/ppo/ray_trainer.py` — look for `WorkerGroup` usage around
   `init_workers()`. Notice how `actor_rollout_wg` is created.

### Key questions
- [ ] What is the difference between a "broadcast" and a "scatter" call on a WorkerGroup?
- [ ] Why would you broadcast model weights but scatter data batches?
- [ ] What does `wg.execute_all_sync(method_name, *args)` roughly do?

### Checkpoint 3 — Toy WorkerGroup
Extend `toy_ray_pipeline.py`. Add a `WorkerGroup` class that:
- Holds a list of Ray actor handles
- Has a `scatter_call(method, items)` — split items across actors, call in parallel, gather results
- Has a `broadcast_call(method, item)` — call the same item on all actors, gather results

**TA hint:** Use a list comprehension of `.remote()` calls, then `ray.get()`.

---

## Stage 4 — Rollout (How Tokens Get Generated)

**Why:** In a real trainer, the student model generates token sequences using
vLLM. Understanding this helps you see what `generate_sequences()` actually returns.

### Read
1. `verl/workers/rollout/vllm_rollout/` — look at the class structure (don't
   read every line, just scan method names and their docstrings/comments)
2. In `mini_verl_opd_flow.py`: re-read `RolloutWorker.generate_sequences()`
   and the `DataProto` it returns

### Key questions
- [ ] What fields does `generate_sequences()` add to the batch?
- [ ] What is `response_mask` used for? (Hint: padding)
- [ ] Why does vLLM run as a *separate* process/actor rather than in the same
   process as the training loop?

### Checkpoint 4 — Toy Rollout with Temperature
Modify your `RolloutActor.generate()` from Stage 2 to simulate temperature
sampling. Instead of reversing the string, treat each character as a token and
sample the next character from a small probability table with a `temperature`
parameter (higher temp = more uniform distribution).

Return a dict with:
- `responses`: list of sampled token strings
- `response_mask`: list of 1s (same length, all valid — no padding yet)

**TA hint:** `temperature` scales the logits before softmax. `torch.multinomial`
samples from a probability distribution.

---

## Stage 5 — Reward and Advantage (The OPD Signal)

**Why:** This is the heart of on-policy distillation. The teacher's log-probs
become the training signal instead of a scalar reward.

### Read
1. `verl/workers/reward_manager/opd.py` — read the `__call__` method carefully
2. `verl/trainer/ppo/core_algos.py` — search for `opd` and read the advantage
   estimator for OPD
3. In `mini_verl_opd_flow.py`: re-read `OPDRewardManager.__call__()` and
   `compute_advantage()`

### Key questions
- [ ] In OPD, what is used as the "advantage"? Is it a scalar per sequence or
   a vector per token?
- [ ] How does this differ from standard PPO where advantage is a scalar (GAE)?
- [ ] Why subtract `old_log_probs` in `ActorWorker.update_actor()`?
   (Hint: think about importance sampling)

### Checkpoint 5 — OPD Advantage From Scratch
In a new file `playground/toy_advantage.py`, implement:

```python
def compute_opd_advantage(
    teacher_log_probs: torch.Tensor,   # [batch, seq_len]
    old_student_log_probs: torch.Tensor,  # [batch, seq_len]
    response_mask: torch.Tensor,       # [batch, seq_len]
) -> torch.Tensor:
    """
    YOUR CODE HERE.
    Return a [batch, seq_len] tensor of per-token advantages.
    Mask out padding positions (set to 0 where response_mask == 0).
    """
    ...
```

Then write a `test_compute_opd_advantage()` function that:
- Creates toy tensors
- Calls your function
- Asserts output shape is correct
- Asserts masked positions are zero

---

## Stage 6 — Actor Update (The Policy Loss)

**Why:** The policy gradient loss is where all the signal flows into the model weights.

### Read
1. `verl/trainer/ppo/core_algos.py` — search for `compute_policy_loss_opd` (or
   similar). Read the loss formula.
2. `verl/workers/actor/dp_actor.py` — read `update_actor()` method structure
3. `verl/workers/fsdp_workers.py` — search for `ActorWorker` and read how it
   wraps `dp_actor.update_actor()` as a Ray-remote call

### Key questions
- [ ] What does FSDP stand for? Why is it needed for large model training?
- [ ] In the toy code, `logits` is a single vector. In the real code, what is
   the equivalent? (Hint: it's a full transformer)
- [ ] What does `.detach()` do on `advantages`? Why is it called?

### Checkpoint 6 — Policy Loss Function
In `playground/toy_policy_loss.py`, implement the OPD policy loss from scratch:

```python
def opd_policy_loss(
    current_log_probs: torch.Tensor,  # [batch, seq_len]
    advantages: torch.Tensor,         # [batch, seq_len]
    response_mask: torch.Tensor,      # [batch, seq_len]
) -> torch.Tensor:
    """
    YOUR CODE HERE.
    Return a scalar loss tensor (mean over valid token positions).
    advantages should be treated as a constant (no gradient through it).
    """
    ...
```

Then hook this into the toy `ActorWorker.update_actor()` from Stage 1.

---

## Stage 7 — Tie It All Together (Mini Trainer Loop)

**Why:** Now you build the full toy loop using your Stage 2–6 components.

### Checkpoint 7 — Toy Trainer
Create `playground/toy_trainer.py`. Wire together:

1. `DataProto` from Stage 1
2. Ray-based `RolloutActor` from Stage 3
3. A toy teacher (returns fixed log-probs)
4. `compute_opd_advantage` from Stage 5
5. `opd_policy_loss` from Stage 6

Run 3 training steps. Print per-step loss and a simulated "validation score".

**TA hint:** You don't need a real model. A `torch.nn.Linear(8, 8)` with a tiny
vocab is enough to make `backward()` work and see the loss go down.

---

## Stage 8 — Read The Real Production Entrypoint

**Why:** Now that you've built toy versions, the real code should make sense.

### Read (in this order)
1. `cross_distill_smoke_1gpu.sh` — what env vars are set? What does it launch?
2. `verl/trainer/main_ppo.py` — find `main()`. What config system does it use?
3. `verl/trainer/ppo/ray_trainer.py` — find `RayPPOTrainer.fit()`. Trace the
   same loop you built in Stage 7.
4. `verl/workers/reward_manager/opd.py` — the real OPD reward manager
5. `recipe/gkd/teacher/worker.py` — the real teacher service

### Key questions
- [ ] What config library does verl use? (Hint: look at `@hydra.main`)
- [ ] How many Ray actors are created in a real run?
- [ ] Where does vLLM run — inside the actor worker or separately?

### No code checkpoint — take notes instead
Write your answers as comments in `toy_trainer.py` with a `# PROD:` prefix,
pointing to where the equivalent happens in the real code.

---

## Progress Tracker

| Stage | Topic | Status |
|-------|-------|--------|
| 0 | Run demo, trace data flow | [ ] |
| 1 | DataProto — toy_dataproto.py | [ ] |
| 2 | Ray basics — toy_ray_pipeline.py | [ ] |
| 3 | WorkerGroup — extend toy_ray_pipeline.py | [ ] |
| 4 | Rollout with temperature | [ ] |
| 5 | OPD advantage — toy_advantage.py | [ ] |
| 6 | Policy loss — toy_policy_loss.py | [ ] |
| 7 | Full loop — toy_trainer.py | [ ] |
| 8 | Read real code, annotate | [ ] |

---

## Quick Reference: Why Each Technology

| Technology | Why it's here | Toy equivalent |
|------------|--------------|----------------|
| **Ray** | Run workers on different GPUs/machines; async task scheduling | `@ray.remote` class, `ray.get()` |
| **verl** | Opinionated framework wrapping Ray for RL training loops | `WorkerGroup`, `DataProto` |
| **vLLM** | High-throughput LLM inference with paged attention | `RolloutWorker.generate_sequences()` |
| **FSDP** | Shard model weights across GPUs so one huge model fits | `ActorWorker` with `torch.nn.Parameter` |
| **Hydra** | Hierarchical config (YAML + CLI overrides) for big experiments | plain Python dict |
| **DataProto** | Typed envelope passed between Ray workers | your `toy_dataproto.py` |

---

## TA Rules

1. Type every checkpoint yourself. Reading != understanding; typing forces you
   to engage with each line.
2. If something doesn't make sense, read the real source first before asking.
   Note your question and the file/line that confused you.
3. You can ask for a hint at any stage. A hint means: a nudge, a question to
   think about, or a partial signature — never the full solution.
4. Mark stages done in the Progress Tracker as you finish them.
