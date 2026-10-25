from pathlib import Path

import torch

from examples.data_preprocess.gsm8k_r1_zero import R1_ZERO_LLAMA_TEMPLATE, R1_ZERO_TEMPLATE, build_prompt
from verl.trainer.ppo.core_algos import compute_policy_loss_opd
from verl.workers.reward_manager.opd import TeacherClient


# Historical non-prefilled student prompt used only to verify migration of old
# checkpoints. New SFT/RL data uses the canonical prefilled template above.
_LEGACY_UNPREFIXED_TEMPLATE = (
    "A conversation between User and Assistant. The User asks a math question, and the Assistant solves it. "
    "The Assistant must write its reasoning between <think> and </think>, then write only the final answer "
    "between <answer> and </answer>. The complete response must use this form: "
    "<think> reasoning process here </think> <answer> final answer here </answer>.\n"
    "User: {question}\n"
    "Assistant:\n"
)


class _TokenizerStub:
    def __init__(self, bos_token: str, eos_token: str):
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.additional_special_tokens = []


class _PieceTokenizerStub:
    """Tiny tokenizer whose teacher side fuses the space before ``<think>``."""

    def __init__(self, pieces: dict[int, str], prompt_ids: list[int] | None = None):
        self.pieces = pieces
        self.prompt_ids = prompt_ids

    def __call__(self, text: str, add_special_tokens: bool = False):
        assert not add_special_tokens
        assert self.prompt_ids is not None
        return {"input_ids": self.prompt_ids}

    def decode(self, ids, skip_special_tokens: bool = False):
        return "".join(self.pieces[token_id] for token_id in ids)


class _ContextSensitiveTokenizerStub(_PieceTokenizerStub):
    """Makes a middle slice decode differently from the same full prefix."""

    def __init__(self, pieces: dict[int, str], middle_prefix: int, unstable_id: int | None = None):
        super().__init__(pieces)
        self.middle_prefix = middle_prefix
        self.unstable_id = unstable_id

    def decode(self, ids, skip_special_tokens: bool = False):
        text = super().decode(ids, skip_special_tokens=skip_special_tokens)
        if ids and ids[0] == self.middle_prefix:
            return " " + text
        if self.unstable_id is not None and ids and ids[-1] == self.unstable_id:
            return text[:-1] + "\ufffd"
        return text


class _PolicyLossConfig(dict):
    clip_ratio = 0.2
    clip_ratio_low = 0.2
    clip_ratio_high = 0.28
    global_batch_info = {}

    def __init__(self):
        super().__init__(
            clip_ratio_c=10.0,
            policy_loss={"opd_loss_max_clamp": 2.0},
        )


def _client() -> TeacherClient:
    client = TeacherClient.__new__(TeacherClient)
    client.prompt_bridge_mode = "r1_zero_llama_to_r1_zero"
    client.student_tokenizer = _TokenizerStub("<s>", "</s>")
    client.tokenizer = _TokenizerStub("<teacher-bos>", "<teacher-eos>")
    client._bridge_teacher_system_prompt = "{instruction}"
    client._bridge_teacher_prompt = (
        "A conversation between User and Assistant. The User asks a question, and the Assistant solves it. "
        "The Assistant first thinks about the reasoning process in the mind and then provides the User with "
        "the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within "
        "<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> "
        "<answer> answer here </answer>.\nUser: {question}\nAssistant: <think>"
    )
    return client


def test_r1_zero_llama_dataset_prompt_is_raw_lllm_template():
    question = "What is 1 + 1?"
    prompt = build_prompt(question, "r1_zero_llama")
    assert prompt == [{"role": "user", "content": R1_ZERO_LLAMA_TEMPLATE.format(question=question)}]
    assert prompt[0]["content"].endswith("Assistant: <think>")


def test_r1_zero_and_legacy_llama_name_render_the_same_prompt():
    question = "What is 2 + 3?"

    assert R1_ZERO_TEMPLATE == R1_ZERO_LLAMA_TEMPLATE
    assert build_prompt(question, "r1_zero") == build_prompt(question, "r1_zero_llama")


def test_raw_r1_zero_chat_template_does_not_prepend_bos():
    script = (Path(__file__).parents[1] / "cross_distill_smoke_1gpu.sh").read_text(encoding="utf-8")

    assert 'r1_zero_chat_template="\\"{{ messages[0][\'content\'] }}\\""' in script
    assert "{{ bos_token }}{{ messages[0]['content'] }}" not in script
    assert "cannot be combined with OPD_PROMPT_BRIDGE_MODE" in script


def test_raw_mode_does_not_invent_teacher_bos_and_translates_eos():
    client = _client()

    teacher_text = client._swap_bos_eos_for_teacher(
        "A conversation between User and Assistant.</answer></s>"
    )

    assert teacher_text == "A conversation between User and Assistant.</answer><teacher-eos>"
    assert not teacher_text.startswith("<teacher-bos>")


def test_raw_mode_replaces_bos_only_when_student_sequence_has_one():
    client = _client()

    teacher_text = client._swap_bos_eos_for_teacher("<s>prompt</s>")

    assert teacher_text == "<teacher-bos>prompt<teacher-eos>"


def test_teacher_logprobs_are_selected_by_actual_token_id_not_column_zero():
    teacher_ids = [10, 20, 30, 40]
    topk_logps = torch.tensor(
        [
            [-0.1, -2.0],
            [-3.0, -0.2],
            [-0.3, -4.0],
        ]
    )
    topk_indices = torch.tensor(
        [
            [99, 20],
            [30, 98],
            [97, 40],
        ],
        dtype=torch.int32,
    )

    selected, columns = TeacherClient._select_actual_teacher_logps(
        teacher_ids, topk_logps, topk_indices
    )

    assert torch.equal(selected, torch.tensor([-2.0, -3.0, -4.0]))
    assert torch.equal(columns, torch.tensor([1, 0, 1]))


def test_teacher_logprob_selection_fails_when_actual_token_is_missing():
    try:
        TeacherClient._select_actual_teacher_logps(
            [10, 20],
            torch.tensor([[-0.1, -0.2]]),
            torch.tensor([[98, 99]], dtype=torch.int32),
        )
    except RuntimeError as exc:
        assert "actual sequence token" in str(exc)
        assert "target_id" in str(exc)
    else:
        raise AssertionError("missing actual teacher token must fail closed")


def test_bridge_keeps_one_think_and_translates_terminal_token():
    client = _client()
    student_text = (
        "<s>"
        + _LEGACY_UNPREFIXED_TEMPLATE.format(question="What is 1 + 1?")
        + "<think>1 + 1 = 2.</think> <answer>2</answer></s>"
    )

    teacher_text = client._bridge_student_text_to_teacher(student_text)

    assert teacher_text.startswith("<teacher-bos>A conversation between User and Assistant.")
    assert "\nUser: What is 1 + 1?\nAssistant: <think>1 + 1 = 2." in teacher_text
    assert "<think><think>" not in teacher_text
    assert teacher_text.endswith("</answer><teacher-eos>")


def test_prefilled_think_is_inside_teacher_alignment_response():
    client = _client()
    student_prompt = "<s>" + _LEGACY_UNPREFIXED_TEMPLATE.format(question="What is 1 + 1?")
    teacher_prompt = client._build_bridge_teacher_prompt(student_prompt)

    response_start_text = client._bridge_teacher_response_start_text(teacher_prompt)

    assert teacher_prompt == response_start_text + "<think>"
    assert response_start_text.endswith("Assistant: ")


def test_unformatted_student_response_aligns_after_teacher_prefill():
    client = _client()
    student_prompt = "<s>" + _LEGACY_UNPREFIXED_TEMPLATE.format(question="What is 1 + 1?")
    teacher_prompt = client._build_bridge_teacher_prompt(student_prompt)

    response_start_text = client._bridge_teacher_response_start_text(
        teacher_prompt, "We calculate 1 + 1 = 2."
    )

    assert response_start_text == teacher_prompt


def test_fused_space_think_boundary_is_found_and_aligns():
    client = TeacherClient.__new__(TeacherClient)
    client.prompt_bridge_mode = "r1_zero_llama_to_r1_zero"
    client.tokenizer = _PieceTokenizerStub(
        {10: "prefix", 11: " <", 12: "think", 13: ">", 14: "work"},
        prompt_ids=[10, 11, 12, 13],
    )
    student_tokenizer = _PieceTokenizerStub({1: "<", 2: "think", 3: ">", 4: "work"})

    response_start, ignore_space = client._bridge_teacher_response_start_tok(
        "prefix <think>", "<think>work"
    )

    assert response_start == 1
    assert ignore_space is True

    _, _, _, truncated, stats = TeacherClient._align_chunks(
        [1, 2, 3, 4],
        [11, 12, 13, 14],
        torch.tensor([-1.0, -2.0, -3.0, -4.0]),
        student_tokenizer,
        client.tokenizer,
        ignore_one_leading_teacher_space=ignore_space,
    )

    assert truncated is False
    assert stats["matched_student_tokens"] == 4
    assert stats["trainable_student_tokens"] == 4


def test_alignment_uses_prefixes_not_context_sensitive_middle_slices():
    student_tokenizer = _PieceTokenizerStub({1: "a", 2: "b", 3: "c", 4: "d"})
    teacher_tokenizer = _ContextSensitiveTokenizerStub(
        {10: "a", 20: "b", 30: "c", 40: "d"},
        middle_prefix=20,
    )

    aligned, chunk_ids, details, truncated, stats = TeacherClient._align_chunks(
        [1, 2, 3, 4],
        [10, 20, 30, 40],
        torch.tensor([-1.0, -2.0, -3.0, -4.0]),
        student_tokenizer,
        teacher_tokenizer,
        debug=True,
    )

    assert truncated is False
    assert torch.isfinite(aligned).all()
    assert (chunk_ids >= 0).all()
    assert stats["matched_student_tokens"] == 4
    assert stats["trainable_student_tokens"] == 4
    assert stats["fallback_student_tokens"] == 0
    assert all(detail["type"] == "matched" for detail in details)


def test_alignment_resynchronizes_after_unstable_prefix_boundary():
    student_tokenizer = _PieceTokenizerStub({1: "a", 2: "b", 3: "c", 4: "d"})
    teacher_tokenizer = _ContextSensitiveTokenizerStub(
        {10: "a", 20: "b", 30: "c", 40: "d"},
        middle_prefix=-1,
        unstable_id=20,
    )

    aligned, chunk_ids, _, truncated, stats = TeacherClient._align_chunks(
        [1, 2, 3, 4],
        [10, 20, 30, 40],
        torch.tensor([-1.0, -2.0, -3.0, -4.0]),
        student_tokenizer,
        teacher_tokenizer,
    )

    assert truncated is False
    assert torch.isfinite(aligned).all()
    assert (chunk_ids >= 0).all()
    assert stats["matched_student_tokens"] == 4
    assert stats["trainable_student_tokens"] == 4
    assert stats["fallback_student_tokens"] == 0
    # The missing boundary at "ab" is absorbed into the next synchronized
    # chunk instead of making tokens c/d an untrainable fallback tail.
    assert stats["matched_chunks"] == 3


def test_raw_boundary_recovery_keeps_prompt_response_fused_token():
    teacher_tokenizer = _PieceTokenizerStub(
        {
            100: "prompt<",
            10: "think>The",
            20: " three",
            30: " friends",
        }
    )
    teacher_ids = [100, 10, 20, 30]

    start, leading_text = TeacherClient._recover_teacher_response_boundary(
        teacher_ids,
        estimated_start=2,
        response_end=4,
        student_response_text="The three friends",
        teacher_tokenizer=teacher_tokenizer,
    )

    assert start == 1
    assert leading_text == "think>"

    student_tokenizer = _PieceTokenizerStub(
        {1: "The", 2: " three", 3: " friends"}
    )
    aligned, chunk_ids, _, truncated, stats = TeacherClient._align_chunks(
        [1, 2, 3],
        teacher_ids[start:4],
        torch.tensor([-1.0, -2.0, -3.0]),
        student_tokenizer,
        teacher_tokenizer,
        teacher_leading_text_to_strip=leading_text,
    )

    assert truncated is False
    assert torch.isfinite(aligned).all()
    assert (chunk_ids >= 0).all()
    assert stats["matched_student_tokens"] == 3
    assert stats["fallback_student_tokens"] == 0


def test_eos_chunk_never_receives_negative_opd_advantage():
    old_log_prob = torch.tensor([[-0.1]])
    current_log_prob = old_log_prob.clone()
    # First half is the teacher EOS logprob; second half is the reserved EOS
    # chunk ID. Without the terminal guard this produces a negative advantage.
    teacher_and_chunk_ids = torch.tensor([[-2.0, -2.0]])

    loss, metrics = compute_policy_loss_opd(
        old_log_prob=old_log_prob,
        log_prob=current_log_prob,
        advantages=teacher_and_chunk_ids,
        response_mask=torch.ones_like(old_log_prob),
        config=_PolicyLossConfig(),
    )

    assert loss.item() == 0.0
    assert metrics["actor/opd_eos_tokens"] == 1
    assert metrics["actor/opd_eos_negative_suppressed"] == 1
    assert metrics["actor/opd_eos_negative_suppressed_ratio"] == 1.0
