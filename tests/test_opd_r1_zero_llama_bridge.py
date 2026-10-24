import torch

from examples.data_preprocess.gsm8k_r1_zero import R1_ZERO_LLAMA_TEMPLATE, R1_ZERO_TEMPLATE, build_prompt
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
