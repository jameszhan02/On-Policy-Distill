"""Process-wide compatibility patches for local smoke tests.

Python imports this module automatically when the repo root is on PYTHONPATH.
Keep this file small: it is meant for compatibility shims that must also apply
inside spawned worker processes.
"""

try:
    from transformers.models.qwen2.tokenization_qwen2 import Qwen2Tokenizer

    if not hasattr(Qwen2Tokenizer, "all_special_tokens_extended"):
        Qwen2Tokenizer.all_special_tokens_extended = property(lambda self: self.all_special_tokens)
except Exception:
    pass
