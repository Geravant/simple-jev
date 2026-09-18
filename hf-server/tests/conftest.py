"""Shared lightweight test doubles for HF service and prompt tests.

FakeCompiler uses the real common plan but a synthetic two-token shared prefix,
so usage expectations are exact and independent of any downloaded tokenizer.
Tokenizer maps UTF-8 bytes to IDs and exposes an inspectable role-based template.
It exercises adapter control flow, not production tokenizer/model compatibility.
"""

from hf_server import Branch, CompiledRequest

from common import prepare_prompt


class FakeCompiler:
    """Small deterministic token tree for service/usage tests without inference."""

    def compile(self, request):
        """Give each real plan question a synthetic [1, 2, unique_leaf] path."""
        plan = prepare_prompt(request)
        return CompiledRequest(
            plan,
            [
                Branch(
                    q.branch_id,
                    [1, 2, i + 3],
                    list(range(len(q.output_labels))),
                    [],
                    q.answer_prefix,
                )
                for i, q in enumerate(plan.questions)
            ],
        )


class Tokenizer:
    """Make labels single-byte tokens and expose exactly which roles are rendered."""

    def encode(self, text, **kwargs):
        """Return deterministic byte IDs; keyword arguments mimic the HF API."""
        return list(text.encode("utf8"))

    def apply_chat_template(self, messages, **kwargs):
        """Require the intended generation settings and open an assistant turn."""
        assert kwargs == {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": False,
        }
        return (
            "\n".join(f"{m['role']}: {m['content']}" for m in messages)
            + "\nassistant: "
        )
