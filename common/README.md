# Shared classifier code

See [Prompt structure — v1](PROMPT_STRUCTURE_V1.md) for the exact formatting
contract, assembly order, and output-label mappings.

Plain Python modules shared by server implementations. Import from `common`
with the repository root on the Python path; no package installation or build
step is needed. The runtime dependencies are Pydantic 2 and NumPy.

- `request_schema.py`: `ClassifierRequest` and question/option validation.
- `prompt_builder.py`: request arguments → model-independent prompt strings.
- `response_scoring.py`: label/token logits → scored answers and JSON-compatible responses.

```python
from common import prepare_prompt, build_response

plan = prepare_prompt(
    {
        "model": "my-model",
        "state": "The bicycle is red.",
        "questions": {
            "color": {
                "type": "choice",
                "instructions": "What color is the bicycle?",
                "criteria": {"red": None, "blue": None},
            }
        },
    }
)

logits = {plan.questions[0].branch_id: {"A": 3.0, "B": 1.0}}
response = build_response(plan, logits, input_tokens=123)
```

The plan exposes `system_prompt_prefix` (cacheable text), `prefix_instruction`
(before context), `suffix_instruction` (after context), and `questions` (ordered
scoring branches). Each branch includes its instruction, answer prefix, output
labels, question ID, and unique plan-local branch ID. There is exactly one branch
per question. `answer_labels` maps each output label to its public answer value.

Call `prepare_prompt(request, version="v1")` to select the template; omitting
`version` uses the shared
`DEFAULT_TEMPLATE_VERSION`. A version fixes the complete prompt format and output-label mapping, independent
of the inference engine. There are no configurable scoring modes or score formats.
Unsupported version arguments raise `ValueError`. Version selection is not a
request-schema field. Future formats should get a new version
and formatter while keeping existing versions unchanged; all adapters must pass
through the requested version. The resolved version is on the plan and in advanced
response metadata. Model-specific chat templates remain separate, so identical
versions guarantee shared classifier text, not identical model token sequences.

Chat rendering stays in the server. To reproduce the HF default format:

1. Combine `system_prompt_prefix + prefix_instruction` into the system text.
2. For state, add a user message containing `"State:\n" + canonical(state) +
   "\n\n" + suffix_instruction + question.instruction`. `canonical` is in
   `common.prompt_builder`.
3. For messages, prepend the system text to an existing leading system turn
   (with a newline), or insert a system turn. Preserve history, then append a
   user turn containing `suffix_instruction + question.instruction`.
4. Apply the model's chat template with a generation prompt and thinking disabled,
   append `question.answer_prefix`, and tokenize. Verify output labels are single
   tokens at that boundary; map their logits back to the exact labels.

The version defines these scoring rules:

| Question type | Labels | Result |
| --- | --- | --- |
| `choice` | Letters assigned to candidates | Highest-probability candidate |
| `score` | Rubric indices (letters internally above 10 levels) | Expected zero-based rubric index |
| `noul` | Digits 1–9 | Expected rating mapped to [0.01, 0.99] |

`options` only contains `raw_logits`; the former `choice_mode`, `score_mode`, and
`score_format` fields are rejected. Changing formatting/scoring rules in the
future requires a new template version rather than per-request switches.

`build_response` takes logits keyed by branch ID, then output label. Token usage
comes from the server; output usage defaults to zero. `build_answers` returns
only the answers. `advanced=True` includes diagnostics; raw logits also require
`options.raw_logits=True`. Scores and confidence are not calibrated.

## Response scoring and PyTorch inputs

[response_scoring.py](response_scoring.py) validates and aligns logits, applies the
fixed scoring rules, filters diagnostics, and builds the response in one module.
`build_answers` returns question-ID-keyed answers; `build_response` additionally
includes the model name, caller-supplied token usage, and optional metadata.

Both accept a mapping from branch ID to one of:

- Output-label-to-float maps, also accepting scalar floating PyTorch tensors.
- A one-dimensional floating tensor in the branch's `output_labels` order.
- A one-dimensional vocabulary tensor with an explicit `token_ids` mapping.

Example with vocabulary logits (token IDs are illustrative):

```python
import torch
from common import build_response

vocab_logits = torch.zeros(100)
vocab_logits[42], vocab_logits[73] = 3.0, 1.0
response = build_response(
    plan,
    {"0": vocab_logits},
    token_ids={"0": {"A": 42, "B": 73}},
    input_tokens=123,
)
```

Use actual token IDs verified against the rendered answer boundary. Supply the
logit vector for the selected scoring position, not an entire batch/sequence.
Per-branch token mappings allow different question types and output labels.
Missing, duplicate, out-of-range, or unexpected token mappings are rejected.

PyTorch is optional for float-map users. Tensors can be on CPU or an accelerator,
including float16/bfloat16 and tensors with gradients. The scorer detaches them,
selects the allowed labels on their device, and transfers only selected logits to
CPU float32. It returns ordinary JSON-compatible data; gradients are not retained.

Default answers omit diagnostics. `advanced=True` retains them and adds metadata
in the full response. Raw logits additionally require `options.raw_logits=True`.
These settings do not change the prompt or numerical answer. Every selected logit
must be finite; unused vocabulary entries are not scored.

Unknown top-level request fields are ignored. Questions and options remain
strict. The HF server imports these modules for validation, prompt formatting, and
response scoring; Noul uses integer labels 1–9 with its existing mapping to [0.01, 0.99].
All question types now share one constant system prefix, including Noul-only
requests. Cache reuse must compare rendered token prefixes.

Run from the repository root with pytest installed:

```sh
python -m pytest common/tests -q
```

Tests cover HF adapter assembly from the shared prompt plan when the server is
present, plus fixed Noul mapping and removed-option validation.
