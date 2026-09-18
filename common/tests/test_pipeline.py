import json
import sys
from pathlib import Path

import pytest

from common import build_answers, build_response, prepare_prompt
from common.prompt_builder import canonical


def payload():
    return {
        "model": "test",
        "state": {"color": "red"},
        "questions": {
            "color": {
                "type": "choice",
                "instructions": "Color?",
                "criteria": {"red": None, "blue": "Blue"},
            },
            "support": {
                "type": "score",
                "instructions": {"ask": "Support?"},
                "criteria": ["no", "maybe", "yes", "certain"],
            },
            "truth": {"type": "noul", "instructions": "Red?"},
        },
        "options": {
            "raw_logits": True,
        },
    }


def logits_for(plan):
    return {
        q.branch_id: {label: float(i) for i, label in enumerate(q.output_labels)}
        for q in plan.questions
    }


def test_context_independence_and_request_snapshot():
    body = payload()
    plan = prepare_prompt(body)
    body["state"] = "different context"
    other = prepare_prompt(body)
    assert plan == other
    body["questions"]["color"]["criteria"]["red"] = "changed"
    assert "changed" not in plan.questions[0].instruction
    assert json.loads(
        json.dumps(build_response(plan, logits_for(plan), input_tokens=20))
    )["usage"] == {"input_tokens": 20, "output_tokens": 0}


def test_mapping_validation_and_order():
    plan = prepare_prompt(payload())
    logits = logits_for(plan)
    expected = build_answers(plan, logits)
    assert (
        build_answers(
            plan,
            {
                k: dict(reversed(list(v.items())))
                for k, v in reversed(list(logits.items()))
            },
        )
        == expected
    )
    assert expected["color"]["choice"] == "blue"
    assert "logits" not in expected["color"]
    assert "logits" in build_answers(plan, logits, advanced=True)["color"]
    with pytest.raises(ValueError, match="branch IDs"):
        build_answers(plan, {})
    logits["0"]["extra"] = 0
    with pytest.raises(ValueError, match="output labels"):
        build_answers(plan, logits)
    logits = logits_for(plan)
    logits["0"]["A"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        build_answers(plan, logits)
    with pytest.raises(ValueError, match="nonnegative integer"):
        build_response(plan, logits_for(plan), input_tokens=-1)


@pytest.mark.parametrize("levels", [2, 10, 11, 50])
def test_score_label_mapping(levels):
    body = payload()
    body["questions"] = {
        "score": {
            "type": "score",
            "instructions": "Level?",
            "criteria": [str(i) for i in range(levels)],
        }
    }
    plan = prepare_prompt(body)
    question = plan.questions[0]
    logits = {
        question.branch_id: {
            label: (0 if i == levels - 1 else -1000)
            for i, label in enumerate(question.output_labels)
        }
    }
    answer = build_answers(plan, logits)["score"]
    assert answer["score"] == levels - 1
    assert answer["confidence"] == 1
    assert list(answer["probabilities"]) == [str(i) for i in range(levels)]


@pytest.mark.parametrize("context", ["state", "chat", "system_chat"])
def test_hf_adapter_uses_common_plan(context):
    # Optional integration test: the HF adapter is currently Git-ignored.
    hf_path = Path(__file__).resolve().parents[2] / "hf-server"
    if not (hf_path / "hf_server.py").is_file():
        pytest.skip("Local HF adapter is unavailable")
    sys.path.insert(0, str(hf_path))
    from hf_server import PromptCompiler

    from common import ClassifierRequest

    class Tokenizer:
        def encode(self, text, **kwargs):
            return list(text.encode("utf8"))

    body = payload()
    if context != "state":
        body.pop("state")
        body["messages"] = [{"role": "user", "content": "Red bicycle"}]
        if context == "system_chat":
            body["messages"].insert(0, {"role": "system", "content": "Original system"})
    plan = prepare_prompt(body)
    request = ClassifierRequest.model_validate(body)
    reference = PromptCompiler(Tokenizer()).compile(request, render_only=True)
    system = plan.system_prompt_prefix + plan.prefix_instruction
    assert len(plan.questions) == len(reference.branches)
    for question, branch in zip(plan.questions, reference.branches):
        content = plan.suffix_instruction + question.instruction
        if context == "state":
            messages = [
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": f"State:\n{canonical(body['state'])}\n\n" + content,
                },
            ]
        else:
            messages = [dict(m) for m in body["messages"]]
            if messages[0]["role"] == "system":
                messages[0]["content"] = system + "\n" + messages[0]["content"]
            else:
                messages.insert(0, {"role": "system", "content": system})
            messages.append({"role": "user", "content": content})
        assert messages == branch.messages
        assert question.answer_prefix == branch.answer_prefix
        assert question.branch_id == branch.branch_id
    assert reference.plan == plan


def test_unknown_request_fields_do_not_change_prompts_or_answers():
    from common import ClassifierRequest

    body = payload()
    expected = prepare_prompt(body)
    body.update(temperature=0.7, max_tokens=100, custom_client_setting={"x": True})
    request = ClassifierRequest.model_validate(body)
    assert (
        not {"temperature", "max_tokens", "custom_client_setting"}
        & request.model_dump().keys()
    )
    actual = prepare_prompt(request)
    assert actual == expected
    assert build_answers(actual, logits_for(actual)) == build_answers(
        expected, logits_for(expected)
    )


@pytest.mark.parametrize("location", ["model", "options", "question"])
def test_known_fields_and_nested_objects_still_validate(location):
    from pydantic import ValidationError

    from common import ClassifierRequest

    body = payload()
    if location == "model":
        body["model"] = ""
    elif location == "options":
        body["options"]["choice_mod"] = "direct"
    else:
        body["questions"]["color"]["unexpected"] = True
    with pytest.raises(ValidationError):
        ClassifierRequest.model_validate(body)


def test_template_version_is_explicit():
    from common import DEFAULT_TEMPLATE_VERSION, ClassifierRequest

    body = payload()
    default = prepare_prompt(body)
    pinned = prepare_prompt(body, version="v1")
    assert pinned == default
    assert prepare_prompt(ClassifierRequest.model_validate(body), "v1") == pinned
    assert pinned.template_version == DEFAULT_TEMPLATE_VERSION == "v1"
    response = build_response(pinned, logits_for(pinned), input_tokens=1, advanced=True)
    assert response["metadata"]["template_version"] == "v1"
    with pytest.raises(ValueError, match="Unsupported template version"):
        prepare_prompt(body, version="unknown-version")
    # Like other unknown request fields, this cannot override builder selection.
    body["template_version"] = "unknown-version"
    assert prepare_prompt(body) == default
    assert "template_version" not in ClassifierRequest.model_fields


@pytest.mark.parametrize(
    "field,value",
    [
        ("choice_mode", "independent"),
        ("score_mode", "rating"),
        ("score_format", "decimal"),
    ],
)
def test_removed_options_are_rejected(field, value):
    from pydantic import ValidationError

    body = payload()
    body["options"][field] = value
    with pytest.raises(ValidationError, match=field):
        prepare_prompt(body)


@pytest.mark.parametrize("bin_index,expected", [(0, 0.01), (4, 0.5), (8, 0.99)])
def test_noul_fixed_mapping_and_shared_system_prefix(bin_index, expected):
    body = payload()
    mixed = prepare_prompt(body)
    body["questions"] = {"truth": body["questions"]["truth"]}
    plan = prepare_prompt(body)
    assert plan.system_prompt_prefix == mixed.system_prompt_prefix
    assert len(mixed.questions) == len(mixed._request.questions)
    question = plan.questions[0]
    logits = {
        question.branch_id: {
            label: (0 if i == bin_index else -1000)
            for i, label in enumerate(question.output_labels)
        }
    }
    assert build_answers(plan, logits)["truth"]["noul"] == pytest.approx(expected)


@pytest.mark.parametrize("dtype_name", ["float32", "float16", "bfloat16"])
def test_tensor_logits_match_label_maps(dtype_name):
    torch = pytest.importorskip("torch")
    plan = prepare_prompt(payload())
    floats = logits_for(plan)
    expected = build_response(plan, floats, input_tokens=12, advanced=True)
    dtype = getattr(torch, dtype_name)
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    if torch.backends.mps.is_available():
        devices.append("mps")
    for device in devices:
        compact, scalar_maps, vocabulary, token_ids = {}, {}, {}, {}
        for q in plan.questions:
            row = torch.tensor(
                list(floats[q.branch_id].values()),
                dtype=dtype,
                device=device,
                requires_grad=True,
            )
            compact[q.branch_id] = row
            scalar_maps[q.branch_id] = dict(zip(q.output_labels, row.unbind()))
            ids = list(range(30, 30 - len(q.output_labels), -1))
            vocab = torch.full((64,), float("nan"), dtype=dtype, device=device)
            vocab[ids] = row.detach()
            vocabulary[q.branch_id] = vocab.requires_grad_()
            token_ids[q.branch_id] = dict(zip(q.output_labels, ids))
        for values, mapping in [
            (compact, None),
            (scalar_maps, None),
            (vocabulary, token_ids),
        ]:
            actual = build_response(
                plan, values, token_ids=mapping, input_tokens=12, advanced=True
            )
            assert actual == expected
            json.dumps(actual)
        assert all(t.grad is None for t in compact.values())


def test_tensor_mapping_validation():
    torch = pytest.importorskip("torch")
    body = payload()
    body["questions"] = {"color": body["questions"]["color"]}
    plan = prepare_prompt(body)
    for tensor, mapping, message in [
        (torch.zeros(1, 2), None, "one-dimensional"),
        (torch.zeros(20), None, "supply token_ids"),
        (torch.zeros(20), {"0": {"A": 1}}, "exactly the output labels"),
        (torch.zeros(20), {"0": {"A": 1, "B": 1}}, "distinct"),
        (torch.zeros(20), {"0": {"A": -1, "B": 2}}, "within"),
        (torch.zeros(20), {"0": {"A": 1, "B": 20}}, "within"),
        (torch.zeros(20), {"0": {"A": True, "B": 2}}, "within"),
        (torch.zeros(20), {}, "branch IDs"),
        (torch.tensor([float("nan"), 1.0]), None, "finite"),
        (torch.tensor([1, 2]), None, "floating-point"),
    ]:
        with pytest.raises(ValueError, match=message):
            build_answers(plan, {"0": tensor}, token_ids=mapping)
    with pytest.raises(ValueError, match="scalars"):
        build_answers(plan, {"0": {"A": torch.zeros(1), "B": 1.0}})
