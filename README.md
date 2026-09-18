![Simple Jev Mascot and Logo](./imgs/Simple-Jev-Logo.png)

# Simple Jev Project

Use compatible open language models for structured classification and scoring, without training a separate classifier head.

Send shared context and a set of questions. Simple Jev reads the model's next-token logits for each question and builds a JSON response containing choices, rubric scores, or truth/support judgments. The model does not generate a JSON completion: the server constructs the response from the scores.

The current implementation runs locally with Hugging Face Transformers and PyTorch. Shared request validation, versioned prompt instructions, and response scoring live in the plain Python `common/` folder so other inference implementations can use the same rules.

## Running the HF Server

Use Python 3.12 or newer.

```bash
# Clone the repository and create an environment.
git clone https://github.com/featherless-ai/simple-jev.git
cd simple-jev
python3.13 -m venv .venv
source .venv/bin/activate

# Install the server, including the shared common modules.
python -m pip install -e './hf-server'

# Start with a small model on CPU.
python hf-server/hf_server.py \
  --model Qwen/Qwen3.5-0.8B \
  --device cpu --dtype float32 \
  --max-model-len 4096 \
  --max-batch-size 4 --max-batch-tokens 4096
```

The first run downloads the model unless it is already cached. A local model directory can also be passed to `--model`. For CUDA or ROCm, install the appropriate PyTorch build for your hardware before installing the server.

The server listens on `http://127.0.0.1:8000`. Once the model is loaded:

```bash
curl http://127.0.0.1:8000/health
```

Open `http://127.0.0.1:8000/docs` for the interactive API documentation. After installation, `simple-jev` and `python -m hf_server` accept the same arguments as the script.

## How do I use the API?

Send a non-streaming `POST /v1/classifier` request. The `model` value must exactly match the ID or path used to start the server. Supply exactly one of:

- `state`: a string, JSON object, or JSON array containing the shared context.
- `messages`: text chat history, rendered using the model's own chat template.

The API takes inspiration from TypeSafe's structured-decision interface and includes project-specific behavior. `/v1/systemone` is an alias of `/v1/classifier`; both run the same implementation. Use this repository's [API reference](hf-server/API_REFERENCE.md) as the contract for clients.

```bash
curl http://127.0.0.1:8000/v1/classifier \
  -H 'Content-Type: application/json' \
  --data-binary @- <<'JSON'
{
  "model": "Qwen/Qwen3.5-0.8B",
  "state": "Mia owns a red bicycle. Her dog is named Max.",
  "questions": {
    "color": {
      "type": "choice",
      "instructions": "What color is Mia's bicycle?",
      "criteria": {"red": null, "blue": null}
    },
    "support": {
      "type": "score",
      "instructions": "How well does the context support that Mia owns a bicycle?",
      "criteria": ["Unsupported", "Partially supported", "Fully supported"]
    },
    "dog": {
      "type": "noul",
      "instructions": "Is Mia's dog named Max?"
    }
  }
}
JSON
```

Question IDs become keys in `answers`. The following response illustrates the shape; the numbers are examples, not promised model outputs:

```json
{
  "model": "Qwen/Qwen3.5-0.8B",
  "answers": {
    "color": {
      "type": "choice",
      "choice": "red",
      "confidence": 0.95,
      "probabilities": {"red": 0.95, "blue": 0.05}
    },
    "support": {
      "type": "score",
      "score": 1.75,
      "confidence": 0.8,
      "probabilities": {"0": 0.05, "1": 0.15, "2": 0.8},
      "legend": {"0": "Unsupported", "1": "Partially supported", "2": "Fully supported"}
    },
    "dog": {"type": "noul", "noul": 0.9}
  },
  "usage": {"input_tokens": 600, "output_tokens": 0}
}
```

| Question type | Input criteria | Result |
| --- | --- | --- |
| `choice` | Object with 2–50 candidate IDs and optional descriptions | Highest-probability candidate, its confidence, and the candidate distribution. |
| `score` | Array of 2–50 rubric levels, lowest to highest | Expected zero-based rubric index, confidence, distribution, and rubric legend. A three-level rubric returns a value from 0 to 2, including fractional values. |
| `noul` | Optional `true` and/or `false` descriptions | Truth/support judgment from 0.01 to 0.99, derived from the model's distribution over nine rating tokens. |

Choice and score confidence is the largest probability among their allowed labels. These distributions, and the Noul value, are not calibrated probabilities of correctness.

For chat input, replace `state` with a `messages` array such as:

```json
[
  {"role": "user", "content": "Mia owns a red bicycle. Her dog is named Max."}
]
```

Unknown top-level request fields are ignored, including completion settings such as `temperature`, `max_tokens`, and `stream`. Unknown fields inside questions and options are rejected. There is no completion sampling or streaming. The HF server currently supports text only; images, audio, video, and tool calls are unsupported.

`usage.input_tokens` counts unique token prefixes within the request, sharing the common context across questions. `usage.output_tokens` is zero because no output tokens are generated. For diagnostic timings, start the server with `ENABLE_OPEN_JEV_ADVANCED_METRICS=1`; adding `"options": {"raw_logits": true}` to a request then includes selected-token logits.

## What is a classifier, and why “System One”?

A classifier maps input to a defined set of answers. For example, a support system might route a message to `billing` or `technical`, score its urgency against an ordered rubric, and judge whether it requests a refund. Those decisions can feed directly into ordinary application code.

TypeSafe uses “System One” to describe models designed for fast, structured decisions, drawing the name from the distinction between fast intuitive thinking and slower deliberate reasoning. Its [introduction to System One and Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev) explains that motivation. Simple Jev explores this style of interface using existing open language models. It does not reproduce TypeSafe's model architecture or training, or establish equivalent accuracy, calibration, or speed.

In this implementation, the useful change is how the model is used:

1. The shared prompt builder creates consistent classifier instructions and one scoring branch per question.
2. The HF server renders those instructions and the context into the model's native chat format.
3. It evaluates the exact common token prefix once and reuses that prefix's KV cache across batches of question suffixes.
4. It reads the next-token logits for the allowed answer labels. Shared scoring code normalizes those scores and constructs the JSON response.

This avoids generating and parsing a prose or JSON answer token by token. Reusing the context can also reduce repeated computation when several questions refer to the same input. Actual latency depends on the model, hardware, context length, and number of questions. Cache reuse currently lasts only for a single request.

A valid response structure does not guarantee a correct decision. Model capability and question wording still matter; evaluate answer quality on your own task separately from validating the server's inference and response pipeline.

## How we speed up questions using shared prompts and only prefills

## Shared prompt contract and project layout

| Location | Purpose |
| --- | --- |
| [`common/`](common/README.md) | Plain Python modules for `ClassifierRequest`, prompt planning, and response scoring. No separate package installation is required. |
| [`common/PROMPT_STRUCTURE_V1.md`](common/PROMPT_STRUCTURE_V1.md) | Language-independent v1 specification: inputs, prompt strings, chat roles, answer labels, and scoring rules. |
| [`hf-server/hf_server.py`](hf-server/hf_server.py) | Single-file Transformers implementation: chat rendering, model loading, cached inference, HTTP API, and CLI. |
| [`hf-server/API_REFERENCE.md`](hf-server/API_REFERENCE.md) | Detailed request/response contract, validation, diagnostics, and configuration. |

`prepare_prompt(request, version="v1")` returns a cacheable system prompt prefix, prefix instruction, suffix instruction, and ordered questions. The inference implementation handles chat formatting and model execution; `common/response_scoring.py` converts label logits or mapped PyTorch tensors into answers.

The version fixes one prompt/scoring configuration so implementations can stay consistent, including implementations in other languages. It defaults to `v1`; the HTTP API currently uses that version. There are no per-request independent/rating modes or score-format switches.

## Testing

From the repository root, with the environment activated:

```bash
python -m pip install -e './hf-server[test]'
python -m pytest -c hf-server/pyproject.toml common/tests hf-server/tests -q
```

The tests cover request validation, prompt construction, response scoring, tensor/token mapping, HTTP behavior, and cached-versus-full inference using tiny locally initialized models. They do not require downloading pretrained model weights and do not measure classification accuracy.

Models need a supported Transformers implementation, a usable chat template, compatible cache operations, and answer labels that each extend the rendered prompt by exactly one distinct token. The server checks label tokenization; compatibility with every open model is not guaranteed.
