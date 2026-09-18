# Simple Jev Classifier Models on Featherless

**Turn context into structured decisions.** Provide text, a conversation, or—on supported models—an image, then ask the model to choose an answer, score a rubric, or judge a proposition. Receive JSON that your application can use directly.

These models use the **Simple Jev classifier API** on Featherless. Useful applications include support routing, intent detection, content moderation, relevance scoring, visual classification, and choosing an agent’s next action.

**Currently in beta on Featherless developer plans.**

## How it works

Simple Jev scores the model’s logits for the allowed answer labels and converts them into structured results. You define the questions and possible answers in each request. Several questions can share the same context.

Use **`POST /v1/classifier`** for this interface. The examples below use classifier requests and responses, rather than chat-completion messages as the output format.

## Try it now — no API key required

This example uses Gemma. Replace `model` with any supported model ID in the table below.

```bash
curl https://simple-jev-demo-api.featherless.ai/v1/classifier \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "featherless-ai/gemma-4-26B-A4B-classifier",
    "state": "Mia owns a red bicycle.",
    "questions": {
      "color": {
        "type": "choice",
        "instructions": "What color is Mia’s bicycle?",
        "criteria": {
          "red": null,
          "blue": null
        }
      }
    }
  }'
```

Example answer, showing the `answers` field only. Values are illustrative; results may vary:

```json
{
  "answers": {
    "color": {
      "type": "choice",
      "choice": "red",
      "confidence": 0.99,
      "probabilities": {
        "red": 0.99,
        "blue": 0.01
      }
    }
  }
}
```

The full response also includes `model` and token `usage`. Your question IDs become keys in `answers`.

The public demo requires no login or authentication and is limited to **2k context and 4 requests per second**. The context budget includes instructions, questions, and model formatting as well as your input. Production limits are separate.

## Use it in production

Send the same request body to:

```text
https://api.featherless.ai/v1/classifier
```

Include these headers, using your Featherless developer API key:

```text
Authorization: Bearer YOUR_API_KEY
Content-Type: application/json
```

[Sign up for a developer account at Featherless.ai](https://featherless.ai/) for production usage and higher limits. Keep your API key on your application’s server rather than embedding it in a public website or client bundle.

## Three ways to ask a question

| Type | Input | Result | Example use |
| --- | --- | --- | --- |
| `choice` | Named candidates with optional descriptions | Selected candidate, confidence, and candidate probabilities | Route a ticket to billing, technical support, or account support |
| `score` | An ordered rubric, lowest to highest | Expected rubric index, confidence, probabilities, and rubric legend | Rate urgency or relevance |
| `noul` | A yes/no proposition | A value from 0.01 to 0.99; higher means more support for “yes” | Determine whether a customer requests a refund |

For `score`, rubric positions start at zero. A three-level rubric produces a score between **0 and 2**, which can be fractional.

Add questions under different keys in the same `questions` object to evaluate multiple decisions against the shared context. Each question is evaluated independently of the others’ answers.

## Supported context

- **Text and structured data:** pass a string, JSON object, or array in `state`.
- **Chat history:** use `messages` with roles to evaluate a conversation. Supply `messages` instead of `state`.
- **Images and vision:** Gemma and Qwen classifier models support image context. RWKV classifier models are text-only.

The quick-start example above uses text. Image input requires the hosted API’s supported image-message format; a plain image URL in `state` is not an image upload.

## Supported models and beta pricing

All models below support text and chat-history context. Prices are **per million input tokens** during beta on Featherless developer plans.

| Model ID | Input price | Images & vision |
| --- | ---: | --- |
| `featherless-ai/RWKV-small-classifier` | $0.03 | Text only |
| `featherless-ai/RWKV-mid-classifier` | $0.10 | Text only |
| `featherless-ai/RWKV-std-classifier` | $0.20 | Text only |
| `featherless-ai/gemma-4-26B-A4B-classifier` | $0.28 | Supported |
| `featherless-ai/Qwen3.6-35B-A3B-classifier` | $0.28 | Supported |
| `featherless-ai/Qwen3.8-27B-classifier` | $0.30 | Supported |

See each model’s listing for production context limits, cached-input pricing, and other applicable billing details. The public demo’s 2k context limit is separate from production model limits.

*Prices may change after beta. Refer to [Featherless.ai official pricing](https://featherless.ai/) for up-to-date information.*

## Interpreting results

Choice and score probabilities are normalized over the candidates or rubric you provide. They are model preferences, not automatically calibrated probabilities of correctness. Noul is a model judgment derived from rating-token scores.

Use clear candidate descriptions and explicit rubrics. Evaluate the model on representative examples before choosing thresholds or automating decisions. Classification quality depends on the task, context, and question wording.

## Open source and further reading

[Simple Jev on GitHub](https://github.com/featherless-ai/simple-jev) includes the shared prompt format, response scoring, an HF reference server, and RFDT fine-tuning scripts.

- [Classifier API reference](https://github.com/featherless-ai/simple-jev/blob/main/hf-server/API_REFERENCE.md) — shared request and response concepts, with details specific to the self-hosted HF implementation.
- [Prompt structure v1](https://github.com/featherless-ai/simple-jev/blob/main/common/PROMPT_STRUCTURE_V1.md) — the versioned, language-independent prompt contract.

Hosted fine-tuned models and fine-tuning support are planned as Simple Jev support and usage on Featherless grow.
