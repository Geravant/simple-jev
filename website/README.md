# simple-jev.com — static site

A demo-first design for Simple Jev. Plain HTML, CSS, and JavaScript; no build, framework, keys, or website backend.

The dedicated [API documentation](docs.html) covers the request contract, all three question types, response interpretation, runnable client examples, errors, and self-hosting differences.

## Editable live playground

Visitors can load customer support, product review, or community moderation scenarios. Each scenario includes several example contexts and an editable starting question set. Loading a scenario explicitly replaces the context and questions; choosing a context preset only changes the context.

The question editor supports:

- Adding common question examples or writing a custom question.
- Editing answer IDs, question types, and instructions.
- Choice options as `answer_id | optional description`, one per line.
- Score rubrics in low-to-high order, one level per line.
- Noul yes/no propositions, returned as support for “yes.”
- Removing questions and running up to six questions per demo request.

Requests update immediately in the JSON inspector, but editing never sends context to the API. Run submits the current valid request. Blank questions, duplicate answer IDs, duplicate choice IDs, and invalid criteria counts are rejected locally. All questions share the demo's 2k-token context budget; six is a UI cap, not an API-wide limit.

Results are rendered from the submitted question definitions, including arbitrary answer IDs and rubric lengths. Scores display their exact expected zero-based level and a rubric distribution. Every edit clears previous results. There are no simulated initial scores.

## Preview

From the repository root:

```bash
python3 -m http.server 8765 --bind 127.0.0.1 --directory website
```

Open the [playground](http://127.0.0.1:8765/) or the [API documentation](http://127.0.0.1:8765/docs.html). Serving over HTTP/HTTPS is recommended; do not rely on opening the HTML through `file://`.

## Files

- `index.html`: homepage, API example, and link to the playground.
- `playground.html`: dedicated interactive editor and results page.
- `syntax.js`: safe syntax highlighting for static examples and live JSON.
- `docs.html`: detailed API reference with copyable curl, JavaScript, and Python examples.
- `docs.css`: documentation layout and responsive reference tables.
- `docs.js`: code example copy controls; no API requests.
- `styles.css`: responsive styles, keyboard focus, and reduced-motion support.
- `demo.js`: dynamic editor, model discovery, classification, result rendering, and copy controls.
- `playground.mjs`: scenario/question examples, request construction, and response validation.
- `tests/playground.test.mjs`: offline request/response contract tests (`node --test website/tests/playground.test.mjs`).
- `assets/simple-jev.png`: the official Simple Jev mascot badge, used as the logo and favicon.
- `assets/featherless.png`: supplied Featherless logo for the “Built by Featherless.ai” attribution.

## API behavior

The client uses `https://simple-jev-demo-api.featherless.ai/v1/models` and `/v1/classifier`, with Gemma selected initially. The model list is fetched on load. Context is sent only after a visitor presses Run (or explicitly invokes the page's `run_classifier` WebMCP tool).

The API allows cross-origin browser requests and requires no authentication. The page displays the demo's 2k-token context and 2 RPS limits. A 1,200-character input cap is a UI convenience, not a token-count guarantee; the API enforces its real context limit. Requests time out after 45 seconds, double submission is blocked, and HTTP 429 activates a retry cooldown. The page never fabricates a successful response when the API fails.

The optional, feature-detected WebMCP tools share the visible UI actions: `stage_classifier_message` only edits the form, while `run_classifier` sends the request. Unsupported browsers ignore this integration. No analytics, local storage, or client-side secret is used. The public API's own data handling is separate from this page.

## Hosting

Serve this folder's public files from any static host and point `simple-jev.com` to that host when ready. There is no build step. Deploy `index.html`, `playground.html`, `syntax.js`, `styles.css`, `demo.js`, `playground.mjs`, `docs.html`, `docs.css`, `docs.js`, and `assets/`; the README is not needed. Use HTTPS. This work does not configure DNS or publish the domain.

The live API model list and one real classification were checked during development. The static page also includes loading, validation, network-error, and rate-limit handling. Classification correctness depends on the selected model; demo outputs are not benchmark results.

## Gemma mixed-question diagnostic

On 2026-09-18, the original App outage preset reproduced an API-side error with `featherless-ai/gemma-4-26B-A4B-classifier`, including in direct HTTP requests outside the browser:

| Questions in the request | Observed result                         |
| ------------------------ | --------------------------------------- |
| Routing + urgency        | HTTP 200                                |
| Refund (Noul) alone      | HTTP 200                                |
| All three together       | HTTP 422: `Expected nine finite logits` |

The combined browser request also returned `Expected one finite logit per choice` on a later attempt. These are API-side answer-scoring failures; the underlying serving bug has not been established. The successful isolated requests above are observations, not a guarantee that isolated calls always succeed. The page preserves the error JSON for inspection/copying and distinguishes this scoring error from a context-length problem. It does not silently switch models, replace Noul with another question type, or manufacture scores. Other inputs, including the Double charge preset, have returned successfully on Gemma.
