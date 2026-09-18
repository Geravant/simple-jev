/* Buildless question playground. Editing never sends a request; Run submits a
 * snapshot. All user/model content is rendered with textContent, never HTML. */
import {
  QUESTION_EXAMPLES,
  SCENARIOS,
  MAX_QUESTIONS,
  buildQuestions,
  validateResponse,
} from "./playground.mjs";
const API_BASE = "https://simple-jev-demo-api.featherless.ai/v1";
const DEFAULT_MODEL = "featherless-ai/gemma-4-26B-A4B-classifier";
const $ = (id) => document.getElementById(id);
const pretty = (value) => JSON.stringify(value, null, 2);
const percentage = (value) => `${Math.round(value * 100)}%`;
const state = {
  busy: false,
  nextAllowedAt: 0,
  response: null,
  rawResponse: null,
  request: null,
  questions: [],
  scenario: "support",
  serial: 0,
};

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}
function currentRequest() {
  return {
    model: $("model").value,
    state: $("context").value.trim(),
    questions: buildQuestions(state.questions),
  };
}
function syncRequest() {
  $("character-count").textContent =
    `${$("context").value.length.toLocaleString()} / 1,200`;
  try {
    $("request-code").textContent = pretty(currentRequest());
    $("builder-error").hidden = true;
    $("copy-request").disabled = false;
    $("run-button").disabled = state.busy;
  } catch (error) {
    $("builder-error").textContent = error.message;
    $("builder-error").hidden = false;
    $("request-code").textContent =
      `Complete the question editor to build a valid request.\n\n${error.message}`;
    $("copy-request").disabled = true;
    $("run-button").disabled = true;
  }
  $("question-count").textContent =
    `${state.questions.length} / ${MAX_QUESTIONS}`;
  $("add-question").disabled =
    state.busy || state.questions.length >= MAX_QUESTIONS;
}
function clearResults() {
  state.response = null;
  state.rawResponse = null;
  state.request = null;
  $("result-source").textContent = "Ready to run";
  $("result-source").classList.remove("live");
  $("result-status").textContent =
    "Edit your questions, then run the classifier.";
  $("elapsed").textContent = "";
  $("error-message").hidden = true;
  $("response-code-title").textContent = "Response";
  $("response-code").textContent =
    "Run the classifier to inspect its actual JSON response.";
  $("copy-response").disabled = true;
  renderResults(null, null);
}
function changed() {
  clearResults();
  syncRequest();
}

function newQuestion(template) {
  const question = structuredClone(QUESTION_EXAMPLES[template]);
  const base = question.id;
  let suffix = 2;
  while (state.questions.some((q) => q.id === question.id))
    question.id = `${base}_${suffix++}`;
  question.uid = ++state.serial;
  return question;
}
function field(labelText, input) {
  const label = element("label", "question-field");
  label.append(element("span", "", labelText), input);
  return label;
}
function renderEditor(openUid = null) {
  const open = new Set(
    [...$("question-editor").querySelectorAll("details[open]")].map((n) =>
      Number(n.dataset.uid),
    ),
  );
  if (openUid !== null) open.add(openUid);
  $("question-editor").replaceChildren(
    ...state.questions.map((q, index) => {
      const details = element("details", "question-item");
      details.dataset.uid = q.uid;
      details.open = open.has(q.uid);
      const summary = element("summary");
      const title = element(
        "span",
        "question-summary",
        q.instructions || "Write your question",
      );
      const badge = element("span", "type-label", q.type.toUpperCase());
      summary.append(
        element("span", "question-index", String(index + 1).padStart(2, "0")),
        title,
        badge,
      );
      const body = element("div", "question-body");
      const id = element("input");
      id.value = q.id;
      id.maxLength = 60;
      id.autocomplete = "off";
      id.addEventListener("input", () => {
        q.id = id.value;
        changed();
      });
      const type = element("select");
      for (const [value, label] of [
        ["choice", "Choice — pick an option"],
        ["score", "Score — ordered rubric"],
        ["noul", "Noul — yes/no support"],
      ]) {
        const option = element("option", "", label);
        option.value = value;
        type.append(option);
      }
      type.value = q.type;
      type.addEventListener("change", () => {
        q.type = type.value;
        q.criteriaText = QUESTION_EXAMPLES[`custom_${type.value}`].criteriaText;
        renderEditor(q.uid);
        changed();
      });
      const top = element("div", "question-fields");
      top.append(field("Answer ID", id), field("Question type", type));
      const instructions = element("textarea");
      instructions.value = q.instructions;
      instructions.rows = 2;
      instructions.maxLength = 400;
      instructions.addEventListener("input", () => {
        q.instructions = instructions.value;
        title.textContent = q.instructions || "Write your question";
        changed();
      });
      body.append(top, field("What do you want to ask?", instructions));
      if (q.type !== "noul") {
        const criteria = element("textarea");
        criteria.value = q.criteriaText;
        criteria.rows = 3;
        criteria.maxLength = 2400;
        criteria.spellcheck = false;
        criteria.addEventListener("input", () => {
          q.criteriaText = criteria.value;
          changed();
        });
        body.append(
          field(
            q.type === "choice"
              ? "Choices — one per line"
              : "Rubric — lowest to highest, one per line",
            criteria,
          ),
        );
        body.append(
          element(
            "p",
            "editor-hint",
            q.type === "choice"
              ? "Use an answer ID, optionally followed by | and a description."
              : "The score is an expected zero-based level, from 0 to the last level.",
          ),
        );
      } else
        body.append(
          element(
            "p",
            "editor-hint",
            "Ask a yes/no proposition. The result is support for “yes” on a 0.01–0.99 scale.",
          ),
        );
      const remove = element("button", "remove-question", "Remove question");
      remove.type = "button";
      remove.setAttribute("aria-label", `Remove question ${index + 1}`);
      remove.addEventListener("click", () => {
        state.questions = state.questions.filter((item) => item.uid !== q.uid);
        renderEditor();
        changed();
        $("add-question").focus();
      });
      body.append(remove);
      details.append(summary, body);
      return details;
    }),
  );
}
function loadScenario(key) {
  if (state.busy) return;
  state.scenario = key;
  state.questions = [];
  for (const template of SCENARIOS[key].questions)
    state.questions.push(newQuestion(template));
  $("context").value = SCENARIOS[key].examples[0][1];
  $("context-presets").replaceChildren(
    ...SCENARIOS[key].examples.map(([label, text], index) => {
      const button = element(
        "button",
        `preset${index === 0 ? " selected" : ""}`,
        label,
      );
      button.type = "button";
      button.setAttribute("aria-pressed", String(index === 0));
      button.addEventListener("click", () => {
        $("context").value = text;
        contextChanged();
      });
      return button;
    }),
  );
  renderEditor(state.questions[0].uid);
  changed();
}
function contextChanged() {
  [...$("context-presets").children].forEach((button, i) => {
    const selected =
      $("context").value === SCENARIOS[state.scenario].examples[i][1];
    button.classList.toggle("selected", selected);
    button.setAttribute("aria-pressed", String(selected));
  });
  changed();
}
function bars(probabilities, labels) {
  const list = element("div", "probability-list");
  const best = Math.max(...Object.values(probabilities));
  for (const [id, label] of labels) {
    const row = element("div", "probability-row");
    const track = element("div", "bar-track");
    const fill = element(
      "div",
      `bar-fill${probabilities[id] === best ? "" : " secondary"}`,
    );
    fill.style.width = `${probabilities[id] * 100}%`;
    track.append(fill);
    row.append(
      element("span", "", label),
      track,
      element("span", "", percentage(probabilities[id])),
    );
    list.append(row);
  }
  return list;
}
function renderResults(data, request) {
  const questions = request
    ? Object.entries(request.questions)
    : state.questions.map((q) => [
        q.id,
        { type: q.type, instructions: q.instructions },
      ]);
  $("decision-results").replaceChildren(
    ...questions.map(([id, q]) => {
      const card = element("article", "decision-card");
      const label = element("div", "decision-label");
      label.append(
        element("span", "", id || "Unnamed question"),
        element("span", "type-label", q.type.toUpperCase()),
      );
      card.append(
        label,
        element(
          "p",
          "result-question",
          q.instructions || "Write your question in the editor.",
        ),
      );
      const answer = data?.answers[id];
      if (!answer) {
        card.append(
          element("strong", "compact-value", "—"),
          element("p", "decision-detail", "Your result will appear here."),
        );
        return card;
      }
      if (q.type === "choice") {
        card.append(
          element("strong", "compact-value", answer.choice),
          element(
            "p",
            "decision-detail",
            `${percentage(answer.confidence)} label probability`,
          ),
          bars(
            answer.probabilities,
            Object.keys(q.criteria).map((key) => [key, key]),
          ),
        );
      } else if (q.type === "score") {
        card.append(
          element(
            "strong",
            "compact-value",
            `${answer.score.toFixed(2)} / ${q.criteria.length - 1}`,
          ),
          element(
            "p",
            "decision-detail",
            "Expected rubric level · lowest to highest",
          ),
          bars(
            answer.probabilities,
            q.criteria.map((text, i) => [String(i), `${i}: ${text}`]),
          ),
        );
      } else
        card.append(
          element("strong", "compact-value", percentage(answer.noul)),
          element("p", "refund-caption", "Support for “yes”"),
          element("p", "decision-detail", "A model judgment, not a guarantee."),
        );
      return card;
    }),
  );
  if (!questions.length)
    $("decision-results").append(
      element("p", "empty-results", "Add a question to get started."),
    );
}
function showResult(data, request, elapsed) {
  state.response = data;
  renderResults(data, request);
  $("result-source").textContent = "Live API result";
  $("result-source").classList.add("live");
  const count = Object.keys(request.questions).length;
  $("result-status").textContent =
    `${count} question${count === 1 ? "" : "s"} scored in one request.`;
  $("elapsed").textContent =
    `${Math.round(elapsed).toLocaleString()} ms round trip`;
}
function setBusy(busy) {
  state.busy = busy;
  document
    .querySelector(".output-panel")
    .setAttribute("aria-busy", String(busy));
  $("demo-form")
    .querySelectorAll("input,textarea,select,button")
    .forEach((control) => {
      control.disabled = busy;
    });
  $("run-label").textContent = busy ? "Classifying…" : "Run classifier";
  if (!busy) syncRequest();
}
async function runClassifier() {
  if (state.busy) throw new Error("A request is already running.");
  let request;
  try {
    request = currentRequest();
  } catch (error) {
    $("builder-error").textContent = error.message;
    $("builder-error").hidden = false;
    throw error;
  }
  if (!request.state || request.state.length > 1200) {
    clearResults();
    $("error-message").textContent =
      "Enter a message between 1 and 1,200 characters.";
    $("error-message").hidden = false;
    $("context").focus();
    throw new Error("Enter a message between 1 and 1,200 characters.");
  }
  if (Date.now() < state.nextAllowedAt) {
    const seconds = Math.ceil((state.nextAllowedAt - Date.now()) / 1000);
    $("error-message").textContent =
      `Please wait ${seconds} second${seconds === 1 ? "" : "s"} before trying again.`;
    $("error-message").hidden = false;
    throw new Error("Rate limit cooldown is active.");
  }
  clearResults();
  setBusy(true);
  $("result-source").textContent = "Running";
  $("result-status").textContent = "Waiting for the model…";
  state.request = request;
  state.nextAllowedAt = Date.now() + 750;
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), 45000);
  const started = performance.now();
  try {
    const response = await fetch(`${API_BASE}/classifier`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request),
      signal: controller.signal,
      credentials: "omit",
    });
    const data = await response.json().catch(() => null);
    // Preserve API error payloads as well as successful answers. A scoring
    // failure is not evidence of an invalid/overlong customer message.
    state.rawResponse = data;
    $("response-code-title").textContent = `Response · HTTP ${response.status}`;
    $("response-code").textContent =
      data === null ? "The API returned a non-JSON response." : pretty(data);
    $("copy-response").disabled = data === null;
    if (!response.ok) {
      if (response.status === 429) {
        const retry = Number(response.headers.get("Retry-After"));
        state.nextAllowedAt =
          Date.now() +
          (Number.isFinite(retry) && retry > 0 ? retry * 1000 : 3000);
        throw new Error(
          "The public demo is busy (2 requests per second). Wait a moment, then try again.",
        );
      }
      if (
        response.status === 422 ||
        response.status === 413 ||
        response.status === 400
      ) {
        const detail =
          typeof data?.detail === "string"
            ? data.detail
            : typeof data?.error?.message === "string"
              ? data.error.message
              : null;
        if (detail && /expected.*finite logit/i.test(detail)) {
          throw new Error(
            `The model's answer scoring failed in the API: “${detail}”. Try another model. The full error is available under “Under the hood”.`,
          );
        }
        throw new Error(
          detail
            ? `Request not accepted: ${detail.slice(0, 300)}`
            : `The API rejected this request (HTTP ${response.status}). Inspect the response under “Under the hood” or try another model.`,
        );
      }
      throw new Error(
        `The demo is temporarily unavailable (HTTP ${response.status}). Please try again shortly.`,
      );
    }
    validateResponse(data, request);
    showResult(data, request, performance.now() - started);
    return data;
  } catch (error) {
    const message =
      error.name === "AbortError"
        ? "The model took too long to respond. Please try again."
        : error instanceof TypeError
          ? "Could not reach the demo. Check your connection and try again."
          : error.message;
    $("error-message").textContent = message;
    $("error-message").hidden = false;
    $("result-source").textContent = "No result";
    $("result-status").textContent = "This request did not produce a result.";
    throw error;
  } finally {
    clearTimeout(timeout);
    setBusy(false);
  }
}

async function loadModels() {
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), 10000);
  try {
    const response = await fetch(`${API_BASE}/models`, {
      signal: controller.signal,
      credentials: "omit",
    });
    if (!response.ok) throw new Error("Model list unavailable");
    const data = await response.json();
    const ids = [
      ...new Set(
        (Array.isArray(data.data) ? data.data : [])
          .map((model) => model.id)
          .filter((id) => typeof id === "string" && id.length > 0),
      ),
    ];
    if (!ids.length) throw new Error("No models available");
    // Don't change the selection halfway through a request if model discovery is slow.
    if (state.busy) return;
    const previousModel = $("model").value;
    $("model").replaceChildren(
      ...ids.map((id) => {
        const option = document.createElement("option");
        option.value = id;
        option.textContent =
          id === DEFAULT_MODEL
            ? "Gemma 4 · 26B A4B"
            : id.replace(/^featherless-ai\//, "").replace(/-classifier$/, "");
        return option;
      }),
    );
    $("model").value = ids.includes(previousModel) ? previousModel : ids[0];
    if ($("model").value !== previousModel) clearResults();
    syncRequest();
  } catch {
    $("model-notice").textContent =
      "Couldn’t refresh the model list. You can still try the default Gemma model.";
    $("model-notice").hidden = false;
  } finally {
    clearTimeout(timeout);
  }
}

$("demo-form").addEventListener("submit", (event) => {
  event.preventDefault();
  runClassifier().catch(() => {});
});
$("context").addEventListener("input", contextChanged);
$("context").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) {
    event.preventDefault();
    if (!state.busy) $("demo-form").requestSubmit();
  }
});
$("model").addEventListener("change", changed);
$("load-scenario").addEventListener("click", () =>
  loadScenario($("scenario").value),
);
$("add-question").addEventListener("click", () => {
  if (state.busy || state.questions.length >= MAX_QUESTIONS) return;
  const question = newQuestion($("question-example").value);
  state.questions.push(question);
  renderEditor(question.uid);
  changed();
  const item = [...$("question-editor").children].find(
    (n) => Number(n.dataset.uid) === question.uid,
  );
  item?.querySelector("textarea")?.focus();
});
async function copyText(button, text) {
  try {
    await navigator.clipboard.writeText(text);
    $("copy-status").textContent = "Copied to clipboard.";
    const previous = button.textContent;
    button.textContent = "Copied";
    setTimeout(() => (button.textContent = previous), 1500);
  } catch {
    $("copy-status").textContent =
      "Clipboard unavailable. Select and copy the JSON manually.";
    const pre =
      button.id === "copy-request" ? $("request-code") : $("response-code");
    const range = document.createRange();
    range.selectNodeContents(pre);
    window.getSelection().removeAllRanges();
    window.getSelection().addRange(range);
    pre.focus();
  }
}
$("copy-request").addEventListener("click", (event) => {
  try {
    copyText(event.currentTarget, pretty(currentRequest()));
  } catch {
    syncRequest();
  }
});
$("copy-response").addEventListener("click", (event) => {
  if (state.rawResponse !== null)
    copyText(event.currentTarget, pretty(state.rawResponse));
});
loadScenario("support");
loadModels();

// Optional agent actions use the same validated builder as the visible form.
if (document.modelContext?.registerTool) {
  const lifecycle = new AbortController();
  window.addEventListener("pagehide", () => lifecycle.abort(), { once: true });
  const tools = [
    {
      name: "stage_classifier_message",
      description:
        "Edit the context without sending a request; preserves the current questions.",
      inputSchema: {
        type: "object",
        properties: {
          message: { type: "string", minLength: 1, maxLength: 1200 },
        },
        required: ["message"],
        additionalProperties: false,
      },
      annotations: { readOnlyHint: false },
      execute(input) {
        if (
          state.busy ||
          !input ||
          Object.keys(input).some((k) => k !== "message") ||
          typeof input.message !== "string" ||
          !input.message.trim() ||
          input.message.length > 1200
        )
          throw new Error("Provide 1–1,200 characters while idle.");
        $("context").value = input.message;
        contextChanged();
        return { message: input.message };
      },
    },
    {
      name: "run_classifier",
      description:
        "Send the context and all edited questions to the public Featherless API and display the decisions.",
      inputSchema: {
        type: "object",
        properties: {},
        additionalProperties: false,
      },
      annotations: { readOnlyHint: false, untrustedContentHint: true },
      execute(input) {
        if (!input || Object.keys(input).length)
          throw new Error("Supply an empty object.");
        return runClassifier();
      },
    },
  ];
  for (const tool of tools) {
    try {
      Promise.resolve(
        document.modelContext.registerTool(tool, { signal: lifecycle.signal }),
      ).catch(() => {});
    } catch {
      /* Optional integration. */
    }
  }
}
