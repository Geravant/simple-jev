import { directions, slide } from "./engine.js";
import { chooseMove } from "./choice.js";
export const endpoint = "https://simple-jev-demo-api.featherless.ai/v1";
export const models = [];
export async function loadModels() {
  const r = await fetch(endpoint + "/models", {
    credentials: "omit",
    signal: AbortSignal.timeout(20000),
  });
  if (!r.ok) throw Error(`Models HTTP ${r.status}`);
  const b = await r.json();
  const ids = [
    ...new Set(
      (b.data ?? [])
        .map((m) => m.id)
        .filter((id) => typeof id === "string" && id),
    ),
  ];
  if (!ids.length) throw Error("No models available");
  models.splice(0, models.length, ...ids);
  return models;
}
export const instruction =
  "Play 2048 using the supplied 4×4 text grid. Rows run top to bottom and columns left to right; 0 means an empty cell. Choose exactly one legal direction: UP, DOWN, LEFT or RIGHT. All tiles slide in that direction; adjacent equal values merge once per move. A new 2 or 4 appears after each valid move. Aim for the largest possible tile, reaching 2048 while avoiding a full blocked board. Prefer keeping the largest tile in a consistent corner, maintaining ordered rows and empty cells, and merging small tiles without scattering large tiles. The board is frozen while you decide, so there is no timing pressure. Only select from the provided legal moves. Future spawned tiles are unknown. No image is supplied.";
export function instructionFor(mode = "text") {
  return mode === "image"
    ? instruction
        .replace(
          "supplied 4×4 text grid. Rows run top to bottom and columns left to right; 0 means an empty cell.",
          "supplied game-board image. Read the tiles visually; blank cells are empty.",
        )
        .replace("No image is supplied.", "No text grid is supplied.")
    : instruction;
}
const entries = [];
let totals = { requests: 0, prompt: 0, completion: 0, missing: 0 };
export async function telemetry() {
  return { totals, entries };
}
export async function classify({
  board,
  model,
  mode = "text",
  image,
  lastAction,
}) {
  if (!models.includes(model)) throw Error("Select an available model");
  const legal = directions.filter((d) => slide(board, d).changed);
  if (!legal.length) throw Error("No valid moves");
  const grid = Array.from({ length: 4 }, (_, r) =>
    board.slice(r * 4, r * 4 + 4).join(" "),
  ).join("\n");
  const context = `Valid moves: ${legal.join(", ")}. Previous action: ${lastAction ?? "none"}. Choose only a valid move.`;
  if (legal.length === 1)
    return {
      action: legal[0],
      probabilities: { [legal[0]]: 1 },
      ms: 0,
      usage: null,
      local: true,
    };
  if (mode === "image" && !/gemma|qwen/i.test(model))
    throw Error("Image mode requires a Gemma or Qwen model.");
  const candidates = legal;
  const request = {
    model,
    messages: [
      { role: "system", content: instructionFor(mode) },
      {
        role: "user",
        content:
          mode === "text"
            ? context + "\nCurrent grid (0 = empty):\n" + grid
            : [
                { type: "text", text: context },
                { type: "image_url", image_url: { url: image } },
              ],
      },
    ],
    questions: {
      move: {
        type: "choice",
        instructions:
          "Choose the best move from the valid moves listed in the message.",
        criteria: Object.fromEntries(
          candidates.map((d) => [d, `Slide ${d.toLowerCase()}`]),
        ),
      },
    },
  };
  const started = performance.now();
  let status = 0,
    body;
  try {
    const r = await fetch(endpoint + "/classifier", {
      method: "POST",
      credentials: "omit",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request),
      signal: AbortSignal.timeout(60000),
    });
    status = r.status;
    body = await r.json();
    if (!r.ok) throw Error(body.error?.message ?? `Classifier HTTP ${status}`);
    const p = body.answers?.move?.probabilities;
    if (
      !p ||
      !candidates.every((d) => Number.isFinite(p[d]) && p[d] >= 0 && p[d] <= 1)
    )
      throw Error("Invalid probabilities");
    return {
      ...chooseMove(p, legal),
      probabilities: p,
      ms: Math.round(performance.now() - started),
      usage: body.usage,
    };
  } finally {
    const usage = body?.usage,
      p = usage?.input_tokens ?? usage?.prompt_tokens,
      c = usage?.output_tokens ?? usage?.completion_tokens,
      valid = (n) => Number.isFinite(n) && n >= 0;
    totals.requests++;
    if (valid(p)) totals.prompt += p;
    if (valid(c)) totals.completion += c;
    if (!valid(p) || !valid(c)) totals.missing++;
    entries.unshift({
      time: new Date().toISOString(),
      model,
      mode,
      status,
      ms: Math.round(performance.now() - started),
      usage: usage ?? null,
      answers: body?.answers ?? null,
    });
    entries.splice(20);
  }
}
