import { directions, slide, spawn } from "./engine.js";
import {
  classify,
  telemetry as usageTelemetry,
  models,
  loadModels,
  instructionFor,
} from "./api.js";
const el = (id) => document.getElementById(id),
  canvas = el("board"),
  ctx = canvas.getContext("2d");
let board = [],
  score = 0,
  moves = 0,
  epoch = 0,
  running = false,
  busy = false,
  lastAction = null;
let animating = false,
  animationVersion = 0;
const moveHistory = [];
const modeLabel = document.createElement("label");
modeLabel.innerHTML =
  'Input mode <select id="input-mode"><option value="text" selected>Text grid</option><option value="image">Image</option></select>';
el("model").after(modeLabel);
el("input-mode").onchange = () => {
  el("prompt").textContent = instructionFor(el("input-mode").value);
};
const historyPanel = document.createElement("section");
historyPanel.className = "metrics";
historyPanel.innerHTML =
  '<h2>Recent moves</h2><label>Show last <select id="history-count"><option>10</option><option selected>20</option><option>50</option><option>100</option></select> moves · newest first</label><pre id="move-history">No moves yet.</pre><details><summary>Current grid (sent only in Text mode)</summary><pre id="text-grid"></pre></details>';
document.querySelector(".layout").after(historyPanel);
function renderHistory() {
  el("move-history").textContent =
    moveHistory
      .slice(-Number(el("history-count").value))
      .reverse()
      .map(
        (m) =>
          `#${m.number}  ${m.direction.padEnd(5)}  ${m.source}  Largest tile ${m.largest}`,
      )
      .join("\n") || "No moves yet.";
  el("text-grid").textContent = Array.from({ length: 4 }, (_, r) =>
    board
      .slice(r * 4, r * 4 + 4)
      .map((v) => String(v).padStart(5))
      .join(" "),
  ).join("\n");
}
el("history-count").onchange = renderHistory;
const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
const colors = {
  2: "#eee4da",
  4: "#ede0c8",
  8: "#f2b179",
  16: "#f59563",
  32: "#f67c5f",
  64: "#f65e3b",
  128: "#edcf72",
  256: "#edcc61",
  512: "#edc850",
  1024: "#edc53f",
  2048: "#edc22e",
};
const legal = () => directions.filter((d) => slide(board, d).changed);
function draw() {
  ctx.fillStyle = "#bbada0";
  ctx.fillRect(0, 0, 400, 400);
  board.forEach((v, i) => {
    const x = 10 + (i % 4) * 98,
      y = 10 + Math.floor(i / 4) * 98;
    ctx.fillStyle = colors[v] ?? (v ? "#3c3a32" : "#cdc1b4");
    ctx.fillRect(x, y, 88, 88);
    if (v) {
      ctx.fillStyle = v <= 4 ? "#776e65" : "#fff";
      ctx.font = `bold ${v >= 1024 ? 28 : 36}px system-ui`;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(String(v), x + 44, y + 45);
    }
  });
  canvas.setAttribute("aria-label", `2048 board: ${board.join(", ")}`);
  el("score").textContent =
    `Score ${score} · Largest tile ${Math.max(...board)} · Moves ${moves}`;
}
function pause() {
  epoch++;
  running = false;
  update();
}
function update() {
  el("start").disabled = running || busy || animating || !models.length;
  el("step").disabled = running || busy || animating || !models.length;
  el("model").disabled = running || busy || animating || !models.length;
  el("input-mode").disabled = running || busy || animating || !models.length;
}
function bars(p = {}, chosen = null) {
  el("bars").replaceChildren(
    ...directions.map((d) => {
      const row = document.createElement("div");
      row.className = "row" + (chosen === d ? " chosen" : "");
      row.textContent = `${chosen === d ? "▶ " : ""}${d} · ${p[d] === undefined ? "—" : (p[d] * 100).toFixed(1) + "%"}`;
      const track = document.createElement("div");
      track.className = "track";
      const fill = document.createElement("div");
      fill.className = "fill";
      fill.style.width = `${(p[d] ?? 0) * 100}%`;
      track.append(fill);
      row.append(track);
      return row;
    }),
  );
}
function tile(value, x, y, scale = 1, highlight = false) {
  ctx.save();
  ctx.translate(x + 44, y + 44);
  ctx.scale(scale, scale);
  ctx.fillStyle = colors[value] ?? "#3c3a32";
  ctx.fillRect(-44, -44, 88, 88);
  if (highlight) {
    ctx.strokeStyle = "#fff";
    ctx.lineWidth = 4;
    ctx.strokeRect(-42, -42, 84, 84);
  }
  ctx.fillStyle = value <= 4 ? "#776e65" : "#fff";
  ctx.font = `bold ${value >= 1024 ? 28 : 36}px system-ui`;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(String(value), 0, 1);
  ctx.restore();
}
function background() {
  ctx.fillStyle = "#bbada0";
  ctx.fillRect(0, 0, 400, 400);
  ctx.fillStyle = "#cdc1b4";
  for (let i = 0; i < 16; i++)
    ctx.fillRect(10 + (i % 4) * 98, 10 + Math.floor(i / 4) * 98, 88, 88);
}
function animate(duration, version, paint) {
  return new Promise((resolve) => {
    const start = performance.now();
    function frame(now) {
      if (version !== animationVersion) {
        resolve(false);
        return;
      }
      const t = Math.min(1, (now - start) / duration);
      paint(t);
      if (t < 1) requestAnimationFrame(frame);
      else resolve(true);
    }
    requestAnimationFrame(frame);
  });
}
async function move(d, source = "Manual") {
  if (animating) return;
  const result = slide(board, d);
  if (!result.changed) return;
  const version = ++animationVersion;
  animating = true;
  canvas.dataset.animating = "true";
  update();
  try {
    if (
      !reducedMotion.matches &&
      !(await animate(320, version, (t) => {
        background();
        const ease = 1 - (1 - t) ** 3;
        result.transitions.forEach(({ from, to, value }) =>
          tile(
            value,
            10 + ((from % 4) + ((to % 4) - (from % 4)) * ease) * 98,
            10 +
              (Math.floor(from / 4) +
                (Math.floor(to / 4) - Math.floor(from / 4)) * ease) *
                98,
          ),
        );
      }))
    )
      return;
    if (version !== animationVersion) return;
    board = spawn(result.board.slice());
    score += result.score;
    moves++;
    lastAction = d;
    moveHistory.push({
      number: moves,
      direction: d,
      source,
      largest: Math.max(...board),
    });
    if (moveHistory.length > 100) moveHistory.shift();
    renderHistory();
    const newTile = board.findIndex((v, i) => v !== result.board[i]);
    if (!reducedMotion.matches)
      await animate(240, version, (t) => {
        background();
        board.forEach((v, i) => {
          if (v)
            tile(
              v,
              10 + (i % 4) * 98,
              10 + Math.floor(i / 4) * 98,
              i === newTile
                ? 0.35 + 0.65 * t
                : result.merges.includes(i)
                  ? 1 + 0.12 * Math.sin(Math.PI * t)
                  : 1,
              i === newTile,
            );
        });
      });
    if (version !== animationVersion) return;
    draw();
    if (!legal().length) {
      pause();
      el("status").textContent =
        "Game over — no legal moves. Start a new game.";
    } else if (Math.max(...board) >= 2048) {
      pause();
      el("status").textContent =
        "2048 reached! You can continue manually or restart the classifier.";
    }
  } finally {
    if (version === animationVersion) {
      animating = false;
      canvas.dataset.animating = "false";
      update();
    }
  }
}
async function telemetry() {
  const t = await usageTelemetry();
  if (!el("model").options.length) {
    models.forEach((m) => el("model").add(new Option(m, m)));
    el("model").value = models.includes("featherless-ai/Qwen3.8-27B-classifier")
      ? "featherless-ai/Qwen3.8-27B-classifier"
      : models[0];
  }
  el("prompt").textContent = instructionFor(el("input-mode").value);
  el("totals").textContent =
    `Prompt ${t.totals.prompt} · Completion ${t.totals.completion} · Requests ${t.totals.requests} · Missing usage ${t.totals.missing}`;
  el("logs").textContent = JSON.stringify(t.entries, null, 2);
}
async function decide(token) {
  busy = true;
  update();
  el("status").textContent =
    `Classifying ${el("input-mode").value === "text" ? "the text grid" : "the board image"}…`;
  try {
    const result = await classify({
      board: board.slice(),
      model: el("model").value,
      lastAction,
      mode: el("input-mode").value,
      ...(el("input-mode").value === "image"
        ? { image: canvas.toDataURL("image/png") }
        : {}),
    });
    if (token !== epoch) return;
    bars(result.probabilities, result.action);
    el("last").textContent =
      `Last response: ${result.ms} ms · Usage ${JSON.stringify(result.usage ?? "unavailable")}`;
    if (!result.action) throw Error("Missing classifier action");
    el("status").textContent =
      `${result.local ? "Only legal move (no API call)" : "Chosen"} ${result.action}${result.tied ? " (random tie-break)" : ""} · ${result.ms} ms`;
    await move(result.action, "AI");
  } catch (e) {
    if (token === epoch) {
      pause();
      el("status").textContent = e.message;
    }
  } finally {
    busy = false;
    update();
    await telemetry().catch(() => {});
  }
}
async function run(single = false) {
  if (busy || running || !models.length) return;
  if (!legal().length) {
    el("status").textContent = "Game over — start a new game.";
    return;
  }
  const token = ++epoch;
  running = true;
  update();
  do {
    await decide(token);
    if (single) break;
    await new Promise((r) => setTimeout(r, 250));
  } while (running && token === epoch && legal().length);
  if (token === epoch) {
    running = false;
    update();
  }
}
el("start").onclick = () => run();
el("step").onclick = () => run(true);
el("stop").onclick = () => {
  pause();
  el("status").textContent =
    "Paused — arrow keys or buttons move manually. Pending responses will be ignored.";
};
async function manual(d) {
  pause();
  if (animating) return;
  bars();
  el("status").textContent = `Manual ${d} — classifier paused`;
  await move(d);
}
directions.forEach((d) => {
  const b = document.createElement("button");
  b.textContent = d;
  b.onclick = () => manual(d);
  el("controls").append(b);
});
window.addEventListener("keydown", (e) => {
  if (e.target.tagName === "SELECT") return;
  const d = {
    ArrowUp: "UP",
    ArrowDown: "DOWN",
    ArrowLeft: "LEFT",
    ArrowRight: "RIGHT",
  }[e.key];
  if (d) {
    e.preventDefault();
    manual(d);
  }
});
el("new").onclick = () => {
  animationVersion++;
  animating = false;
  canvas.dataset.animating = "false";
  pause();
  board = spawn(spawn(Array(16).fill(0)));
  score = 0;
  moves = 0;
  lastAction = null;
  moveHistory.length = 0;
  renderHistory();
  bars();
  draw();
  el("status").textContent = "New game — ready.";
};
el("new").click();
loadModels()
  .then(telemetry)
  .then(update)
  .catch((e) => {
    el("status").textContent = e.message + " — reload to retry";
    update();
  });

function syncVision() {
  const supported = /gemma|qwen/i.test(el("model").value);
  el("input-mode").querySelector('option[value="image"]').disabled = !supported;
  if (!supported) el("input-mode").value = "text";
  el("prompt").textContent = instructionFor(el("input-mode").value);
}
el("model").addEventListener("change", () => {
  pause();
  bars();
  syncVision();
});
new MutationObserver(syncVision).observe(el("model"), { childList: true });
// Swipes provide manual control on touchscreens and invalidate pending AI moves.
let touchStart = null;
canvas.addEventListener("pointerdown", (e) => {
  touchStart = { x: e.clientX, y: e.clientY };
  canvas.setPointerCapture(e.pointerId);
});
canvas.addEventListener("pointerup", (e) => {
  if (!touchStart) return;
  const dx = e.clientX - touchStart.x,
    dy = e.clientY - touchStart.y;
  touchStart = null;
  if (Math.max(Math.abs(dx), Math.abs(dy)) < 24) return;
  manual(
    Math.abs(dx) > Math.abs(dy)
      ? dx > 0
        ? "RIGHT"
        : "LEFT"
      : dy > 0
        ? "DOWN"
        : "UP",
  );
});
canvas.addEventListener("pointercancel", () => (touchStart = null));
