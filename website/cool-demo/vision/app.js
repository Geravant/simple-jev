import { fetchWithRetry, wait } from "./queue.mjs";
import { buildBatchRequest, answerFrom } from "./request.mjs";
const $ = (id) => document.getElementById(id),
  base = "https://simple-jev-demo-api.featherless.ai/v1";
const labels = {
  hot_dog: "Hot dog",
  sandwich: "Sandwich",
  neither: "Neither / unclear",
};
let catalog = [],
  busy = false,
  ready = false,
  controller = null;
const results = new Map();
function sync() {
  $("classify").disabled = busy || !ready || !catalog.length;
  $("classify").textContent = `Classify all (${catalog.length})`;
  $("model").disabled = busy || !ready;
  $("cancel").hidden = !busy;
}
function resultNode(id) {
  return document.getElementById(`result-${id}`);
}
function text(node, value) {
  node.textContent = value;
}
function render() {
  $("catalog").replaceChildren();
  for (const item of catalog) {
    const card = document.createElement("article");
    card.className = "photo-card";
    const label = document.createElement("div");
    label.className = "photo-label";
    const image = document.createElement("img");
    image.src = item.src;
    image.alt = `Food photograph ${item.id}`;
    image.loading = "lazy";
    const title = document.createElement("span");
    title.textContent = `Photo ${item.id}`;
    label.append(image, title);
    const source = document.createElement("a");
    source.className = "photo-source";
    source.href = item.source;
    source.target = "_blank";
    source.rel = "noopener noreferrer";
    source.textContent = `${item.credit} · photo source ↗`;
    const result = document.createElement("div");
    result.id = `result-${item.id}`;
    result.className = "photo-result";
    result.textContent = "Ready to classify.";
    card.append(label, source, result);
    $("catalog").append(card);
  }
  sync();
}
async function dataURL(src, signal) {
  const response = await fetch(src, { signal });
  if (!response.ok) throw Error("Photo could not be loaded.");
  const blob = await response.blob();
  return await new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(Error("Could not read the photo."));
    reader.readAsDataURL(blob);
  });
}
function showAnswer(node, answer, ms) {
  node.replaceChildren();
  const heading = document.createElement("strong");
  heading.textContent = labels[answer.choice];
  node.append(heading);
  for (const [key, label] of Object.entries(labels)) {
    const row = document.createElement("div");
    row.className = "probability";
    const name = document.createElement("span"),
      value = document.createElement("span");
    name.textContent = label;
    value.textContent = `${(answer.probabilities[key] * 100).toFixed(1)}%`;
    row.append(name, value);
    const bar = document.createElement("progress");
    bar.max = 1;
    bar.value = answer.probabilities[key];
    bar.setAttribute("aria-label", `${label} probability`);
    node.append(row, bar);
  }
  const time = document.createElement("small");
  time.textContent = `${ms} ms · actual API response`;
  node.append(time);
}
$("classify").addEventListener("click", async () => {
  if (busy || !ready || !catalog.length) return;
  busy = true;
  controller = new AbortController();
  results.clear();
  sync();
  const signal = controller.signal,
    model = $("model").value;
  let completed = 0,
    failed = 0,
    requests = 0,
    batchNumber = 0;
  const batches = [];
  for (let i = 0; i < catalog.length; i += 4)
    batches.push(catalog.slice(i, i + 4));
  for (const item of catalog) {
    resultNode(item.id).classList.remove("error");
    text(resultNode(item.id), "Queued…");
  }
  $("responses").textContent = "Waiting for responses…";
  try {
    for (const batch of batches) {
      signal.throwIfAborted();
      if (batchNumber) await wait(500, signal);
      batchNumber++;
      const started = performance.now();
      const progress = () =>
        `${completed + failed} / ${catalog.length} processed · Batch ${batchNumber} / ${batches.length}`;
      text(
        $("status"),
        `${progress()} · Classifying ${batch.length} photos together…`,
      );
      for (const item of batch) text(resultNode(item.id), "Classifying…");
      try {
        const photos = await Promise.all(
          batch.map(async (item) => ({
            id: item.id,
            image: await dataURL(item.src, signal),
          })),
        );
        signal.throwIfAborted();
        const response = await fetchWithRetry(
          base + "/classifier",
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(buildBatchRequest(model, photos)),
            credentials: "omit",
            redirect: "error",
            signal,
          },
          {
            onAttempt: () => requests++,
            onRetry: (retry, delay) =>
              text(
                $("status"),
                `${progress()} · Rate limited. Retrying in ${Math.ceil(delay / 1000)}s (${retry}/3)…`,
              ),
          },
        );
        const data = await response.json().catch(() => null);
        results.set(`batch_${batchNumber}`, {
          model,
          photos: batch.map((x) => x.id),
          status: response.status,
          response: data,
        });
        if (!response.ok)
          throw Error(
            response.status === 429
              ? "Rate limit persists after retries. Try again later."
              : data?.error?.message ||
                  (typeof data?.detail === "string"
                    ? data.detail
                    : `API error ${response.status}`),
          );
        signal.throwIfAborted();
        const answers = batch.map((item) =>
          answerFrom(data, `photo_${item.id}`),
        );
        batch.forEach((item, i) =>
          showAnswer(
            resultNode(item.id),
            answers[i],
            Math.round(performance.now() - started),
          ),
        );
        completed += batch.length;
      } catch (error) {
        if (signal.aborted) throw error;
        failed += batch.length;
        for (const item of batch) {
          resultNode(item.id).classList.add("error");
          text(resultNode(item.id), error.message);
        }
        if (!results.has(`batch_${batchNumber}`))
          results.set(`batch_${batchNumber}`, {
            photos: batch.map((x) => x.id),
            error: error.message,
          });
        // Stop issuing calls once the service has exhausted its retry allowance.
        if (error.message.includes("Rate limit persists")) throw error;
      } finally {
        $("responses").textContent = JSON.stringify(
          { requests, batches: Object.fromEntries(results) },
          null,
          2,
        );
      }
    }
    text(
      $("status"),
      `Done: ${completed} classified, ${failed} failed · ${requests} API request${requests === 1 ? "" : "s"}.`,
    );
  } catch (error) {
    text(
      $("status"),
      `${signal.aborted ? "Stopped." : error.message} ${completed} classified · ${requests} API requests.`,
    );
  } finally {
    for (const item of catalog) {
      const node = resultNode(item.id);
      if (["Queued…", "Classifying…"].includes(node.textContent))
        text(node, "Not processed — run Classify all to try again.");
    }
    busy = false;
    sync();
  }
});
$("cancel").addEventListener("click", () => controller?.abort());
$("model").addEventListener("change", () => {
  results.clear();
  for (const item of catalog) {
    resultNode(item.id).classList.remove("error");
    text(resultNode(item.id), "Ready to classify with the selected model.");
  }
  $("responses").textContent = "Run a classification to see the API responses.";
});
window.addEventListener("pagehide", () => controller?.abort());
try {
  const [photos, response] = await Promise.all([
    fetch("catalog.json").then((r) => {
      if (!r.ok) throw Error("Catalog unavailable");
      return r.json();
    }),
    fetch(base + "/models", {
      credentials: "omit",
      signal: AbortSignal.timeout(15000),
    }),
  ]);
  catalog = photos;
  render();
  if (!response.ok) throw Error("Could not load models. Reload to retry.");
  const data = await response.json();
  const ids = [
    ...new Set(
      (data.data || [])
        .map((x) => x.id)
        .filter((x) => typeof x === "string" && /gemma|qwen/i.test(x)),
    ),
  ];
  if (!ids.length) throw Error("No vision models available.");
  $("model").replaceChildren(...ids.map((id) => new Option(id, id)));
  if (ids.includes("featherless-ai/Qwen3.8-27B-classifier"))
    $("model").value = "featherless-ai/Qwen3.8-27B-classifier";
  ready = true;
  text($("status"), "Classify the full catalog, four photos per API request.");
  sync();
} catch (error) {
  text($("status"), error.message);
  sync();
}
