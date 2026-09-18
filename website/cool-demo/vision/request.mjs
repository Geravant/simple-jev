export const categories = {
  hot_dog:
    "A sausage served lengthwise in a split bun. Treat hot dogs as a separate category, not as sandwiches.",
  sandwich:
    "Filling between bread slices or in a bread roll, including burgers, but excluding hot dogs.",
  neither:
    "Neither a hot dog nor a sandwich, or the image is too unclear to identify.",
};
export function buildRequest(model, image) {
  if (!/gemma|qwen/i.test(model))
    throw Error("Choose a Gemma or Qwen vision model.");
  if (!/^data:image\/(jpeg|png|webp);base64,/.test(image))
    throw Error("A supported image is required.");
  return {
    model,
    messages: [
      {
        role: "user",
        content: [
          {
            type: "text",
            text: "Classify the main food in this photograph based on what you see.",
          },
          { type: "image_url", image_url: { url: image } },
        ],
      },
    ],
    questions: {
      food: {
        type: "choice",
        instructions:
          "Which category best describes the main food? Use the supplied definitions; hot dogs are separate from sandwiches.",
        criteria: categories,
      },
    },
  };
}
export function answerFrom(data, id = "food") {
  const a = data?.answers?.[id];
  if (
    !a ||
    !Object.hasOwn(categories, a.choice) ||
    !Object.keys(categories).every(
      (k) =>
        Number.isFinite(a.probabilities?.[k]) &&
        a.probabilities[k] >= 0 &&
        a.probabilities[k] <= 1,
    )
  )
    throw Error("The API returned an invalid classification.");
  return a;
}

// One shared multimodal context, with one keyed question per image.
export function buildBatchRequest(model, photos) {
  if (!photos.length) throw Error("Select at least one photo.");
  const ids = new Set();
  const content = [],
    questions = {};
  for (const { id, image } of photos) {
    if (!/^[a-zA-Z0-9_-]+$/.test(id) || ids.has(id))
      throw Error("Photo IDs must be unique.");
    ids.add(id);
    buildRequest(model, image); // Reuse model and image validation.
    content.push(
      { type: "text", text: `Photo ${id}:` },
      { type: "image_url", image_url: { url: image } },
    );
    questions[`photo_${id}`] = {
      type: "choice",
      instructions: `Classify only Photo ${id}. Hot dogs are separate from sandwiches.`,
      criteria: categories,
    };
  }
  return { model, messages: [{ role: "user", content }], questions };
}
