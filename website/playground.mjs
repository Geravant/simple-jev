/* Pure request-building helpers, shared by the UI and offline contract tests. */
export const QUESTION_EXAMPLES = {
  routing: {
    id: "team",
    type: "choice",
    instructions: "Which team should handle this customer message?",
    criteriaText:
      "billing | Payments, invoices, and refunds\ntechnical | Bugs, outages, and product errors\naccount | Profile, login, and subscription changes",
  },
  urgency: {
    id: "urgency",
    type: "score",
    instructions: "How urgently does this message need attention?",
    criteriaText:
      "Routine: no blocked work\nImportant: use is affected but a workaround exists\nCritical: work is blocked or there is an outage",
  },
  refund: {
    id: "refund",
    type: "noul",
    instructions: "Does the customer explicitly ask for money back?",
    criteriaText: "",
  },
  sentiment: {
    id: "sentiment",
    type: "choice",
    instructions: "What is the overall sentiment?",
    criteriaText:
      "positive | Mostly favorable\nneutral | Mixed or factual\nnegative | Mostly unfavorable",
  },
  quality: {
    id: "quality",
    type: "score",
    instructions: "How useful is this review to a potential buyer?",
    criteriaText:
      "Not useful: no product details\nSomewhat useful: a general opinion\nVery useful: specific experiences and tradeoffs",
  },
  spam: {
    id: "spam",
    type: "noul",
    instructions: "Is this message unsolicited promotional spam?",
    criteriaText: "",
  },
  topic: {
    id: "topic",
    type: "choice",
    instructions: "Which category best describes this post?",
    criteriaText:
      "discussion | A genuine question or discussion\npromotion | Advertising or self-promotion\nother | Neither of the above",
  },
  custom_choice: {
    id: "question",
    type: "choice",
    instructions: "",
    criteriaText: "option_a | First option\noption_b | Second option",
  },
  custom_score: {
    id: "question",
    type: "score",
    instructions: "",
    criteriaText: "Low\nMedium\nHigh",
  },
  custom_noul: {
    id: "question",
    type: "noul",
    instructions: "",
    criteriaText: "",
  },
};
export const SCENARIOS = {
  support: {
    questions: ["routing", "urgency", "refund"],
    examples: [
      [
        "Double charge",
        "Hey, I was charged twice for my Pro subscription this month. Could you refund the extra payment? Everything else is working great.",
      ],
      [
        "App outage",
        "Our entire team is getting a server error when we open the app. Nobody can access their work, and we have a client deadline in an hour. Please help!",
      ],
      [
        "Account update",
        "How can I change the name displayed on my profile? No rush, and I am happy with my subscription.",
      ],
    ],
  },
  reviews: {
    questions: ["sentiment", "quality"],
    examples: [
      [
        "Mixed review",
        "The headphones sound fantastic and last two days on a charge. But the ear cups get uncomfortable after an hour, and the microphone struggles outside.",
      ],
      [
        "Happy customer",
        "I have used this backpack daily for six months. The zippers still work perfectly, my 16-inch laptop fits, and the waterproof fabric handled a heavy downpour.",
      ],
      ["Vague review", "It is okay, I guess."],
    ],
  },
  moderation: {
    questions: ["topic", "spam"],
    examples: [
      [
        "Community question",
        "Has anyone tried growing tomatoes on a north-facing balcony? I get about four hours of sunlight. Looking for practical advice.",
      ],
      [
        "Promotional post",
        "LIMITED OFFER! Buy followers now and triple your reach overnight. Visit our store today for 80% off.",
      ],
      [
        "Useful recommendation",
        "Someone asked about beginner gardening books yesterday. I borrowed The Vegetable Garden from the library and found the planting calendar helpful.",
      ],
    ],
  },
};
export const MAX_QUESTIONS = 6;

export function buildQuestions(drafts) {
  if (!drafts.length || drafts.length > MAX_QUESTIONS)
    throw new Error(`Add between 1 and ${MAX_QUESTIONS} questions.`);
  const entries = [];
  const ids = new Set();
  for (const [index, draft] of drafts.entries()) {
    const id = draft.id.trim();
    const name = `Question ${index + 1}`;
    if (!id) throw new Error(`${name}: give it an answer ID.`);
    if (ids.has(id))
      throw new Error(`${name}: the answer ID “${id}” is already used.`);
    ids.add(id);
    if (!draft.instructions.trim())
      throw new Error(`${name}: enter the question you want to ask.`);
    if (!["choice", "score", "noul"].includes(draft.type))
      throw new Error(`${name}: select a supported question type.`);
    const question = {
      type: draft.type,
      instructions: draft.instructions.trim(),
    };
    if (draft.type !== "noul") {
      const lines = draft.criteriaText
        .split("\n")
        .map((line) => line.trim())
        .filter(Boolean);
      if (lines.length < 2 || lines.length > 50)
        throw new Error(
          `${name}: enter 2–50 ${draft.type === "choice" ? "choices" : "rubric levels"}, one per line.`,
        );
      if (draft.type === "score") question.criteria = lines;
      else {
        const options = lines.map((line) => {
          const separator = line.indexOf("|");
          return separator < 0
            ? [line, null]
            : [
                line.slice(0, separator).trim(),
                line.slice(separator + 1).trim() || null,
              ];
        });
        if (
          options.some(([key]) => !key) ||
          new Set(options.map(([key]) => key)).size !== options.length
        )
          throw new Error(`${name}: choice IDs must be nonempty and unique.`);
        question.criteria = Object.fromEntries(options);
      }
    }
    entries.push([id, question]);
  }
  // Object.fromEntries also handles names such as __proto__ as data, not setters.
  return Object.fromEntries(entries);
}

export function validateResponse(data, request) {
  const finite = (v, min, max) =>
    typeof v === "number" && Number.isFinite(v) && v >= min && v <= max;
  for (const [id, question] of Object.entries(request.questions)) {
    const a = data?.answers?.[id];
    let valid = a?.type === question.type;
    if (question.type === "choice")
      valid &&=
        Object.hasOwn(question.criteria, a?.choice) &&
        finite(a?.confidence, 0, 1) &&
        Object.keys(question.criteria).every((key) =>
          finite(a?.probabilities?.[key], 0, 1),
        );
    if (question.type === "score")
      valid &&=
        finite(a?.score, 0, question.criteria.length - 1) &&
        finite(a?.confidence, 0, 1) &&
        question.criteria.every((_, i) =>
          finite(a?.probabilities?.[String(i)], 0, 1),
        );
    if (question.type === "noul") valid &&= finite(a?.noul, 0.01, 0.99);
    if (!valid)
      throw new Error(
        `The API returned an unexpected answer for “${id}”. Inspect the raw response under “Under the hood”.`,
      );
  }
}
