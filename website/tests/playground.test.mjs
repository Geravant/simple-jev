import test from "node:test";
import assert from "node:assert/strict";
import {
  buildQuestions,
  validateResponse,
  QUESTION_EXAMPLES,
  SCENARIOS,
} from "../playground.mjs";

test("every scenario builds the documented question types", () => {
  for (const scenario of Object.values(SCENARIOS)) {
    const questions = buildQuestions(
      scenario.questions.map((key) => structuredClone(QUESTION_EXAMPLES[key])),
    );
    assert.equal(Object.keys(questions).length, scenario.questions.length);
  }
});
test("edits change the request and preserve choice descriptions containing separators", () => {
  const q = {
    id: "custom",
    type: "choice",
    instructions: "Pick one",
    criteriaText: "a | A | detailed\nb | B",
  };
  assert.deepEqual(buildQuestions([q]).custom.criteria, {
    a: "A | detailed",
    b: "B",
  });
  q.type = "score";
  q.criteriaText = "Poor\nOkay\nExcellent";
  assert.deepEqual(buildQuestions([q]).custom.criteria, [
    "Poor",
    "Okay",
    "Excellent",
  ]);
  q.type = "noul";
  assert.equal(Object.hasOwn(buildQuestions([q]).custom, "criteria"), false);
});
test("invalid and ambiguous drafts cannot be sent", () => {
  const good = structuredClone(QUESTION_EXAMPLES.routing);
  for (const rows of [
    [],
    [good, good],
    [{ ...good, id: " " }],
    [{ ...good, instructions: "" }],
    [{ ...good, criteriaText: "x\nx" }],
    [{ ...good, criteriaText: "only one" }],
    Array(7).fill(good),
  ])
    assert.throws(() => buildQuestions(rows));
});
test("arbitrary question and choice IDs remain data", () => {
  const result = buildQuestions([
    {
      id: "__proto__",
      type: "choice",
      instructions: "Pick",
      criteriaText: "__proto__\nconstructor",
    },
  ]);
  assert.equal(Object.hasOwn(result, "__proto__"), true);
  assert.deepEqual(Object.keys(result.__proto__.criteria), [
    "__proto__",
    "constructor",
  ]);
});
test("response validation follows edited IDs and rubric length", () => {
  const request = {
    questions: buildQuestions([
      {
        id: "rating",
        type: "score",
        instructions: "Quality?",
        criteriaText: "Bad\nGood",
      },
    ]),
  };
  const response = {
    answers: {
      rating: {
        type: "score",
        score: 0.6,
        confidence: 0.6,
        probabilities: { 0: 0.4, 1: 0.6 },
      },
    },
  };
  assert.doesNotThrow(() => validateResponse(response, request));
  response.answers.rating.score = 2;
  assert.throws(() => validateResponse(response, request));
  assert.throws(() => validateResponse({ answers: {} }, request));
});
