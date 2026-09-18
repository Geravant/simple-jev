# Simple Jev Pilot

A local, playable adaptation of [JevPilot by Standard Agents](https://github.com/standardagents/jevpilot). The original uses Jev by TypeSafe AI; this adaptation calls the public Featherless Simple Jev classifier directly from the browser.

## Play

Serve the repository's `website/` directory and open `/cool-demo/drive/`. The Cool demos page links here. WASD/arrow keys drive, Space brakes, J toggles autopilot, C changes camera, and P pauses. City, town, and highway worlds are available. The JSON button shows the exact compact API input, choices, usage, and telemetry.

Autopilot uses `featherless-ai/Qwen3.8-27B-classifier` at `https://simple-jev-demo-api.featherless.ai/v1/classifier`, without authentication. Calls are serial and separated by at least 300 ms; rate limits trigger backoff. The simulation keeps running during inference, following the last valid plan. Existing checks reject stale responses and brake when the applied decision is older than 1.8 seconds or a hazard requires it. Manual driving does not send API requests. The public API is shared and limited to 2k context / 4 RPS; errors are displayed, never replaced with fabricated model choices.

`src/simple-jev-api.js` creates compact candidate observations, reuses the original drive/stop and conditional path questions, resolves single-candidate questions locally, validates returned selections, and maps them back into simulation controls. Local routing retains the current route; the original model-based alternate-route question is omitted. Collision checks and the simulation safety brake remain local. Model choices can still be poor or crash the simulated car.

## Rebuild

From this folder:

```sh
npm ci
npm test
npm run build
```

Vite builds the static deployment into `website/cool-demo/drive/`. The rest of the website remains build-free. Dependencies are not committed. The copied original simulation tests and adapter tests run offline; real API smoke calls are separate.

## Provenance and assets

Source: https://github.com/standardagents/jevpilot (main, retrieved September 18, 2026). See `UPSTREAM_COMMIT` for the exact revision. No root code license file was present in that revision; no new license is asserted for upstream code. Retained model, texture, and decoder credits/license files live under `public/`, and attribution is also retained in the simulator's help dialog. New Simple Jev adapter and branding changes are documented above.

## Validation notes

Adapter tests pass. Real public API smoke calls for seed 42 in town, city, and highway returned valid controls at 524, 520, and 547 input tokens respectively. Browser verification confirmed rendering and moving autopilot with real responses. The upstream `random batches change while seeded runs remain reproducible` test fails its all-candidates-on-road assertion at line 89, identically in the untouched upstream checkout; this is not introduced by the API adapter. The broad upstream suite is slow and was interrupted; it is not claimed passing.

The focused collision, manual-driving, and adapter suites pass (13 tests). Three traffic-safety assertions also fail identically in the untouched upstream checkout, for four reproduced upstream failures total. The large world-generation suite was interrupted after about 83 seconds.
