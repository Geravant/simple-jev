# Simple Jev 2048

Static adaptation of the supplied `proj-classifier/public-2048` reference. Open `/cool-demo/2048/` from the website server; there is no build step or proxy.

- Arrow keys, swipe gestures, or direction buttons play manually.
- One AI move calls the public classifier once; Start classifier continues until paused, an error, a win, or game over.
- Manual input and New game invalidate pending model decisions. Requests already sent may finish, but their moves are ignored.
- Models come from the public demo catalog. Text grid is the default; image mode sends the rendered board PNG and is limited to Gemma/Qwen.
- Only legal moves are candidates; a sole legal move resolves locally without a request. Equal highest probabilities use a random tie break.
- Requests are serial, with a 250 ms pause between automatic turns plus API/animation time. API errors stop automatic play.
- Usage and recent response metadata stay in memory for the current page session. No key, cookies, or browser storage is used.

Run rules/adapter checks from the repository root: `node --test website/tests/2048.test.mjs`.
