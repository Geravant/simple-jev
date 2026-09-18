# Hot dog or sandwich? Vision demo

Static page at `/cool-demo/vision/`, linked from the homepage and Cool demos, with shared navigation. The catalog has 11 credited food photos, including hot dogs, sandwiches, burgers, and pizza. Classify all processes the catalog in order without manual selection.

Each batch sends up to four images together in one `/v1/classifier` request, with one keyed question per photo. Batches run serially with a 500 ms pause. HTTP 429 retries up to three times (1, 2, then 4 seconds minimum), honoring numeric or HTTP-date Retry-After. A cooldown longer than 60 seconds or exhausted retries stops the run; other batch errors are shown and the queue continues. Cancel aborts the request or backoff and preserves finished results. Requests already accepted by the server may still execute.

Photos share the API context budget within their batch. Raw responses, request-attempt count, progress, and per-photo errors are visible. No credentials or browser storage are used. Filenames, credits, and expected labels are never sent to the model.

Original photo URLs and credits are recorded in `catalog.json`, linked from every card, and used under https://unsplash.com/license. Images are stored locally at a maximum width of 640 pixels. Include this folder and `shared/` in deployment; no build is needed.

Run `node --test website/tests/vision.test.mjs website/tests/vision-queue.test.mjs`.
