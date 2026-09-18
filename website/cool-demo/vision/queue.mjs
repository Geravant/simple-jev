// Shared, cancelable pacing and bounded rate-limit retries. Other errors are
// returned to the caller; we never silently substitute model results.
export function wait(ms, signal) {
  signal?.throwIfAborted();
  return new Promise((resolve, reject) => {
    const abort = () => {
      clearTimeout(timer);
      reject(signal.reason);
    };
    const timer = setTimeout(() => {
      signal?.removeEventListener("abort", abort);
      resolve();
    }, ms);
    signal?.addEventListener("abort", abort, { once: true });
  });
}
export function retryDelay(value, attempt, now = Date.now()) {
  const seconds = value?.trim() ? Number(value) : NaN;
  const date = Number.isFinite(seconds) ? NaN : Date.parse(value);
  const requested = Number.isFinite(seconds)
    ? seconds * 1000
    : Number.isFinite(date)
      ? date - now
      : 0;
  return Math.max(1000 * 2 ** attempt, requested);
}
export async function fetchWithRetry(
  url,
  options,
  {
    fetcher = fetch,
    sleep = wait,
    onRetry = () => {},
    onAttempt = () => {},
  } = {},
) {
  for (let attempt = 0; attempt <= 3; attempt++) {
    options.signal?.throwIfAborted();
    onAttempt();
    const response = await fetcher(url, {
      ...options,
      signal: AbortSignal.any([
        options.signal ?? new AbortController().signal,
        AbortSignal.timeout(60000),
      ]),
    });
    if (response.status !== 429 || attempt === 3) return response;
    const delay = retryDelay(response.headers.get("Retry-After"), attempt);
    // Very long server cooldowns are surfaced instead of tying up the page.
    if (delay > 60000) return response;
    await response.body?.cancel();
    onRetry(attempt + 1, delay);
    await sleep(delay, options.signal);
  }
}
