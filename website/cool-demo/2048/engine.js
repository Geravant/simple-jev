export const directions = ["UP", "DOWN", "LEFT", "RIGHT"];
export function slide(board, direction) {
  const next = board.slice(),
    transitions = [],
    merges = [];
  let score = 0;
  for (let lane = 0; lane < 4; lane++) {
    const ids = Array.from({ length: 4 }, (_, i) =>
      direction === "LEFT"
        ? lane * 4 + i
        : direction === "RIGHT"
          ? lane * 4 + 3 - i
          : direction === "UP"
            ? i * 4 + lane
            : (3 - i) * 4 + lane,
    );
    const sources = ids.filter((i) => board[i]),
      values = sources.map((i) => board[i]),
      merged = [];
    for (let i = 0; i < values.length; i++) {
      const to = ids[merged.length];
      transitions.push({ from: sources[i], to, value: values[i] });
      if (values[i] === values[i + 1]) {
        transitions.push({ from: sources[i + 1], to, value: values[i + 1] });
        merges.push(to);
        merged.push(values[i] * 2);
        score += values[i] * 2;
        i++;
      } else merged.push(values[i]);
    }
    ids.forEach((id, i) => (next[id] = merged[i] ?? 0));
  }
  return {
    board: next,
    score,
    transitions,
    merges,
    changed: next.some((v, i) => v !== board[i]),
  };
}
export function spawn(board, random = Math.random) {
  const empty = board.map((v, i) => (v ? null : i)).filter((i) => i !== null);
  if (empty.length)
    board[empty[Math.floor(random() * empty.length)]] = random() < 0.9 ? 2 : 4;
  return board;
}
