export function chooseMove(probabilities, legal, random = Math.random) {
  const maximum = Math.max(...legal.map((d) => probabilities[d]));
  const tied = legal.filter((d) => Math.abs(probabilities[d] - maximum) < 1e-9);
  if (!tied.length) throw Error("No valid move scores");
  return {
    action: tied[Math.floor(random() * tied.length)],
    tied: tied.length > 1,
  };
}
