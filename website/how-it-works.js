const a = document.querySelector("#logit-a"),
  b = document.querySelector("#logit-b"),
  result = document.querySelector("#logit-result");
function update() {
  const x = Number(a.value),
    y = Number(b.value),
    p = 1 / (1 + Math.exp(y - x));
  result.textContent = `Logits: A = ${x.toFixed(1)}, B = ${y.toFixed(1)} → red ${(p * 100).toFixed(1)}% · blue ${((1 - p) * 100).toFixed(1)}%`;
}
a.addEventListener("input", update);
b.addEventListener("input", update);
update();
