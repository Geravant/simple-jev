/* Small build-free highlighter. Token text is always inserted as text nodes,
 * including live API/user content; no HTML from examples is interpreted. */
(() => {
  const previous = new WeakMap();
  function highlight(node) {
    const source = node.textContent;
    if (previous.get(node) === source) return;
    previous.set(node, source);
    const fragment = document.createDocumentFragment();
    const tokens =
      /(^\s*##[^\n]*|"(?:\\.|[^"\\])*"|'[^'\n]*'|\b(?:true|false|null|None|const|await|import|from|try|except|if|throw|new)\b|\b\d+(?:\.\d+)?(?:e[+-]?\d+)?\b|\bcurl\b|--?[a-zA-Z][\w-]*|[{}\[\]:,])/gm;
    let end = 0;
    for (const match of source.matchAll(tokens)) {
      fragment.append(document.createTextNode(source.slice(end, match.index)));
      const span = document.createElement("span");
      const token = match[0];
      let kind = "punctuation";
      if (token.trimStart().startsWith("##")) kind = "comment";
      else if (/^["']/.test(token))
        kind = /^\s*:/.test(source.slice(match.index + token.length))
          ? "key"
          : "string";
      else if (/^\d/.test(token)) kind = "number";
      else if (/^[a-zA-Z-]/.test(token)) kind = "keyword";
      span.className = `syntax-${kind}`;
      span.textContent = token;
      fragment.append(span);
      end = match.index + token.length;
    }
    fragment.append(document.createTextNode(source.slice(end)));
    node.replaceChildren(fragment);
  }
  document
    .querySelectorAll("pre code, #request-code, #response-code")
    .forEach((node) => {
      const observer = new MutationObserver(() => {
        observer.disconnect();
        highlight(node);
        observer.observe(node, {
          childList: true,
          subtree: true,
          characterData: true,
        });
      });
      highlight(node);
      observer.observe(node, {
        childList: true,
        subtree: true,
        characterData: true,
      });
    });
})();
