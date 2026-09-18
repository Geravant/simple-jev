/* Copy exactly the visible example; no API calls or credentials on this page. */
document.querySelectorAll(".copy-code").forEach((button) => {
  button.addEventListener("click", async () => {
    const pre = button.closest(".docs-code").querySelector("pre");
    try {
      await navigator.clipboard.writeText(pre.textContent);
      button.textContent = "Copied";
      document.getElementById("docs-copy-status").textContent =
        "Example copied to clipboard.";
      setTimeout(() => (button.textContent = "Copy"), 1500);
    } catch {
      const range = document.createRange();
      range.selectNodeContents(pre);
      const selection = window.getSelection();
      selection.removeAllRanges();
      selection.addRange(range);
      pre.focus();
      document.getElementById("docs-copy-status").textContent =
        "Select and copy the highlighted example manually.";
    }
  });
});
