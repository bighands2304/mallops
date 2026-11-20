// Click (or Enter) on a Mermaid diagram opens it full screen.
// Wheel / pinch to zoom, drag to pan, Esc or click outside to close.
(() => {
  const dialog = document.createElement("dialog");
  dialog.className = "mermaid-zoom";
  dialog.setAttribute("aria-label", "Diagram viewer");
  dialog.innerHTML =
    '<button class="mermaid-zoom__close" aria-label="Close">&times;</button>' +
    '<div class="mermaid-zoom__stage"></div>';
  document.body.append(dialog);

  // Own shadow root: the cloned SVG keeps Material's styles and its ids don't clash.
  const stage = dialog.querySelector(".mermaid-zoom__stage").attachShadow({ mode: "open" });
  let panZoom = null;

  dialog.querySelector("button").addEventListener("click", () => dialog.close());
  dialog.addEventListener("click", (e) => { if (e.target === dialog) dialog.close(); });
  dialog.addEventListener("close", () => {
    panZoom?.destroy();
    panZoom = null;
    stage.replaceChildren();
  });

  const open = (host) => {
    if (!host.shadowRoot) return;
    stage.replaceChildren(...[...host.shadowRoot.children].map((n) => n.cloneNode(true)));
    const svg = stage.querySelector("svg");
    if (!svg) return;
    svg.removeAttribute("style");
    svg.setAttribute("width", "100%");
    svg.setAttribute("height", "100%");
    dialog.showModal();
    panZoom = svgPanZoom(svg, {
      controlIconsEnabled: true,
      fit: true,
      center: true,
      minZoom: 0.5,
      maxZoom: 20,
    });
  };

  // Shadow DOM retargets events to the host, so e.target is the diagram container.
  document.addEventListener("click", (e) => {
    const host = e.target.closest?.("div.mermaid");
    if (host) open(host);
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && e.target.matches?.("div.mermaid")) open(e.target);
  });

  // Diagrams render asynchronously: make them focusable as they appear.
  new MutationObserver(() => {
    for (const el of document.querySelectorAll("div.mermaid:not([tabindex])")) {
      el.tabIndex = 0;
      el.setAttribute("role", "button");
      el.setAttribute("aria-label", "Open diagram full screen");
      el.title = "Click to zoom";
    }
  }).observe(document.body, { childList: true, subtree: true });
})();
