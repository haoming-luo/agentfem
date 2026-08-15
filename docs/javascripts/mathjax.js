window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"], ["$$", "$$"]],
    packages: { "[+]": ["boldsymbol"] },
    processEscapes: true,
    processEnvironments: true,
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex",
  },
};

(() => {
  let requestedRevision = 0;

  const root = document.documentElement;
  root.classList.add("af-math-pending");
  const delay = (milliseconds) =>
    new Promise((resolve) => window.setTimeout(resolve, milliseconds));
  const nextFrame = () => new Promise((resolve) => requestAnimationFrame(resolve));

  const waitForMathJax = async (revision) => {
    const deadline = performance.now() + 30000;
    while (performance.now() < deadline) {
      if (revision !== requestedRevision) {
        return undefined;
      }
      const mathJax = window.MathJax;
      if (mathJax && mathJax.startup && mathJax.startup.promise) {
        await Promise.race([mathJax.startup.promise, delay(1000)]);
      }
      if (mathJax && mathJax.typesetPromise) {
        return mathJax;
      }
      await delay(50);
    }
    return undefined;
  };

  const revealMath = (state) => {
    root.classList.remove("af-math-pending");
    root.classList.toggle("af-math-ready", state === "ready");
    root.classList.toggle("af-math-failed", state === "failed");
  };

  const typesetCurrentPage = async () => {
    const revision = ++requestedRevision;
    root.classList.add("af-math-pending");
    root.classList.remove("af-math-ready", "af-math-failed");

    /*
     * Material's instant navigation can publish a new document before the
     * deferred MathJax bundle has finished its first startup.  Wait against
     * elapsed time rather than animation-frame count: background tabs and a
     * cold browser cache can throttle frames heavily.  Process only formula
     * nodes that remain unrendered so repeated navigation cannot nest output.
     */
    const mathJax = await waitForMathJax(revision);
    if (revision !== requestedRevision) {
      return;
    }
    if (!mathJax) {
      revealMath("failed");
      throw new Error("MathJax did not become ready within 30 seconds");
    }

    const content = document.querySelector("[data-md-component='content']");
    if (!content) {
      revealMath("ready");
      return;
    }
    const pending = [...content.querySelectorAll(".arithmatex")].filter(
      (node) => !node.querySelector("mjx-container"),
    );
    if (pending.length > 0) {
      await mathJax.typesetPromise(pending);
    }

    /* CHTML inserts its font rules while typesetting.  Reveal equations only
     * after those same-origin fonts have settled, preventing partial glyphs
     * and late width changes on a first visit. */
    await nextFrame();
    if (document.fonts && document.fonts.ready) {
      await document.fonts.ready;
    }
    await nextFrame();
    if (revision === requestedRevision) {
      revealMath("ready");
    }
  };

  const requestTypeset = () => {
    typesetCurrentPage().catch((error) => {
      revealMath("failed");
      console.warn("AgentFEM documentation math rendering failed", error);
    });
  };

  if (typeof document$ !== "undefined") {
    document$.subscribe(requestTypeset);
  } else if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", requestTypeset, { once: true });
  } else {
    requestTypeset();
  }
})();
