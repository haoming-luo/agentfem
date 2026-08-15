window.MathJax = {
  loader: {
    load: ["[tex]/boldsymbol"],
  },
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

  const nextFrame = () =>
    new Promise((resolve) => {
      requestAnimationFrame(resolve);
    });

  const waitForMathJax = async (revision) => {
    for (let attempt = 0; attempt < 120; attempt += 1) {
      if (revision !== requestedRevision) {
        return undefined;
      }
      const mathJax = window.MathJax;
      if (mathJax && mathJax.startup && mathJax.startup.promise) {
        await mathJax.startup.promise;
      }
      if (mathJax && mathJax.typesetPromise) {
        return mathJax;
      }
      await nextFrame();
    }
    return undefined;
  };

  const typesetCurrentPage = async () => {
    const revision = ++requestedRevision;

    /*
     * Material's instant navigation can publish a new document before the
     * deferred MathJax bundle has finished its first startup.  Wait for the
     * live MathJax object instead of capturing its earlier configuration
     * object, then process only formula nodes that remain unrendered.
     */
    const mathJax = await waitForMathJax(revision);
    await nextFrame();
    await nextFrame();
    if (revision !== requestedRevision || !mathJax) {
      return;
    }

    const content = document.querySelector("[data-md-component='content']");
    if (!content) {
      return;
    }
    const pending = [...content.querySelectorAll(".arithmatex")].filter(
      (node) => !node.querySelector("mjx-container"),
    );
    if (pending.length > 0) {
      await mathJax.typesetPromise(pending);
    }
  };

  const requestTypeset = () => {
    typesetCurrentPage().catch((error) => {
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
