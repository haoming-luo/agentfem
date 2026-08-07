(() => {
  const storageKey = "agentfem.navigation.primary.scroll.v1";
  const sidebarSelector = ".md-sidebar--primary .md-sidebar__scrollwrap";
  let observedSidebar = null;
  let saveScheduled = false;

  const sidebar = () => document.querySelector(sidebarSelector);

  const storedPosition = () => {
    try {
      const value = Number.parseFloat(sessionStorage.getItem(storageKey));
      return Number.isFinite(value) ? value : null;
    } catch (_error) {
      return null;
    }
  };

  const savePosition = () => {
    const element = sidebar();
    if (!element) return;

    try {
      sessionStorage.setItem(storageKey, String(element.scrollTop));
    } catch (_error) {
      // Navigation should keep working when browser storage is unavailable.
    }
  };

  const restorePosition = () => {
    const element = sidebar();
    const position = storedPosition();
    if (!element || position === null) return;

    const apply = () => {
      const maximum = Math.max(0, element.scrollHeight - element.clientHeight);
      element.scrollTop = Math.min(position, maximum);
    };

    // Material may expand the active navigation branch after DOMContentLoaded.
    // Apply once immediately and once after its layout has settled.
    requestAnimationFrame(() => {
      apply();
      window.setTimeout(apply, 80);
    });
  };

  const bindSidebar = () => {
    restorePosition();

    const element = sidebar();
    if (!element || element === observedSidebar) return;
    observedSidebar = element;

    element.addEventListener(
      "scroll",
      () => {
        if (saveScheduled) return;
        saveScheduled = true;
        requestAnimationFrame(() => {
          saveScheduled = false;
          savePosition();
        });
      },
      { passive: true },
    );
  };

  document.addEventListener(
    "click",
    (event) => {
      if (event.target.closest(".md-sidebar--primary a.md-nav__link")) {
        savePosition();
      }
    },
    true,
  );
  window.addEventListener("pagehide", savePosition);

  if (typeof document$ !== "undefined") {
    document$.subscribe(bindSidebar);
  } else if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", bindSidebar, { once: true });
  } else {
    bindSidebar();
  }
})();
