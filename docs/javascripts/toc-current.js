(() => {
  const tocLinkSelector =
    ".md-sidebar--secondary a.md-nav__link[href*='#']";
  const currentClass = "af-toc-current";
  let updateFrame = null;

  const decodedHash = (link) => {
    try {
      return decodeURIComponent(new URL(link.href, document.baseURI).hash.slice(1));
    } catch (_error) {
      return "";
    }
  };

  const updateCurrentSection = () => {
    updateFrame = null;
    const links = Array.from(document.querySelectorAll(tocLinkSelector));
    if (links.length === 0) return;

    const linksById = new Map();
    for (const link of links) {
      const id = decodedHash(link);
      if (!id) continue;
      const matching = linksById.get(id) || [];
      matching.push(link);
      linksById.set(id, matching);
    }

    const targets = Array.from(document.querySelectorAll(".md-typeset [id]"))
      .filter((element) => linksById.has(element.id));
    const headerBottom =
      document.querySelector(".md-header")?.getBoundingClientRect().bottom || 0;
    const threshold = headerBottom + 12;
    let current = null;

    for (const target of targets) {
      if (target.getBoundingClientRect().top <= threshold) current = target;
      else break;
    }

    for (const link of links) link.classList.remove(currentClass);
    if (!current) return;
    for (const link of linksById.get(current.id) || []) {
      link.classList.add(currentClass);
    }
  };

  const scheduleUpdate = () => {
    if (updateFrame !== null) return;
    updateFrame = requestAnimationFrame(updateCurrentSection);
  };

  window.addEventListener("scroll", scheduleUpdate, { passive: true });
  window.addEventListener("resize", scheduleUpdate, { passive: true });
  window.addEventListener("hashchange", scheduleUpdate);

  const bindPage = () => {
    scheduleUpdate();
    window.setTimeout(scheduleUpdate, 100);
  };

  if (typeof document$ !== "undefined") {
    document$.subscribe(bindPage);
  } else if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", bindPage, { once: true });
  } else {
    bindPage();
  }
})();
