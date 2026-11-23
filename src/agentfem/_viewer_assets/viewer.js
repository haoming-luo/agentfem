/* SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
 * SPDX-License-Identifier: Apache-2.0 */
// Keep the live websocket document in place when downloading an export.
window.AgentFEMViewer = {
  install(app) {
    app.config.globalProperties.$viewer = {
      async download(relative, filename, mime) {
        const url = new URL(relative, window.location.href);
        if (url.origin !== window.location.origin || !url.pathname.startsWith('/viewer-exports/')) {
          throw new Error('Select a completed local export.');
        }
        const response = await window.fetch(url);
        if (!response.ok) throw new Error('Export is unavailable. Please export again.');
        const bytes = await response.blob();
        const objectUrl = URL.createObjectURL(new Blob([bytes], { type: mime }));
        const anchor = document.createElement('a');
        anchor.href = objectUrl;
        anchor.download = filename;
        document.body.appendChild(anchor);
        anchor.click();
        anchor.remove();
        window.setTimeout(() => URL.revokeObjectURL(objectUrl), 60000);
      },
    };
  },
};
