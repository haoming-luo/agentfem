from __future__ import annotations

import html
import re
import shutil
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SITE_DIR = ROOT / "site"
LOGO_FILES = (
    "AgentFEM_logo.png",
    "AgentFEM_logo_transparent.png",
)


@dataclass(frozen=True)
class Page:
    title: str
    source: Path
    output: Path


PAGES = [
    Page("Home", ROOT / "README.md", SITE_DIR / "index.html"),
    Page("Install", ROOT / "INSTALL.md", SITE_DIR / "install.html"),
    Page("Agent Guide", ROOT / "AGENT_GUIDE.md", SITE_DIR / "agent-guide.html"),
    Page("Workflow", ROOT / "WORKFLOW.md", SITE_DIR / "workflow.html"),
    Page("Concepts", ROOT / "CONCEPTS.md", SITE_DIR / "concepts.html"),
    Page("Examples", ROOT / "examples" / "README.md", SITE_DIR / "examples.html"),
    Page(
        "Product Roadmap",
        ROOT / "docs" / "product_roadmap.md",
        SITE_DIR / "product-roadmap.html",
    ),
    Page(
        "Nonlinear Materials",
        ROOT / "docs" / "nonlinear_materials.md",
        SITE_DIR / "nonlinear-materials.html",
    ),
    Page(
        "Nonlinear Solid Architecture",
        ROOT / "docs" / "nonlinear_solid_architecture.md",
        SITE_DIR / "nonlinear-solid-architecture.html",
    ),
    Page(
        "Procedures, Thermal Stress, and Creep",
        ROOT / "docs" / "solution_procedures_and_thermal_creep.md",
        SITE_DIR / "solution-procedures-and-thermal-creep.html",
    ),
    Page(
        "Scientific Function Reference",
        ROOT / "docs" / "reference" / "scientific_function_reference.md",
        SITE_DIR / "scientific-function-reference.html",
    ),
    Page(
        "Results and Campaigns",
        ROOT / "docs" / "results_and_campaigns.md",
        SITE_DIR / "results-and-campaigns.html",
    ),
    Page(
        "Stable Steps and Compact Output",
        ROOT / "docs" / "step_and_output_architecture.md",
        SITE_DIR / "step-and-output-architecture.html",
    ),
    Page(
        "Mesh Interoperability",
        ROOT / "docs" / "mesh_interoperability.md",
        SITE_DIR / "mesh-interoperability.html",
    ),
    Page(
        "Abaqus Periodic Cell",
        ROOT / "docs" / "abaqus_periodic_cell.md",
        SITE_DIR / "abaqus-periodic-cell.html",
    ),
    Page(
        "Abaqus User Materials",
        ROOT / "docs" / "abaqus_user_material_bridge.md",
        SITE_DIR / "abaqus-user-material-bridge.html",
    ),
    Page("Module Map", ROOT / "docs" / "module_map.md", SITE_DIR / "module-map.html"),
    Page("API Style", ROOT / "docs" / "api_style.md", SITE_DIR / "api-style.html"),
    Page(
        "Extension Rules",
        ROOT / "docs" / "extension_rules.md",
        SITE_DIR / "extension-rules.html",
    ),
    Page(
        "Tutorial Design",
        ROOT / "docs" / "tutorial_design.md",
        SITE_DIR / "tutorial-design.html",
    ),
    Page("Validation", ROOT / "docs" / "validation.md", SITE_DIR / "validation.html"),
    Page("Licensing", ROOT / "docs" / "licensing.md", SITE_DIR / "licensing.html"),
    Page("Publishing", ROOT / "docs" / "publishing.md", SITE_DIR / "publishing.html"),
    Page(
        "26 August 2026 Release Gate",
        ROOT / "docs" / "release_2026_08_26.md",
        SITE_DIR / "release-2026-08-26.html",
    ),
    Page(
        "Architecture Review",
        ROOT / "docs" / "architecture_review.md",
        SITE_DIR / "architecture-review.html",
    ),
    Page(
        "AIR Architecture Roadmap",
        ROOT / "docs" / "air_architecture_roadmap.md",
        SITE_DIR / "air-architecture-roadmap.html",
    ),
    Page(
        "AI-Native Campaigns and Learning",
        ROOT / "docs" / "ai_native_learning.md",
        SITE_DIR / "ai-native-learning.html",
    ),
    Page(
        "Documentation Site",
        ROOT / "docs" / "documentation_site.md",
        SITE_DIR / "documentation-site.html",
    ),
    Page(
        "AgentFEM Skill",
        ROOT / "skills" / "agentfem" / "SKILL.md",
        SITE_DIR / "skill.html",
    ),
]


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9\s-]", "", text).strip().lower()
    slug = re.sub(r"[\s-]+", "-", slug)
    return slug or "section"


def inline_markdown(text: str) -> str:
    text = html.escape(text)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', text)
    return text


def markdown_to_html(markdown: str) -> str:
    lines = markdown.splitlines()
    out: list[str] = []
    in_code = False
    list_tag: str | None = None
    in_table = False
    code_lines: list[str] = []
    table_rows: list[list[str]] = []
    paragraph: list[str] = []

    def flush_paragraph() -> None:
        nonlocal paragraph
        if paragraph:
            out.append(f"<p>{inline_markdown(' '.join(paragraph))}</p>")
            paragraph = []

    def flush_list() -> None:
        nonlocal list_tag
        if list_tag:
            out.append(f"</{list_tag}>")
            list_tag = None

    def flush_table() -> None:
        nonlocal in_table, table_rows
        if not in_table:
            return
        if table_rows:
            header = table_rows[0]
            body = table_rows[2:] if len(table_rows) > 1 else []
            out.append("<table>")
            out.append(
                "<thead><tr>"
                + "".join(f"<th>{inline_markdown(cell)}</th>" for cell in header)
                + "</tr></thead>"
            )
            if body:
                out.append("<tbody>")
                for row in body:
                    out.append(
                        "<tr>"
                        + "".join(f"<td>{inline_markdown(cell)}</td>" for cell in row)
                        + "</tr>"
                    )
                out.append("</tbody>")
            out.append("</table>")
        table_rows = []
        in_table = False

    for raw in lines:
        line = raw.rstrip()
        if line.startswith("```"):
            flush_paragraph()
            flush_list()
            flush_table()
            if in_code:
                out.append(f"<pre><code>{html.escape(chr(10).join(code_lines))}</code></pre>")
                code_lines = []
                in_code = False
            else:
                in_code = True
            continue

        if in_code:
            code_lines.append(line)
            continue

        if not line:
            flush_paragraph()
            flush_list()
            flush_table()
            continue

        if (
            line.startswith("<p")
            and line.endswith("</p>")
            and "<img " in line
        ):
            flush_paragraph()
            flush_list()
            flush_table()
            out.append(line)
            continue

        heading = re.match(r"^(#{1,4})\s+(.*)$", line)
        if heading:
            flush_paragraph()
            flush_list()
            flush_table()
            level = len(heading.group(1))
            title = heading.group(2).strip()
            out.append(
                f'<h{level} id="{slugify(title)}">{inline_markdown(title)}</h{level}>'
            )
            continue

        if line.startswith("|") and line.endswith("|"):
            flush_paragraph()
            flush_list()
            in_table = True
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            table_rows.append(cells)
            continue

        ordered = re.match(r"^\d+\.\s+(.*)$", line)
        unordered = re.match(r"^-\s+(.*)$", line)
        if ordered or unordered:
            flush_paragraph()
            flush_table()
            current_tag = "ol" if ordered else "ul"
            if list_tag != current_tag:
                flush_list()
                out.append(f"<{current_tag}>")
                list_tag = current_tag
            item = ordered.group(1) if ordered else unordered.group(1)
            out.append(f"<li>{inline_markdown(item.strip())}</li>")
            continue

        if list_tag and raw.startswith(("  ", "   ")) and out and out[-1].endswith("</li>"):
            out[-1] = out[-1][:-5] + f"<br>{inline_markdown(line.strip())}</li>"
            continue

        paragraph.append(line.strip())

    flush_paragraph()
    flush_list()
    flush_table()
    return "\n".join(out)


STYLE = """
:root {
  color-scheme: light;
  --bg: #f7f8fa;
  --panel: #ffffff;
  --text: #1d232a;
  --muted: #65717f;
  --line: #d9dee5;
  --brand: #2563eb;
  --code: #f0f3f7;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  font: 16px/1.65 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  background: var(--bg);
  color: var(--text);
}
.layout {
  display: grid;
  grid-template-columns: 280px minmax(0, 1fr);
  min-height: 100vh;
}
aside {
  border-right: 1px solid var(--line);
  background: var(--panel);
  padding: 28px 22px;
  position: sticky;
  top: 0;
  height: 100vh;
  overflow: auto;
}
.brand {
  display: block;
  margin-bottom: 4px;
}
.brand img {
  display: block;
  height: auto;
  max-width: 100%;
  object-fit: contain;
  width: 190px;
}
.project-logo {
  margin: 0 auto 24px;
  text-align: center;
}
.project-logo img,
article > p[align="center"] img {
  height: auto;
  max-width: min(100%, 320px);
}
.tagline {
  color: var(--muted);
  font-size: 13px;
  margin-bottom: 24px;
}
nav a {
  display: block;
  color: var(--text);
  text-decoration: none;
  padding: 7px 8px;
  border-radius: 6px;
  font-size: 14px;
}
nav a.active,
nav a:hover {
  background: #eaf1ff;
  color: var(--brand);
}
main {
  max-width: 980px;
  width: 100%;
  padding: 44px 56px 80px;
}
article {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 36px 42px;
}
h1, h2, h3, h4 {
  line-height: 1.25;
  margin: 1.4em 0 0.55em;
}
h1 { margin-top: 0; font-size: 36px; }
h2 { font-size: 25px; border-top: 1px solid var(--line); padding-top: 24px; }
h3 { font-size: 19px; }
p, ul, table, pre { margin: 0 0 16px; }
code {
  background: var(--code);
  padding: 0.14em 0.34em;
  border-radius: 4px;
  font-size: 0.92em;
}
pre {
  overflow: auto;
  background: #111827;
  color: #f9fafb;
  padding: 16px;
  border-radius: 8px;
}
pre code {
  background: transparent;
  padding: 0;
  color: inherit;
}
table {
  width: 100%;
  border-collapse: collapse;
}
th, td {
  border: 1px solid var(--line);
  padding: 9px 11px;
  text-align: left;
  vertical-align: top;
}
th { background: #f3f6fa; }
a { color: var(--brand); }
@media (max-width: 780px) {
  .layout { display: block; }
  aside {
    position: static;
    height: auto;
    border-right: 0;
    border-bottom: 1px solid var(--line);
  }
  main { padding: 22px 16px 48px; }
  article { padding: 24px 20px; }
}
"""


def render_page(page: Page) -> str:
    nav = "\n".join(
        f'<a class="{"active" if item.output.name == page.output.name else ""}" '
        f'href="{item.output.name}">{html.escape(item.title)}</a>'
        for item in PAGES
    )
    content = markdown_to_html(page.source.read_text())
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(page.title)} - AgentFEM</title>
  <link rel="stylesheet" href="assets/style.css">
</head>
<body>
  <div class="layout">
    <aside>
      <a class="brand" href="index.html">
        <img src="logo/AgentFEM_logo_transparent.png" alt="AgentFEM">
      </a>
      <div class="tagline">AI-assisted finite-element workflows</div>
      <nav>{nav}</nav>
    </aside>
    <main>
      <article>
{content}
      </article>
    </main>
  </div>
</body>
</html>
"""


def main() -> None:
    (SITE_DIR / "assets").mkdir(parents=True, exist_ok=True)
    (SITE_DIR / "logo").mkdir(parents=True, exist_ok=True)
    (SITE_DIR / "assets" / "style.css").write_text(STYLE.strip() + "\n")
    for filename in LOGO_FILES:
        shutil.copy2(ROOT / "logo" / filename, SITE_DIR / "logo" / filename)
    for page in PAGES:
        page.output.write_text(render_page(page))
    print(f"Built AgentFEM documentation site: {SITE_DIR / 'index.html'}")


if __name__ == "__main__":
    main()
