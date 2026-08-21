"""Render an AgentFEM failure-aware report from official benchmark output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentfem.integrations.pdeagent_bench.report import (
    combine_official_summaries,
    read_official_summary,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path, nargs="+")
    parser.add_argument(
        "--case-catalog",
        type=Path,
        action="append",
        help=(
            "Optional public agent-view JSONL or official result-family directory "
            "used to stratify results by dimension."
        ),
    )
    parser.add_argument("--json", dest="json_path", type=Path)
    parser.add_argument("--markdown", dest="markdown_path", type=Path)
    args = parser.parse_args()
    if len(args.summary) == 1:
        catalog = None if not args.case_catalog else args.case_catalog[0]
        report = read_official_summary(args.summary[0], case_catalog=catalog)
    else:
        report = combine_official_summaries(
            args.summary,
            case_catalogs=tuple(args.case_catalog or ()),
        )
    if args.json_path:
        args.json_path.write_text(json.dumps(report.as_dict(), indent=2) + "\n")
    if args.markdown_path:
        args.markdown_path.write_text(report.markdown())
    if not args.json_path and not args.markdown_path:
        print(report.markdown(), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
