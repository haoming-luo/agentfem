"""Render an AgentFEM failure-aware report from official benchmark output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentfem.integrations.pdeagent_bench.report import read_official_summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    parser.add_argument(
        "--case-catalog",
        type=Path,
        help=(
            "Optional public agent-view JSONL or official result-family directory "
            "used to stratify results by dimension."
        ),
    )
    parser.add_argument("--json", dest="json_path", type=Path)
    parser.add_argument("--markdown", dest="markdown_path", type=Path)
    args = parser.parse_args()
    report = read_official_summary(args.summary, case_catalog=args.case_catalog)
    if args.json_path:
        args.json_path.write_text(json.dumps(report.as_dict(), indent=2) + "\n")
    if args.markdown_path:
        args.markdown_path.write_text(report.markdown())
    if not args.json_path and not args.markdown_path:
        print(report.markdown(), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
