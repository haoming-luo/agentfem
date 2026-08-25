"""Prepare an immutable, reviewable fresh-agent acceptance task.

This coordinator does not run or impersonate the AI agent. It copies the
exact wheel candidate, records its source identity, and writes one bounded
scientific task plus the review instructions needed after a genuinely fresh
agent has completed the work.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def prepare(
    *,
    wheel: Path,
    output: Path,
    source_commit: str,
    agentfem_version: str,
) -> dict[str, object]:
    wheel_path = Path(wheel).resolve()
    if not wheel_path.is_file():
        raise FileNotFoundError(wheel_path)
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    candidate = root / wheel_path.name
    shutil.copy2(wheel_path, candidate)
    project = root / "project"
    project.mkdir()

    contract = {
        "schema": "agentfem.agent-trial-contract",
        "schema_version": "0.1.0",
        "agentfem_version": str(agentfem_version),
        "source_commit": str(source_commit),
        "wheel": candidate.name,
        "wheel_sha256": _sha256(candidate),
        "fresh_context_required": True,
        "human_interventions_allowed": 0,
        "project_directory": "project",
        "required_outputs": [
            "project/agentfem.toml",
            "project/case.py",
            "project/result.json",
            "project/explanation.md",
            "agent-transcript.md",
        ],
    }
    (root / "trial-contract.json").write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (root / "TASK.md").write_text(
        """# Fresh AgentFEM task

Work in the empty `project/` directory using only the supplied AgentFEM wheel
and public AgentFEM guidance. Start by inspecting the runtime and installed
capabilities.

Create, run, check and verify a two-dimensional plane-strain linear-elastic
cantilever: length 1.0 m, height 0.2 m, Young's modulus 210 GPa, Poisson ratio
0.30, left edge fixed, and a uniform downward traction of 1 MPa on the right
edge. Request displacement, stress and von Mises output.

Keep the engineering assumptions and units visible. Do not copy a pre-existing
project. Finish by writing `project/explanation.md`: explain the model,
boundary conditions, expected deformation, verification evidence and any
applicability limits. A successful process leaves `agentfem.toml`, `case.py`,
`result.json` and the explanation in `project/`.
""",
        encoding="utf-8",
    )
    (root / "REVIEW.md").write_text(
        """# Independent review

1. Export the complete fresh-task transcript as `agent-transcript.md`.
2. Record every human correction or redirect; zero means none after TASK.md.
3. Review `project/explanation.md` for mechanics, units, boundary conditions,
   evidence and limits—not literary style.
4. Run the repository's `tools/agent_trial_acceptance.py` with the wheel and
   source commit recorded in `trial-contract.json`.
5. Retain the task, transcript, project, explanation, result artifacts and
   acceptance JSON together.
""",
        encoding="utf-8",
    )
    return contract


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit")
    parser.add_argument("--agentfem-version")
    options = parser.parse_args()
    repository = Path(__file__).resolve().parents[1]
    if options.agentfem_version is None:
        from agentfem import __version__

        version = __version__
    else:
        version = options.agentfem_version
    report = prepare(
        wheel=options.wheel,
        output=options.output,
        source_commit=options.source_commit or _git_commit(repository),
        agentfem_version=version,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
