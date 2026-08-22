"""Fixed AgentFEM entry point for PDEAgent-Bench's ``--solver-path`` mode."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src"
source_root = str(SOURCE_ROOT)
if SOURCE_ROOT.is_dir() and source_root not in sys.path:
    # Solver-path evaluation must use the frozen checkout selected by its path.
    sys.path.insert(0, source_root)

from agentfem.integrations.pdeagent_bench import solve_case


solve = solve_case
