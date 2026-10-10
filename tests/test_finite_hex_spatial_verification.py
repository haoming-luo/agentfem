# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""A small independent spatial gate, not an industrial validation claim."""

from pathlib import Path
import runpy


def test_nonuniform_finite_deformation_converges_and_artificial_energy_decreases():
    run = runpy.run_path(
        str(Path(__file__).parents[1] / "tools" / "verify_finite_hex_manufactured.py")
    )["run"]
    records = [run(size) for size in (2, 4, 8)]
    for coarse, fine in zip(records, records[1:]):
        assert (
            fine["displacement_relative_l2"] < 0.3 * coarse["displacement_relative_l2"]
        )
        assert fine["hourglass_to_physical"] < 0.3 * coarse["hourglass_to_physical"]
    assert records[-1]["displacement_relative_l2"] < 0.014
    assert records[-1]["hourglass_to_physical"] < 0.005
    assert all(record["equilibrium_relative"] < 1e-9 for record in records)
