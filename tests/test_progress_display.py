# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Terminal presentation must not change numerical evidence."""
from types import SimpleNamespace
import math

import pytest

from agentfem.time import ProgressPrinter, format_quantity


@pytest.mark.parametrize("value, expected", [
    (0.00039999999999999996, "0.0004 s"),
    (3.683206e-8, "3.68321e-08 s"),
    (-0.0, "0 s"), (1e12, "1e+12 s"),
    (math.nan, "nan s"), (math.inf, "inf s"),
])
def test_quantity(value, expected):
    assert format_quantity(value, "s") == expected


@pytest.mark.parametrize("precision", [0, 18, True, 2.5])
def test_invalid_precision(precision):
    with pytest.raises(ValueError):
        format_quantity(1, precision=precision)


def test_station_and_precision_preservation(capsys):
    printer = ProgressPrinter(100000, 100, SimpleNamespace(rank=0))
    values = {"Max |u_i|": (3.683206e-6, "mm")}
    before = repr(values)
    printer.update(400, time=.00039999999999999996,
                   elapsed_seconds=92.056, quantities=values)
    assert capsys.readouterr().out == (
        "Step 400/100,000 (0.4%) | Time: 0.0004 s | Elapsed: 92.1 s | "
        "Max |u_i|: 3.68321e-06 mm\n")
    assert repr(values) == before


def test_cadence_force_units_and_quiet(capsys):
    printer = ProgressPrinter(1000, 100, SimpleNamespace(rank=0))
    printer.update(2, time=2, elapsed_seconds=1)
    assert capsys.readouterr().out == ""
    printer.update(2, time=2, elapsed_seconds=1, time_unit="ms", force=True)
    assert "Time: 2 ms" in capsys.readouterr().out
    printer.every = 0
    printer.update(1000, time=1, elapsed_seconds=1, force=True)
    assert capsys.readouterr().out == ""
    printer.status("Partial run completed")
    assert capsys.readouterr().out == "Partial run completed\n"


def test_nonroot_and_legacy(capsys):
    printer = ProgressPrinter(10, 1, SimpleNamespace(rank=1))
    printer.update(1, time=0, elapsed_seconds=0)
    printer.status("Run failed")
    printer.print(1, "legacy")
    assert capsys.readouterr().out == ""
    printer.comm.rank = 0
    printer.print(1, "legacy")
    assert "legacy" in capsys.readouterr().out


@pytest.mark.parametrize("elapsed", [-1, math.nan, math.inf])
def test_bad_wall_time(elapsed):
    with pytest.raises(ValueError):
        ProgressPrinter(10, 1, SimpleNamespace(rank=0)).update(
            1, time=0, elapsed_seconds=elapsed)
