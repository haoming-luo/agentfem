# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Terminal presentation must not change numerical evidence."""
from types import SimpleNamespace
import math
from io import StringIO

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
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == (
        "Step 400/100,000 (0.4%) | Time: 0.0004 s | Elapsed: 00:01:32 | "
        "Max |u_i|: 3.68321e-06 mm\n")
    assert repr(values) == before


def test_cadence_force_units_and_quiet(capsys):
    printer = ProgressPrinter(1000, 100, SimpleNamespace(rank=0))
    printer.update(2, time=2, elapsed_seconds=1)
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""
    printer.update(2, time=2, elapsed_seconds=1, time_unit="ms", force=True)
    assert "Time: 2 ms" in capsys.readouterr().err
    printer.every = 0
    printer.update(1000, time=1, elapsed_seconds=1, force=True)
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""
    printer.status("Partial run completed")
    assert capsys.readouterr().err == "Partial run completed\n"


def test_nonroot_and_legacy(capsys):
    printer = ProgressPrinter(10, 1, SimpleNamespace(rank=1))
    printer.update(1, time=0, elapsed_seconds=0)
    printer.status("Run failed")
    printer.print(1, "legacy")
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""
    printer.comm.rank = 0
    printer.print(1, "legacy")
    assert "legacy" in capsys.readouterr().out


@pytest.mark.parametrize("elapsed", [-1, math.nan, math.inf])
def test_bad_wall_time(elapsed):
    with pytest.raises(ValueError):
        ProgressPrinter(10, 1, SimpleNamespace(rank=0)).update(
            1, time=0, elapsed_seconds=elapsed)


def test_narrow_output_preserves_whole_fields():
    output = StringIO()
    printer = ProgressPrinter(1000, 1, SimpleNamespace(rank=0), stream=output, width=40)
    printer.update(600, time=.0006, elapsed_seconds=14640.5,
                   quantities={"Maximum stress": (123.456789, "MPa")})
    text = output.getvalue()
    assert text == ("Step 600/1,000 (60.0%) | Time: 0.0006 s\n"
                    "  Elapsed: 04:04:00\n"
                    "  Maximum stress: 123.457 MPa\n")
    assert "\x1b" not in text and "\r" not in text


def test_control_characters_cannot_overwrite_terminal():
    output = StringIO()
    printer = ProgressPrinter(1, 1, SimpleNamespace(rank=0), stream=output)
    printer.status("Run failed\nforged", quantities={"bad\rlabel": (1, "mm\x1b[2J")})
    text = output.getvalue()
    assert len(text.splitlines()) == 1
    assert "\\n" in text and "\\r" in text and "\\x1b" in text
    assert "\x1b" not in text


@pytest.mark.parametrize("kwargs", [{"total_steps": 0}, {"total_steps": True},
    {"every": -1}, {"every": 1.5}, {"width": 1}, {"width": True}])
def test_invalid_settings(kwargs):
    settings = dict(total_steps=100, every=10, comm=SimpleNamespace(rank=0))
    settings.update(kwargs)
    with pytest.raises(ValueError):
        ProgressPrinter(**settings)


@pytest.mark.parametrize("step", [-1, 101, True, 1.5])
def test_bad_step_even_off_cadence(step):
    with pytest.raises(ValueError):
        ProgressPrinter(100, 10, SimpleNamespace(rank=0)).update(
            step, time=0, elapsed_seconds=0)


def test_terminal_width_and_last_station(monkeypatch):
    import os
    output = StringIO()
    output.isatty = lambda: True
    output.fileno = lambda: 2
    monkeypatch.setattr(os, "get_terminal_size", lambda fd: os.terminal_size((50, 20)))
    printer = ProgressPrinter(101, 50, SimpleNamespace(rank=0), stream=output)
    printer.update(101, time=1.01, elapsed_seconds=9)
    assert "(100.0%)" in output.getvalue()
    assert all(len(line) <= 50 for line in output.getvalue().splitlines())
