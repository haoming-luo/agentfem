# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Runtime helpers shared by finite-element drivers."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import os
import sys
from typing import Mapping, TextIO

from mpi4py import MPI


def _display_text(value: str) -> str:
    """Keep caller labels from injecting terminal controls or extra lines."""
    return "".join(char if char.isprintable() else ascii(char)[1:-1] for char in str(value))


def format_quantity(value: float, unit: str = "", *, precision: int = 6) -> str:
    """Compact terminal display only; never round values stored as evidence.

    Units are caller-declared. Nonfinite diagnostics remain visible instead of
    being hidden as zero. No unit conversion or physical reduction is performed.
    """
    if isinstance(precision, bool) or not isinstance(precision, int) or not 1 <= precision <= 17:
        raise ValueError("precision must be an integer between 1 and 17")
    number = float(value)
    text = format(number, f".{precision}g") if number != 0 else "0"
    return f"{text} {_display_text(unit)}" if unit else text


def format_duration(seconds: float) -> str:
    """Format elapsed seconds as ``HH:MM:SS``."""

    total = int(round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


@dataclass(frozen=True)
class TimeStep:
    """Metadata for one transient-solve step."""

    index: int
    time: float
    total_steps: int
    should_save: bool
    should_print: bool

    @property
    def is_first(self) -> bool:
        return self.index == 1

    @property
    def is_last(self) -> bool:
        return self.index == self.total_steps

    @property
    def percent(self) -> float:
        return 100.0 * self.index / self.total_steps


@dataclass(frozen=True)
class TimeStepper:
    """Iterate over transient-solve step metadata."""

    total_steps: int
    dt: float
    save_every: int
    print_every: int = 0
    start_step: int = 1
    start_time: float = 0.0
    stop_step: int | None = None

    def __iter__(self):
        stop = self.total_steps if self.stop_step is None else self.stop_step
        if not 1 <= self.start_step <= self.total_steps + 1:
            raise ValueError("start_step must lie in [1, total_steps + 1].")
        if not self.start_step - 1 <= stop <= self.total_steps:
            raise ValueError("stop_step must lie in [start_step - 1, total_steps].")
        for index in range(self.start_step, stop + 1):
            time = self.start_time + index * self.dt
            is_last = index == self.total_steps
            is_stop = index == stop
            yield TimeStep(
                index=index,
                time=time,
                total_steps=self.total_steps,
                should_save=self.save_every > 0
                and (index % self.save_every == 0 or is_last or is_stop),
                should_print=self.print_every > 0
                and (
                    index == self.start_step
                    or index % self.print_every == 0
                    or is_last
                    or is_stop
                ),
            )


@dataclass
class ProgressPrinter:
    """Rank-zero progress printer controlled by a fixed step interval."""

    total_steps: int
    every: int
    comm: MPI.Comm
    stream: TextIO | None = field(default=None, kw_only=True, repr=False)
    width: int | None = field(default=None, kw_only=True)

    def __post_init__(self):
        for name, value, minimum in (("total_steps", self.total_steps, 1),
                                      ("every", self.every, 0)):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if self.width is not None and (
            isinstance(self.width, bool) or not isinstance(self.width, int) or self.width < 20
        ):
            raise ValueError("width must be None or an integer >= 20")

    def _write(self, parts):
        """Append readable lines; never redraw or truncate scientific values."""
        stream = self.stream if self.stream is not None else sys.stderr
        width = self.width
        if width is None and getattr(stream, "isatty", lambda: False)():
            try:
                width = max(20, os.get_terminal_size(stream.fileno()).columns)
            except (AttributeError, OSError, ValueError):
                width = 80
        line = parts[0]
        lines = []
        for part in parts[1:]:
            if width is not None and len(line) + 3 + len(part) > width:
                lines.append(line)
                line = "  " + part
            else:
                line += " | " + part
        lines.append(line)
        print("\n".join(lines), file=stream, flush=True)

    def should_print(self, step: int) -> bool:
        """Return true when this step should be reported."""

        return self.every > 0 and (
            step == 1 or step == self.total_steps or step % self.every == 0
        )

    def print(self, step: int, message: str) -> None:
        """Print a formatted progress line on rank zero."""

        if self.comm.rank != 0:
            return
        percent = 100.0 * step / self.total_steps
        print(
            f"step {step:>6}/{self.total_steps} ({percent:5.1f}%) | {message}",
            flush=True,
        )

    def update(
        self, step: int, *, time: float, elapsed_seconds: float,
        time_unit: str = "s",
        quantities: Mapping[str, tuple[float, str]] | None = None,
        force: bool = False,
    ) -> None:
        """Print an accepted station at the configured cadence, on rank zero.

        The caller owns acceptance and any MPI reductions. ``time`` uses the
        declared model unit; wall-clock elapsed time is always seconds. ``force``
        can display a partial/restart stop, but does not override ``every=0``.
        This method does not write machine logs or modify supplied quantities.
        """
        if self.comm.rank != 0 or self.every <= 0:
            return
        if isinstance(step, bool) or not isinstance(step, int) or not 0 <= step <= self.total_steps:
            raise ValueError("step must lie between zero and positive total_steps")
        if not (force or self.should_print(step)):
            return
        if not math.isfinite(elapsed_seconds) or elapsed_seconds < 0:
            raise ValueError("elapsed_seconds must be finite and nonnegative")
        parts = [
            f"Step {step:,}/{self.total_steps:,} ({100 * step / self.total_steps:.1f}%)",
            f"Time: {format_quantity(time, time_unit)}",
            f"Elapsed: {format_duration(elapsed_seconds) if elapsed_seconds >= 60 else f'{elapsed_seconds:.1f} s'}",
        ]
        parts.extend(self._quantities(quantities))
        self._write(parts)

    @staticmethod
    def _quantities(quantities):
        return [f"{_display_text(name)}: {format_quantity(value, unit)}"
                for name, (value, unit) in (quantities or {}).items()]

    def status(
        self, message: str, *, quantities: Mapping[str, tuple[float, str]] | None = None,
    ) -> None:
        """Print an explicit lifecycle message, even when progress is disabled.

        Use accurate labels such as 'Model ready', 'Partial run completed' or
        'Run failed'. The printer never infers scientific verification status.
        """
        if self.comm.rank == 0:
            self._write([_display_text(message), *self._quantities(quantities)])
