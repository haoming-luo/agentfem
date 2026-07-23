"""Runtime helpers shared by finite-element drivers."""

from __future__ import annotations

from dataclasses import dataclass

from mpi4py import MPI


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

    def __iter__(self):
        for index in range(self.start_step, self.total_steps + 1):
            time = self.start_time + index * self.dt
            is_last = index == self.total_steps
            yield TimeStep(
                index=index,
                time=time,
                total_steps=self.total_steps,
                should_save=self.save_every > 0 and (index % self.save_every == 0 or is_last),
                should_print=self.print_every > 0
                and (index == self.start_step or index % self.print_every == 0 or is_last),
            )


@dataclass
class ProgressPrinter:
    """Rank-zero progress printer controlled by a fixed step interval."""

    total_steps: int
    every: int
    comm: MPI.Comm

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
