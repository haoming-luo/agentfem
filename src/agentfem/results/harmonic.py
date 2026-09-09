"""Result-owned response quantities sampled during harmonic sweeps."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class HarmonicResponse:
    """One named scalar real or complex response sampled at each frequency."""

    name: str
    evaluate: object
    unit: str | None = None
    description: str = ""

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        if not name:
            raise ValueError("HarmonicResponse.name must not be empty.")
        if not callable(self.evaluate):
            raise TypeError("HarmonicResponse.evaluate must be callable.")
        object.__setattr__(self, "name", name)

    def sample(self, step) -> complex:
        value = np.asarray(self.evaluate(step))
        if value.shape != ():
            raise ValueError(
                f"Harmonic response {self.name!r} must return one scalar."
            )
        selected = complex(value.item())
        if not np.isfinite(selected.real) or not np.isfinite(selected.imag):
            raise ValueError(
                f"Harmonic response {self.name!r} returned a non-finite value."
            )
        comm = step.solution_real.function_space.mesh.comm
        gathered = comm.allgather((selected.real, selected.imag))
        if any(
            not np.allclose(item, gathered[0], rtol=1.0e-12, atol=1.0e-14)
            for item in gathered[1:]
        ):
            raise RuntimeError(
                f"Harmonic response {self.name!r} differs across MPI ranks. "
                "Return a globally reduced quantity."
            )
        real, imaginary = gathered[0]
        return complex(real, imaginary)

    def to_ir(self) -> dict[str, object]:
        """Expose the evaluator to the common scientific-input fingerprint."""

        return {
            "name": self.name,
            "unit": self.unit,
            "description": self.description,
            "evaluate": self.evaluate,
        }

    def summary(self) -> dict[str, object]:
        return {
            "name": self.name,
            "unit": self.unit,
            "description": self.description,
            "evaluator": {
                "module": getattr(self.evaluate, "__module__", None),
                "qualname": getattr(
                    self.evaluate,
                    "__qualname__",
                    getattr(self.evaluate, "__name__", None),
                ),
                "identity_contract": "scientific_input_manifest",
            },
        }


def harmonic_response(
    name: str,
    evaluate,
    *,
    unit: str | None = None,
    description: str = "",
) -> HarmonicResponse:
    """Declare a traceable scalar response for an ordered frequency sweep."""

    return HarmonicResponse(
        name=name,
        evaluate=evaluate,
        unit=unit,
        description=description,
    )


__all__ = ["HarmonicResponse", "harmonic_response"]
