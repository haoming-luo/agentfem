"""External-data contracts for mixed-mode cohesive structure benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np


def _curve_values(values, *, name: str) -> np.ndarray:
    selected = np.asarray(values, dtype=float).reshape(-1)
    if selected.size < 2 or not np.all(np.isfinite(selected)):
        raise ValueError(f"{name} must contain at least two finite values.")
    return selected


@dataclass(frozen=True)
class MixedModeBendingCurve:
    """One traceable load/displacement/mode-mix curve versus crack length."""

    crack_length: np.ndarray
    load: np.ndarray
    displacement: np.ndarray
    mode_i_fraction: np.ndarray
    source: str
    identity_sha256: str

    @classmethod
    def create(
        cls,
        *,
        crack_length,
        load,
        displacement,
        mode_i_fraction,
        source: str,
    ) -> "MixedModeBendingCurve":
        arrays = {
            "crack_length": _curve_values(crack_length, name="crack_length"),
            "load": _curve_values(load, name="load"),
            "displacement": _curve_values(displacement, name="displacement"),
            "mode_i_fraction": _curve_values(
                mode_i_fraction, name="mode_i_fraction"
            ),
        }
        sizes = {value.size for value in arrays.values()}
        if len(sizes) != 1:
            raise ValueError("Mixed-mode curve columns must have equal length.")
        if np.any(np.diff(arrays["crack_length"]) <= 0.0):
            raise ValueError("crack_length must be strictly increasing.")
        if np.any(
            (arrays["mode_i_fraction"] < 0.0)
            | (arrays["mode_i_fraction"] > 1.0)
        ):
            raise ValueError("mode_i_fraction must remain within [0, 1].")
        selected_source = str(source).strip()
        if not selected_source:
            raise ValueError("A mixed-mode reference must declare its source.")
        digest = sha256()
        digest.update(selected_source.encode("utf-8"))
        for name, values in arrays.items():
            digest.update(name.encode("ascii"))
            digest.update(np.asarray(values, dtype="<f8").tobytes())
        return cls(
            source=selected_source,
            identity_sha256=digest.hexdigest(),
            **{name: value.copy() for name, value in arrays.items()},
        )

    @classmethod
    def read_csv(
        cls,
        path,
        *,
        source: str,
    ) -> "MixedModeBendingCurve":
        """Read the four-column external curve contract without pandas."""

        selected = Path(path)
        records = np.genfromtxt(
            selected,
            delimiter=",",
            names=True,
            dtype=float,
            encoding="utf-8",
        )
        required = {
            "crack_length",
            "load",
            "displacement",
            "mode_i_fraction",
        }
        names = set(records.dtype.names or ())
        if names != required:
            raise ValueError(
                "Mixed-mode CSV columns must be exactly "
                f"{sorted(required)}; received {sorted(names)}."
            )
        return cls.create(
            crack_length=records["crack_length"],
            load=records["load"],
            displacement=records["displacement"],
            mode_i_fraction=records["mode_i_fraction"],
            source=source,
        )


@dataclass(frozen=True)
class MixedModeBendingComparison:
    """Curve-level errors under explicitly declared scientific tolerances."""

    reference_identity_sha256: str
    predicted_identity_sha256: str
    load_relative_l2_error: float
    displacement_relative_l2_error: float
    mode_i_fraction_maximum_error: float
    load_relative_tolerance: float
    displacement_relative_tolerance: float
    mode_i_fraction_absolute_tolerance: float
    accepted: bool

    def summary(self) -> dict[str, object]:
        return {
            "schema": "agentfem.mixed-mode-bending-comparison.v1",
            **self.__dict__,
        }


def _relative_l2(observed: np.ndarray, reference: np.ndarray) -> float:
    scale = float(np.linalg.norm(reference))
    if scale <= np.finfo(float).eps:
        return float(np.linalg.norm(observed - reference))
    return float(np.linalg.norm(observed - reference) / scale)


def compare_mixed_mode_bending_curves(
    reference: MixedModeBendingCurve,
    predicted: MixedModeBendingCurve,
    *,
    load_relative_tolerance: float,
    displacement_relative_tolerance: float,
    mode_i_fraction_absolute_tolerance: float,
) -> MixedModeBendingComparison:
    """Compare a computed curve on the reference crack-length coordinates."""

    tolerances = (
        float(load_relative_tolerance),
        float(displacement_relative_tolerance),
        float(mode_i_fraction_absolute_tolerance),
    )
    if any(not np.isfinite(value) or value < 0.0 for value in tolerances):
        raise ValueError("Mixed-mode comparison tolerances must be finite and nonnegative.")
    if (
        predicted.crack_length[0] > reference.crack_length[0]
        or predicted.crack_length[-1] < reference.crack_length[-1]
    ):
        raise ValueError("Predicted crack-length range must cover the reference range.")
    coordinates = reference.crack_length
    load_error = _relative_l2(
        np.interp(coordinates, predicted.crack_length, predicted.load),
        reference.load,
    )
    displacement_error = _relative_l2(
        np.interp(coordinates, predicted.crack_length, predicted.displacement),
        reference.displacement,
    )
    mix_error = float(
        np.max(
            np.abs(
                np.interp(
                    coordinates,
                    predicted.crack_length,
                    predicted.mode_i_fraction,
                )
                - reference.mode_i_fraction
            )
        )
    )
    return MixedModeBendingComparison(
        reference_identity_sha256=reference.identity_sha256,
        predicted_identity_sha256=predicted.identity_sha256,
        load_relative_l2_error=load_error,
        displacement_relative_l2_error=displacement_error,
        mode_i_fraction_maximum_error=mix_error,
        load_relative_tolerance=tolerances[0],
        displacement_relative_tolerance=tolerances[1],
        mode_i_fraction_absolute_tolerance=tolerances[2],
        accepted=(
            load_error <= tolerances[0]
            and displacement_error <= tolerances[1]
            and mix_error <= tolerances[2]
        ),
    )


__all__ = [
    "MixedModeBendingComparison",
    "MixedModeBendingCurve",
    "compare_mixed_mode_bending_curves",
]
