"""Reusable modal, frequency-response, and vibration-signal tools.

The functions in this module operate on arrays and on the structured result
layer.  They deliberately do not own a particular beam, excitation, or
material model, so the same contracts can be used by FEM steps, experiments,
campaigns, and external providers.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import pi
from typing import Mapping, Sequence

import numpy as np


def _one_dimensional(values, *, name: str, minimum: int = 1) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size < minimum:
        raise ValueError(f"{name} must be a one-dimensional array with at least {minimum} values.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _uniform_spacing(coordinate, *, name: str) -> tuple[np.ndarray, float]:
    values = _one_dimensional(coordinate, name=name, minimum=2)
    differences = np.diff(values)
    if np.any(differences <= 0.0):
        raise ValueError(f"{name} must be strictly increasing.")
    spacing = float(np.mean(differences))
    if not np.allclose(differences, spacing, rtol=1.0e-7, atol=1.0e-14):
        raise ValueError(f"{name} must be uniformly sampled for FFT processing.")
    return values, spacing


def _history_arrays(value, *, name: str) -> tuple[np.ndarray, np.ndarray]:
    """Return the abscissa and scalar values of a HistoryResult-like object."""

    if not hasattr(value, "abscissa") or not hasattr(value, "values"):
        raise TypeError(f"{name} must provide abscissa and values.")
    coordinate = _one_dimensional(value.abscissa, name=f"{name}.abscissa", minimum=2)
    samples = _one_dimensional(value.values, name=f"{name}.values", minimum=2)
    if coordinate.size != samples.size:
        raise ValueError(f"{name}.abscissa and {name}.values must have equal length.")
    return coordinate, samples


def _window(name: str | None, size: int) -> np.ndarray:
    selected = "none" if name is None else str(name).strip().lower().replace("-", "_")
    if selected in {"none", "rectangular", "boxcar"}:
        return np.ones(size, dtype=float)
    if selected in {"hann", "hanning"}:
        return np.hanning(size)
    if selected == "hamming":
        return np.hamming(size)
    if selected == "blackman":
        return np.blackman(size)
    raise ValueError("window must be 'none', 'hann', 'hamming', or 'blackman'.")


@dataclass(frozen=True)
class SignalSpectrum:
    """One-sided spectrum of a uniformly sampled real signal."""

    frequency: np.ndarray
    values: np.ndarray
    amplitude: np.ndarray
    phase: np.ndarray
    window: str
    sample_spacing: float

    def __post_init__(self) -> None:
        frequency = _one_dimensional(self.frequency, name="frequency")
        values = np.asarray(self.values, dtype=complex)
        amplitude = _one_dimensional(self.amplitude, name="amplitude")
        phase = _one_dimensional(self.phase, name="phase")
        if values.ndim != 1 or not (
            frequency.size == values.size == amplitude.size == phase.size
        ):
            raise ValueError("Spectrum arrays must have one common one-dimensional shape.")
        if self.sample_spacing <= 0.0:
            raise ValueError("sample_spacing must be positive.")
        object.__setattr__(self, "frequency", frequency.copy())
        object.__setattr__(self, "values", values.copy())
        object.__setattr__(self, "amplitude", amplitude.copy())
        object.__setattr__(self, "phase", phase.copy())

    @property
    def dominant_frequency(self) -> float:
        """Return the strongest non-zero frequency bin."""

        if self.frequency.size < 2:
            return 0.0
        return float(self.frequency[1 + np.argmax(self.amplitude[1:])])

    def to_result(self, *, name: str = "signal_spectrum", unit: str | None = None):
        """Return a :class:`SimulationResult` retaining the frequency axis."""

        from .results import SimulationResult

        result = SimulationResult(name=name)
        result.add_histories(
            self.frequency,
            {
                "amplitude": self.amplitude,
                "phase": self.phase,
                "real": self.values.real,
                "imaginary": self.values.imag,
            },
            units={"amplitude": unit, "phase": "rad", "real": unit, "imaginary": unit},
            abscissa_name="frequency",
            abscissa_unit="Hz",
        )
        result.add_quantity("dominant_frequency", self.dominant_frequency, unit="Hz")
        result.metadata["signal_processing"] = {
            "method": "one_sided_rfft",
            "window": self.window,
            "sample_spacing": self.sample_spacing,
        }
        return result


@dataclass(frozen=True)
class FrequencyResponse:
    """Complex frequency-response function with inspectable coherence mask."""

    frequency: np.ndarray
    response: np.ndarray
    valid: np.ndarray
    estimator: str = "spectral_ratio"

    def __post_init__(self) -> None:
        frequency = _one_dimensional(self.frequency, name="frequency")
        response = np.asarray(self.response, dtype=complex)
        valid = np.asarray(self.valid, dtype=bool)
        if response.ndim != 1 or valid.ndim != 1:
            raise ValueError("response and valid must be one-dimensional.")
        if not (frequency.size == response.size == valid.size):
            raise ValueError("Frequency-response arrays must have a common shape.")
        object.__setattr__(self, "frequency", frequency.copy())
        object.__setattr__(self, "response", response.copy())
        object.__setattr__(self, "valid", valid.copy())

    @property
    def amplitude(self) -> np.ndarray:
        return np.abs(self.response)

    @property
    def phase(self) -> np.ndarray:
        return np.angle(self.response)

    def to_result(self, *, name: str = "frequency_response", unit: str | None = None):
        from .results import SimulationResult

        result = SimulationResult(name=name)
        frequency = self.frequency[self.valid]
        response = self.response[self.valid]
        result.add_histories(
            frequency,
            {
                "amplitude": np.abs(response),
                "phase": np.angle(response),
                "real": response.real,
                "imaginary": response.imag,
            },
            units={"amplitude": unit, "phase": "rad", "real": unit, "imaginary": unit},
            abscissa_name="frequency",
            abscissa_unit="Hz",
        )
        result.metadata["frequency_response"] = {
            "estimator": self.estimator,
            "valid_bins": int(np.count_nonzero(self.valid)),
            "total_bins": int(self.valid.size),
        }
        return result


@dataclass(frozen=True)
class DampingEstimate:
    """Free-decay damping estimate from same-sign displacement peaks."""

    logarithmic_decrement: float
    damping_ratio: float
    quality_factor: float
    peak_indices: np.ndarray

    def as_dict(self) -> dict[str, object]:
        return {
            "logarithmic_decrement": self.logarithmic_decrement,
            "damping_ratio": self.damping_ratio,
            "quality_factor": self.quality_factor,
            "peak_indices": np.asarray(self.peak_indices, dtype=int).tolist(),
        }


@dataclass(frozen=True)
class ModalBasis:
    """Mass-normalized modes and their natural frequencies."""

    eigenvalues: np.ndarray
    modes: np.ndarray
    residual_norms: np.ndarray | None = None
    metadata: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        eigenvalues = _one_dimensional(self.eigenvalues, name="eigenvalues")
        modes = np.asarray(self.modes, dtype=float)
        if modes.ndim != 2 or modes.shape[1] != eigenvalues.size:
            raise ValueError("modes must have shape (dofs, number_of_eigenvalues).")
        if np.any(eigenvalues <= 0.0):
            raise ValueError("Modal eigenvalues must be positive after rigid-mode filtering.")
        residuals = (
            np.full(eigenvalues.size, np.nan)
            if self.residual_norms is None
            else _one_dimensional(self.residual_norms, name="residual_norms")
        )
        if residuals.size != eigenvalues.size:
            raise ValueError("residual_norms must match eigenvalues.")
        object.__setattr__(self, "eigenvalues", eigenvalues.copy())
        object.__setattr__(self, "modes", modes.copy())
        object.__setattr__(self, "residual_norms", residuals.copy())
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    @property
    def angular_frequencies(self) -> np.ndarray:
        return np.sqrt(self.eigenvalues)

    @property
    def frequencies(self) -> np.ndarray:
        return self.angular_frequencies / (2.0 * pi)

    @property
    def mode_count(self) -> int:
        return int(self.eigenvalues.size)

    def to_result(self, *, name: str = "modal_analysis"):
        from .results import SimulationResult

        result = SimulationResult(name=name)
        quantities = {
            "eigenvalues": self.eigenvalues,
            "angular_frequencies": self.angular_frequencies,
            "frequencies": self.frequencies,
        }
        if np.all(np.isfinite(self.residual_norms)):
            quantities["residual_norms"] = self.residual_norms
        result.add_quantities(
            quantities,
            units={
                "eigenvalues": "rad^2/s^2",
                "angular_frequencies": "rad/s",
                "frequencies": "Hz",
            },
            kind="modal",
        )
        result.metadata["modal_basis"] = {
            "mode_count": self.mode_count,
            "residual_norms_available": bool(
                np.all(np.isfinite(self.residual_norms))
            ),
            **dict(self.metadata or {}),
        }
        return result


def spectrum(
    time,
    signal=None,
    *,
    window: str | None = "hann",
    remove_mean: bool = True,
) -> SignalSpectrum:
    """Compute a correctly scaled one-sided FFT for a real signal.

    A scalar :class:`~agentfem.results.HistoryResult` may be supplied directly,
    or ``time`` and ``signal`` may be passed as separate arrays.
    """

    if signal is None:
        time, signal = _history_arrays(time, name="history")
    _, dt = _uniform_spacing(time, name="time")
    values = _one_dimensional(signal, name="signal", minimum=2)
    if values.size != np.asarray(time).size:
        raise ValueError("time and signal must have equal length.")
    processed = values - np.mean(values) if remove_mean else values.copy()
    weights = _window(window, processed.size)
    coherent_gain = float(np.mean(weights))
    if coherent_gain <= 0.0:
        raise ValueError("Selected FFT window has zero coherent gain.")
    transformed = np.fft.rfft(processed * weights) / (processed.size * coherent_gain)
    amplitude = np.abs(transformed)
    if processed.size > 1:
        upper = -1 if processed.size % 2 == 0 else None
        amplitude[1:upper] *= 2.0
    return SignalSpectrum(
        frequency=np.fft.rfftfreq(processed.size, dt),
        values=transformed,
        amplitude=amplitude,
        phase=np.angle(transformed),
        window="none" if window is None else str(window),
        sample_spacing=dt,
    )


def frequency_response(
    time,
    excitation,
    response=None,
    *,
    window: str | None = "hann",
    minimum_input_ratio: float = 1.0e-10,
) -> FrequencyResponse:
    """Estimate an FRF from synchronous input and output time histories.

    Bins with negligible input amplitude are explicitly marked invalid instead
    of silently producing very large ratios.
    """

    if response is None:
        input_time, input_values = _history_arrays(time, name="excitation_history")
        output_time, output_values = _history_arrays(
            excitation,
            name="response_history",
        )
        if not np.array_equal(input_time, output_time):
            raise ValueError("Excitation and response histories must share one time axis.")
        time, excitation, response = input_time, input_values, output_values
    input_spectrum = spectrum(time, excitation, window=window)
    output_spectrum = spectrum(time, response, window=window)
    scale = float(np.max(np.abs(input_spectrum.values)))
    threshold = max(np.finfo(float).eps, minimum_input_ratio * scale)
    valid = np.abs(input_spectrum.values) > threshold
    ratio = np.full(input_spectrum.values.shape, np.nan + 1j * np.nan)
    ratio[valid] = output_spectrum.values[valid] / input_spectrum.values[valid]
    return FrequencyResponse(input_spectrum.frequency, ratio, valid)


def damping_from_free_decay(signal, *, peak_indices: Sequence[int] | None = None) -> DampingEstimate:
    """Estimate damping from positive peaks of an underdamped free decay."""

    if hasattr(signal, "abscissa") and hasattr(signal, "values"):
        _, signal = _history_arrays(signal, name="history")
    values = _one_dimensional(signal, name="signal", minimum=3)
    if peak_indices is None:
        candidates = np.flatnonzero(
            (values[1:-1] > values[:-2]) & (values[1:-1] >= values[2:]) & (values[1:-1] > 0.0)
        ) + 1
    else:
        candidates = np.asarray(peak_indices, dtype=int)
    if candidates.ndim != 1 or candidates.size < 2:
        raise ValueError("At least two positive free-decay peaks are required.")
    peaks = values[candidates]
    if np.any(peaks <= 0.0) or np.any(np.diff(candidates) <= 0):
        raise ValueError("peak_indices must identify ordered positive peaks.")
    decrements = np.log(peaks[:-1] / peaks[1:])
    decrement = float(np.mean(decrements))
    if decrement <= 0.0:
        raise ValueError("Peak amplitudes must decay monotonically on average.")
    ratio = decrement / np.sqrt((2.0 * pi) ** 2 + decrement**2)
    return DampingEstimate(
        logarithmic_decrement=decrement,
        damping_ratio=float(ratio),
        quality_factor=float(1.0 / (2.0 * ratio)),
        peak_indices=candidates,
    )


def solve_dense_modes(stiffness, mass, *, modes: int | None = None) -> ModalBasis:
    """Solve a small dense symmetric generalized eigenproblem.

    This deterministic reference is intended for reduced models and unit
    tests. Distributed finite-element modes use the SLEPc-backed modal Step.
    """

    stiffness = np.asarray(stiffness, dtype=float)
    mass = np.asarray(mass, dtype=float)
    if stiffness.ndim != 2 or stiffness.shape[0] != stiffness.shape[1]:
        raise ValueError("stiffness must be a square matrix.")
    if mass.shape != stiffness.shape:
        raise ValueError("mass must match stiffness shape.")
    if not np.allclose(stiffness, stiffness.T) or not np.allclose(mass, mass.T):
        raise ValueError("Dense modal reference requires symmetric stiffness and mass.")
    try:
        from scipy.linalg import eigh

        eigenvalues, vectors = eigh(stiffness, mass, check_finite=True)
    except ImportError:
        chol = np.linalg.cholesky(mass)
        transformed = np.linalg.solve(chol, stiffness)
        transformed = np.linalg.solve(chol, transformed.T).T
        eigenvalues, reduced = np.linalg.eigh(transformed)
        vectors = np.linalg.solve(chol.T, reduced)
    positive = eigenvalues > max(1.0, float(np.max(np.abs(eigenvalues)))) * 1.0e-12
    eigenvalues = eigenvalues[positive]
    vectors = vectors[:, positive]
    if modes is not None:
        if int(modes) <= 0:
            raise ValueError("modes must be positive.")
        eigenvalues = eigenvalues[: int(modes)]
        vectors = vectors[:, : int(modes)]
    residuals = np.asarray(
        [
            np.linalg.norm(stiffness @ vectors[:, i] - eigenvalues[i] * mass @ vectors[:, i])
            for i in range(eigenvalues.size)
        ]
    )
    return ModalBasis(eigenvalues, vectors, residuals, {"solver": "dense_reference"})


def modal_frequency_response(
    basis: ModalBasis,
    frequencies,
    modal_force,
    *,
    damping_ratio=0.0,
) -> np.ndarray:
    """Return modal coordinates for harmonic forcing by modal superposition."""

    frequency = _one_dimensional(frequencies, name="frequencies")
    force = _one_dimensional(modal_force, name="modal_force")
    if force.size != basis.mode_count:
        raise ValueError("modal_force must provide one value per mode.")
    damping = np.broadcast_to(np.asarray(damping_ratio, dtype=float), force.shape)
    if np.any(damping < 0.0):
        raise ValueError("damping_ratio must be nonnegative.")
    omega = 2.0 * pi * frequency[:, None]
    natural = basis.angular_frequencies[None, :]
    denominator = natural**2 - omega**2 + 2j * damping[None, :] * natural * omega
    return force[None, :] / denominator


__all__ = [
    "DampingEstimate",
    "FrequencyResponse",
    "ModalBasis",
    "SignalSpectrum",
    "damping_from_free_decay",
    "frequency_response",
    "modal_frequency_response",
    "solve_dense_modes",
    "spectrum",
]
