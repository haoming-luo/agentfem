"""Backend-neutral modal clustering and invariant-subspace evidence.

Individual eigenvectors are not unique inside a repeated eigenspace.  This
module therefore keeps eigenvalue clustering and subspace comparison separate
from both dense and distributed eigensolver orchestration.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import degrees
from typing import Sequence

import numpy as np


def _finite_eigenvalues(values, *, name: str) -> np.ndarray:
    selected = np.asarray(values, dtype=float)
    if selected.ndim != 1 or selected.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array.")
    if not np.all(np.isfinite(selected)):
        raise ValueError(f"{name} must contain only finite values.")
    if np.any(np.diff(selected) < 0.0):
        raise ValueError(f"{name} must be sorted in nondecreasing order.")
    return selected


@dataclass(frozen=True)
class EigenvalueCluster:
    """One singleton or repeated group in an ordered real eigenspectrum."""

    start: int
    stop: int
    eigenvalues: tuple[float, ...]
    relative_spread: float

    @property
    def indices(self) -> tuple[int, ...]:
        return tuple(range(self.start, self.stop))

    @property
    def multiplicity(self) -> int:
        return self.stop - self.start

    @property
    def repeated(self) -> bool:
        return self.multiplicity > 1

    def as_dict(self) -> dict[str, object]:
        return {
            "indices": tuple(index + 1 for index in self.indices),
            "multiplicity": self.multiplicity,
            "minimum_eigenvalue": min(self.eigenvalues),
            "maximum_eigenvalue": max(self.eigenvalues),
            "relative_spread": self.relative_spread,
            "comparison_object": (
                "invariant_subspace" if self.repeated else "individual_mode"
            ),
        }


def cluster_eigenvalues(
    eigenvalues,
    *,
    relative_tolerance: float = 1.0e-6,
    absolute_tolerance: float = 0.0,
) -> tuple[EigenvalueCluster, ...]:
    """Group adjacent eigenvalues that cannot be resolved as distinct modes."""

    values = _finite_eigenvalues(eigenvalues, name="eigenvalues")
    if not np.isfinite(relative_tolerance) or relative_tolerance < 0.0:
        raise ValueError("relative_tolerance must be finite and nonnegative.")
    if not np.isfinite(absolute_tolerance) or absolute_tolerance < 0.0:
        raise ValueError("absolute_tolerance must be finite and nonnegative.")
    boundaries = [0]
    for index in range(1, values.size):
        left = float(values[index - 1])
        right = float(values[index])
        scale = max(abs(left), abs(right), np.finfo(float).tiny)
        threshold = float(absolute_tolerance) + float(relative_tolerance) * scale
        if abs(right - left) > threshold:
            boundaries.append(index)
    boundaries.append(values.size)
    clusters = []
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        selected = values[start:stop]
        scale = max(float(np.max(np.abs(selected))), np.finfo(float).tiny)
        clusters.append(
            EigenvalueCluster(
                start=start,
                stop=stop,
                eigenvalues=tuple(float(value) for value in selected),
                relative_spread=float(np.ptp(selected) / scale),
            )
        )
    return tuple(clusters)


def cluster_summaries(
    eigenvalues,
    *,
    relative_tolerance: float = 1.0e-6,
    absolute_tolerance: float = 0.0,
) -> tuple[dict[str, object], ...]:
    return tuple(
        cluster.as_dict()
        for cluster in cluster_eigenvalues(
            eigenvalues,
            relative_tolerance=relative_tolerance,
            absolute_tolerance=absolute_tolerance,
        )
    )


def _metric_matrix(metric, *, size: int) -> np.ndarray:
    if metric is None:
        return np.eye(size)
    selected = np.asarray(metric, dtype=float)
    if selected.shape != (size, size):
        raise ValueError("metric must have shape (dofs, dofs).")
    if not np.all(np.isfinite(selected)) or not np.allclose(
        selected,
        selected.T,
    ):
        raise ValueError("metric must be finite and symmetric.")
    eigenvalues = np.linalg.eigvalsh(selected)
    if np.min(eigenvalues) <= 0.0:
        raise ValueError("metric must be positive definite.")
    return selected


def _metric_orthonormal_basis(values: np.ndarray, metric: np.ndarray) -> np.ndarray:
    gram = values.T @ metric @ values
    gram = 0.5 * (gram + gram.T)
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    scale = max(float(np.max(np.abs(eigenvalues))), np.finfo(float).tiny)
    if np.min(eigenvalues) <= 100.0 * np.finfo(float).eps * scale:
        raise ValueError("Modal cluster vectors must be linearly independent.")
    inverse_root = (
        eigenvectors
        @ np.diag(1.0 / np.sqrt(eigenvalues))
        @ eigenvectors.T
    )
    return values @ inverse_root


@dataclass(frozen=True)
class ModalClusterComparison:
    """Basis-independent comparison of one corresponding modal cluster."""

    indices: tuple[int, ...]
    canonical_correlations: tuple[float, ...]
    maximum_principal_angle_degrees: float
    projection_distance: float
    modal_assurance: float | None

    def as_dict(self) -> dict[str, object]:
        return {
            "indices": tuple(index + 1 for index in self.indices),
            "multiplicity": len(self.indices),
            "canonical_correlations": self.canonical_correlations,
            "maximum_principal_angle_degrees": self.maximum_principal_angle_degrees,
            "projection_distance": self.projection_distance,
            "modal_assurance": self.modal_assurance,
            "comparison_object": (
                "invariant_subspace" if len(self.indices) > 1 else "individual_mode"
            ),
        }


@dataclass(frozen=True)
class ModalBasisComparison:
    """Frequency and invariant-subspace evidence for two modal bases."""

    eigenvalue_relative_errors: tuple[float, ...]
    clusters: tuple[ModalClusterComparison, ...]
    cluster_partition_matches: bool
    eigenvalue_tolerance: float
    subspace_tolerance: float

    @property
    def accepted(self) -> bool:
        return (
            self.cluster_partition_matches
            and max(self.eigenvalue_relative_errors, default=float("inf"))
            <= self.eigenvalue_tolerance
            and all(
                cluster.projection_distance <= self.subspace_tolerance
                for cluster in self.clusters
            )
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "accepted": self.accepted,
            "eigenvalue_relative_errors": self.eigenvalue_relative_errors,
            "maximum_eigenvalue_relative_error": max(
                self.eigenvalue_relative_errors,
                default=float("inf"),
            ),
            "cluster_partition_matches": self.cluster_partition_matches,
            "eigenvalue_tolerance": self.eigenvalue_tolerance,
            "subspace_tolerance": self.subspace_tolerance,
            "clusters": tuple(cluster.as_dict() for cluster in self.clusters),
        }


def compare_modal_bases(
    reference_eigenvalues,
    reference_modes,
    candidate_eigenvalues,
    candidate_modes,
    *,
    metric=None,
    cluster_tolerance: float = 1.0e-6,
    eigenvalue_tolerance: float = 1.0e-5,
    subspace_tolerance: float = 1.0e-4,
) -> ModalBasisComparison:
    """Compare modes without assigning meaning to a basis of a repeated space.

    Singleton clusters use the familiar modal assurance criterion.  Repeated
    clusters use canonical correlations and the Frobenius projection distance,
    which are unchanged by rotations or sign changes within either basis.
    """

    reference_values = _finite_eigenvalues(
        reference_eigenvalues,
        name="reference_eigenvalues",
    )
    candidate_values = _finite_eigenvalues(
        candidate_eigenvalues,
        name="candidate_eigenvalues",
    )
    if reference_values.shape != candidate_values.shape:
        raise ValueError("Reference and candidate bases must contain equal mode counts.")
    reference = np.asarray(reference_modes, dtype=float)
    candidate = np.asarray(candidate_modes, dtype=float)
    expected_shape = (reference.shape[0], reference_values.size)
    if reference.ndim != 2 or reference.shape != expected_shape:
        raise ValueError("reference_modes must have shape (dofs, modes).")
    if candidate.shape != reference.shape:
        raise ValueError("candidate_modes must match reference_modes shape.")
    if not np.all(np.isfinite(reference)) or not np.all(np.isfinite(candidate)):
        raise ValueError("Modal bases must contain only finite values.")
    for value, name in (
        (cluster_tolerance, "cluster_tolerance"),
        (eigenvalue_tolerance, "eigenvalue_tolerance"),
        (subspace_tolerance, "subspace_tolerance"),
    ):
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and nonnegative.")
    selected_metric = _metric_matrix(metric, size=reference.shape[0])
    reference_clusters = cluster_eigenvalues(
        reference_values,
        relative_tolerance=cluster_tolerance,
    )
    candidate_clusters = cluster_eigenvalues(
        candidate_values,
        relative_tolerance=cluster_tolerance,
    )
    reference_partition = tuple(cluster.indices for cluster in reference_clusters)
    candidate_partition = tuple(cluster.indices for cluster in candidate_clusters)
    comparisons = []
    for cluster in reference_clusters:
        indices = cluster.indices
        reference_basis = _metric_orthonormal_basis(
            reference[:, indices],
            selected_metric,
        )
        candidate_basis = _metric_orthonormal_basis(
            candidate[:, indices],
            selected_metric,
        )
        cross = reference_basis.T @ selected_metric @ candidate_basis
        correlations = np.clip(np.linalg.svd(cross, compute_uv=False), 0.0, 1.0)
        minimum = float(np.min(correlations))
        projection_distance = float(
            np.sqrt(max(0.0, len(indices) - float(np.sum(correlations**2))))
        )
        comparisons.append(
            ModalClusterComparison(
                indices=indices,
                canonical_correlations=tuple(float(value) for value in correlations),
                maximum_principal_angle_degrees=degrees(float(np.arccos(minimum))),
                projection_distance=projection_distance,
                modal_assurance=(
                    float(correlations[0] ** 2) if len(indices) == 1 else None
                ),
            )
        )
    scale = np.maximum(np.abs(reference_values), np.finfo(float).tiny)
    eigenvalue_errors = np.abs(candidate_values - reference_values) / scale
    return ModalBasisComparison(
        eigenvalue_relative_errors=tuple(float(value) for value in eigenvalue_errors),
        clusters=tuple(comparisons),
        cluster_partition_matches=reference_partition == candidate_partition,
        eigenvalue_tolerance=float(eigenvalue_tolerance),
        subspace_tolerance=float(subspace_tolerance),
    )


def selected_clusters_are_complete(
    all_eigenvalues: Sequence[float],
    selected_positions: Sequence[int],
    *,
    relative_tolerance: float,
) -> bool:
    """Return whether selection contains every member of each touched cluster."""

    values = _finite_eigenvalues(all_eigenvalues, name="all_eigenvalues")
    selected = {int(position) for position in selected_positions}
    if any(position < 0 or position >= values.size for position in selected):
        raise ValueError("selected_positions must index all_eigenvalues.")
    return all(
        not selected.intersection(cluster.indices)
        or selected.issuperset(cluster.indices)
        for cluster in cluster_eigenvalues(
            values,
            relative_tolerance=relative_tolerance,
        )
    )


__all__ = [
    "EigenvalueCluster",
    "ModalBasisComparison",
    "ModalClusterComparison",
    "cluster_eigenvalues",
    "cluster_summaries",
    "compare_modal_bases",
    "selected_clusters_are_complete",
]
