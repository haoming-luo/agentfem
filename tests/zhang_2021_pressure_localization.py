"""Offline pressure-defect localization for the bounded Zhang unit-cell fixture.

No FEM runtime is imported. Reconstructed channels must agree with the accepted
candidate before any spatial attribution is reported. This is a diagnostic,
not an a posteriori error bound or an automatic refinement prescription.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

from zhang_2021_plane_strain_promotion import load_candidate


def localize(
    *,
    coordinates,
    weights,
    gradient,
    stress,
    condensed,
    bulk,
    cell_ids,
    phase,
    reference_volume,
    expected,
):
    x, w, F, S, e, k = (
        np.asarray(value, dtype=float)
        for value in (coordinates, weights, gradient, stress, condensed, bulk)
    )
    if (
        w.ndim != 2
        or not w.size
        or x.shape != (*w.shape, 2)
        or F.shape != (*w.shape, 3, 3)
        or S.shape != F.shape
    ):
        raise ValueError("Expected aligned cell-by-quadrature arrays.")
    if (
        e.shape != w.shape
        or k.shape != w.shape
        or phase.shape != w.shape
        or phase.dtype != np.dtype(bool)
    ):
        raise ValueError("Scalar channels must match physical weights.")
    if len(cell_ids) != len(w) or len(np.unique(cell_ids)) != len(w):
        raise ValueError("Cell identities must be unique and aligned.")
    if not all(np.all(np.isfinite(value)) for value in (x, w, F, S, e, k)):
        raise ValueError("Non-finite integration-point data.")
    if np.any(w < 0) or np.any(w.sum(axis=1) <= 0) or np.any(k <= 0):
        raise ValueError("Positive cell measures and bulk moduli are required.")
    if not np.isfinite(reference_volume) or reference_volume <= 0:
        raise ValueError("Positive finite reference volume is required.")
    J = np.linalg.det(F)
    if np.any(J <= 0):
        raise ValueError("Non-positive deformation Jacobian.")
    # Mixed J2 pressure is mean Kirchhoff stress, not mean Cauchy stress.
    p = J * np.trace(S, axis1=-2, axis2=-1) / 3.0
    residual = np.log(J) - p / k
    defect = 0.5 * k * residual**2
    orthogonality = p * residual
    channels = {
        "pressure_constraint_defect_energy_density": float(
            np.sum(w * defect) / reference_volume
        ),
        "pressure_orthogonality_density": float(
            np.sum(w * orthogonality) / reference_volume
        ),
        "condensed_elastic_energy_density": float(np.sum(w * e) / reference_volume),
        "primal_elastic_energy_density": float(
            np.sum(w * (e + orthogonality + defect)) / reference_volume
        ),
    }
    for name, value in channels.items():
        reference = float(expected[name])
        if not np.isfinite(reference) or not np.isclose(
            value, reference, rtol=1e-8, atol=1e-12
        ):
            raise ValueError(f"Reconstruction disagrees with accepted channel {name}.")
    cell_energy = np.sum(w * defect, axis=1) / reference_volume
    total = float(cell_energy.sum())
    centres = np.sum(x * w[..., None], axis=1) / w.sum(axis=1)[:, None]
    order = np.lexsort((cell_ids, -cell_energy))
    rows = [
        {
            "cell_id": int(cell_ids[i]),
            "quadrature_centroid": centres[i].tolist(),
            "defect_energy_per_reference_volume": float(cell_energy[i]),
        }
        for i in order
    ]
    return {
        "schema": "agentfem.zhang-2021-pressure-localization.v1",
        "benchmark_promotion_authorized": False,
        "reconstruction_verified": True,
        "channels": channels,
        "inclusion_defect_fraction": float(
            np.sum((w * defect)[phase]) / reference_volume / total
        )
        if total
        else None,
        "concentration": [
            {
                "cell_fraction": fraction,
                "cell_count": max(1, int(np.ceil(fraction * len(w)))),
                "defect_fraction": float(
                    cell_energy[order[: max(1, int(np.ceil(fraction * len(w))))]].sum()
                    / total
                )
                if total
                else None,
            }
            for fraction in (0.01, 0.05, 0.1)
        ],
        "cells_descending": rows,
        "interpretation": "Weighted pressure-defect localization; not an error bound or refinement guarantee.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--integration-points", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    candidate = load_candidate(args.candidate)["payload"]
    settings = candidate["candidate"]
    if settings["formulation"] != "2D_plane_strain_Q2_DPC1" or settings[
        "cell_repetitions"
    ] != [1, 1]:
        raise ValueError(
            "This diagnostic supports only the declared 2D single-cell fixture."
        )
    card_path = (
        Path(__file__).resolve().parents[1]
        / "src/agentfem/knowledge/benchmarks/finite_strain_j2_zhang_2021_table5.json"
    )
    card = json.loads(card_path.read_text())
    parameters = card["parameters"]
    source = settings["geometry_source"]
    if source == "figure-10a":
        geometry = parameters["figure_10a_alternative"]
    elif source == "section-3.2.1-text":
        geometry = parameters
    else:
        raise ValueError("Unknown geometry source.")
    with h5py.File(args.integration_points, "r") as archive:
        if (
            archive.attrs["schema"] != "agentfem.integration-point-dataset.v1"
            or len(archive["rules"]) != 1
        ):
            raise ValueError("Expected one explicit integration rule.")
        rule = next(iter(archive["rules"].values()))
        x = rule["coordinates"][:]
        distance = np.linalg.norm(
            x[:, :, None, :]
            - np.asarray(geometry["inclusion_centres"])[None, None, :, :],
            axis=-1,
        )
        phase = np.any(distance < parameters["diameter"] / 2, axis=-1)
        if np.any(np.any(phase, axis=1) != np.all(phase, axis=1)):
            raise ValueError(
                "Analytic phase assignment straddles a cell; explicit material tags required."
            )
        if len(x) != settings["global_cells"]:
            raise ValueError("Archive and candidate cell counts disagree.")
        bulk = parameters["matrix_bulk_modulus"] * np.where(
            phase, parameters["inclusion_stiffness_ratio"], 1.0
        )
        report = localize(
            coordinates=x,
            weights=rule["physical_weights"][:],
            gradient=rule["fields/F"][:],
            stress=rule["fields/S"][:],
            condensed=rule["fields/ELENER"][:],
            bulk=bulk,
            cell_ids=rule["cell_id"][:],
            phase=phase,
            reference_volume=settings["reference_cell_area"],
            expected=candidate["mixed_elastic_energy_diagnostics"],
        )
    report["geometry_source"] = source
    report["phase_assignment"] = (
        "Explicit benchmark analytic circles, uniform within every cell; globally checked against four accepted energy channels."
    )
    report["artifacts"] = []
    for path in (args.candidate, args.integration_points, card_path, Path(__file__)):
        with path.open("rb") as source_file:
            report["artifacts"].append(
                {
                    "path": str(path),
                    "sha256": hashlib.file_digest(source_file, "sha256").hexdigest(),
                }
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "reconstruction_verified",
                    "concentration",
                    "inclusion_defect_fraction",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
