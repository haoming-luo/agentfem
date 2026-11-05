"""Auditable within-source life calibration screen, not an engineering allowable.

One preselected monotone log-linear empirical baseline. No constitutive-law
fitting, fatigue/creep damage separation, temperature interpolation or life
extrapolation. The six training rows alone determine the coefficients.
"""

import json
from pathlib import Path
import numpy as np
from scipy.optimize import lsq_linear

SOURCE = Path(__file__).resolve().parents[1] / "evidence/creep/p92-life-table.json"


def load_table(path=SOURCE):
    data = json.loads(Path(path).read_text())
    if data.get("schema") != "agentfem.research.p92-life-table.v1":
        raise ValueError("Unknown life table schema.")
    rows = data["rows"]
    ids = [row["id"] for row in rows]
    train, test = data["calibration_ids"], data["validation_ids"]
    if (
        len(ids) != len(set(ids))
        or set(train) & set(test)
        or set(train + test) != set(ids)
    ):
        raise ValueError("Calibration/validation IDs must partition unique specimens.")
    for row in rows:
        features(row)
        if not np.isfinite(row["cycles"]) or row["cycles"] <= 0:
            raise ValueError("Observed cycles must be finite and positive.")
    return data


def features(row):
    temperature, strain, dwell = (
        row[key] for key in ("temperature_C", "strain_range", "dwell_s")
    )
    if (
        any(
            isinstance(v, (bool, np.bool_)) or np.iscomplexobj(v)
            for v in (temperature, strain, dwell)
        )
        or not np.isfinite([temperature, strain, dwell]).all()
        or temperature not in (600, 650)
        or not 0.008 <= strain <= 0.012
        or not 60 <= dwell <= 600
    ):
        raise ValueError(
            "Research domain: 600 or 650 C, total strain range .008-.012, tensile dwell 60-600 s."
        )
    return np.array(
        [
            1.0,
            np.log(strain / 0.008),
            np.log1p(dwell / 60.0),
            (temperature - 600.0) / 50.0,
        ]
    )


def calibrate(data):
    selected = [row for row in data["rows"] if row["id"] in data["calibration_ids"]]
    design = np.array([features(row) for row in selected])
    response = np.log([row["cycles"] for row in selected])
    if design.shape[0] <= 4 or np.linalg.matrix_rank(design) != 4:
        raise ValueError(
            "Require more than four calibration rows and full design rank."
        )
    fit = lsq_linear(
        design, response, bounds=([-np.inf] * 4, [np.inf, 0.0, 0.0, 0.0]), tol=1e-12
    )
    if not fit.success or not np.isfinite(fit.x).all():
        raise RuntimeError("Life screen calibration failed.")
    return fit.x.copy()


def screen(data=None):
    data = load_table() if data is None else data
    coefficient = calibrate(data)
    predictions = []
    for row in data["rows"]:
        predicted = float(np.exp(features(row) @ coefficient))
        ratio = predicted / row["cycles"]
        predictions.append(
            {
                "id": row["id"],
                "role": "calibration"
                if row["id"] in data["calibration_ids"]
                else "validation",
                "observed_cycles": row["cycles"],
                "predicted_cycles": predicted,
                "prediction_ratio": ratio,
                "error_factor": max(ratio, 1 / ratio),
            }
        )
    return {
        "schema": "agentfem.research.p92-life-screen.v1",
        "source": data["source"],
        "failure_definition": data["failure_definition"],
        "coefficients": coefficient.tolist(),
        "formula": "ln N = a + b ln(strain_range/.008) + c ln(1+dwell_s/60) + d*(T_C-600)/50",
        "calibration_ids": data["calibration_ids"],
        "validation_ids": data["validation_ids"],
        "scope": data["split_scope"],
        "source_overlap_audit": data.get("source_overlap_audit"),
        "predictions": predictions,
        "engineering_qualified": False,
        "life_semantics": "Combined creep-fatigue test life; not a pure-fatigue curve for adding creep damage again",
        "limitations": [
            "Two held-out points in one source; no independent batch",
            "No repeat-test scatter or design confidence bound",
            "No zero-dwell fatigue or independent creep-rupture separation",
            "No multiaxial, weld, oxidation or component-life validation",
            "Temperature between the two measured levels is not admitted",
        ],
    }


if __name__ == "__main__":
    print(json.dumps(screen(), indent=2, allow_nan=False))
