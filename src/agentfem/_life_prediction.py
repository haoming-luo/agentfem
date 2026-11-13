"""Identity-preserving comparisons, separate from fitting and qualification."""

from collections.abc import Mapping
from math import log

import numpy as np

from ._life_observations import LifeObservation, _real, _text


def life_prediction_errors(observations, predictions, *, life_unit, model_source):
    """Compare point predictions with specimen outcomes in one explicit unit.

    ``predictions`` maps specimen IDs to positive lives. Exact errors are
    available only for observed failures. A run-out supplies a bound, not an
    error or a failed prediction. No model is fitted, threshold selected or
    independence/engineering qualification inferred.
    """
    rows = tuple(observations)
    source = _text(model_source, "model_source")
    if not rows or not all(isinstance(row, LifeObservation) for row in rows):
        raise ValueError("Require nonempty LifeObservation records.")
    ids = [row.specimen_id for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate specimen IDs are not independent observations.")
    if not isinstance(predictions, Mapping) or set(predictions) != set(ids):
        raise ValueError("Prediction IDs must exactly match observation IDs.")
    if any(row.life_unit != life_unit for row in rows):
        raise ValueError("Prediction and observation life units must match; convert explicitly.")
    if len({row.failure_definition for row in rows}) != 1:
        raise ValueError("Failure definitions must match within a comparison.")
    records, errors = [], []
    for row in rows:
        predicted = _real(predictions[row.specimen_id], "predicted life")
        if predicted <= 0:
            raise ValueError("Predicted life must be positive.")
        # Log differences remain finite even when the direct ratio overflows.
        error = log(predicted) - log(row.life)
        failure = row.status == "failure"
        if failure:
            errors.append(error)
        records.append({
            "observation": row.as_dict(), "predicted_life": predicted,
            "log_prediction_error": error if failure else None,
            "overpredicts_observed_failure": predicted > row.life if failure else None,
            "prediction_below_runout": predicted < row.life if not failure else None,
        })
    return {
        "schema": "agentfem.life-prediction-errors.v1", "model_source": source,
        "life_unit": life_unit, "records": records,
        "failure_count": len(errors), "right_censored_count": len(rows)-len(errors),
        "mean_log_error_on_failures": float(np.mean(errors)) if errors else None,
        "rms_log_error_on_failures": float(np.sqrt(np.mean(np.square(errors)))) if errors else None,
        "max_abs_log_error_on_failures": float(np.max(np.abs(errors))) if errors else None,
        "engineering_qualified": False,
        "scope": "Descriptive point-prediction errors on failures, not censor-aware model scoring or reliability",
        "validation_independence": "Not inferred; retain calibration and validation provenance externally",
    }
