"""Reconstruct all source-table cases using independently read density curves."""

import json
import math
from pathlib import Path

from tools.p92_creep_equation_reference import integrate
from tools.p92_prior_damage_audit import audit

DENSITY_SOURCE = Path(__file__).resolve().parents[1] / "evidence/creep/p92-prior-density-figure.json"


def screen():
    density = json.loads(DENSITY_SOURCE.read_text())
    observations = audit()
    readings = {row[0]: row for row in density["rows"]}
    if len(readings) != len(density["rows"]) or set(readings) != set(range(2, 16)):
        raise ValueError("Density readings must cover every prior-loaded source row once")
    records = []
    for row in observations["records"]:
        if row["id"] == 1:
            ratios = [1.0] * 3
        else:
            _, panel, curve, x, y = readings[row["id"]]
            axes = density["panels"][panel]
            fraction = (x - axes["x_zero"]) / (axes["x_100percent"] - axes["x_zero"])
            if abs(fraction - row["prior_fraction"]) > 0.005:
                raise ValueError("Figure x coordinate disagrees with prior-life fraction")
            pixels_per_density = (axes["y_density_one"] - axes["y_density_one_point_five"]) / 0.5
            ratios = [(1 + (axes["y_density_one"] - y - offset) / pixels_per_density)
                      * density["density_unit_m2"] / density["rho0_m2"]
                      for offset in (0, -density["reading_uncertainty_px"], density["reading_uncertainty_px"])]
        if not all(math.isfinite(ratio) and 0 < ratio <= 1 for ratio in ratios):
            raise ValueError("Density ratio outside the source model domain")
        results = [integrate(lath_um=row["lath_um"], density_ratio=ratio)["time_h"] for ratio in ratios]
        records.append(dict(table_id=row["id"], density_ratio=ratios[0],
                            computed_h=results[0], reading_sensitivity_h=[min(results), max(results)],
                            published_prediction_h=row["published_prediction_h"], observed_h=row["observed_h"],
                            relative_to_published=results[0] / row["published_prediction_h"] - 1,
                            relative_to_observed=results[0] / row["observed_h"] - 1))
    return dict(schema="agentfem.research.p92-prior-equation-screen.v1", records=records,
                source=density["source"], source_sha256=density["sha256"],
                density_origin="digitized author-predicted curves, not measured specimen values",
                parameters="fixed Table 2; equation 9 stress-scaled backstress",
                engineering_qualified=False, fitted=False, independent_experimental_validation=False,
                limitations=density["limitations"])


if __name__ == "__main__":
    print(json.dumps(screen(), indent=2, allow_nan=False))
