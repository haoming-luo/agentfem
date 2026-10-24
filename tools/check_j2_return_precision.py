# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Optional mpmath precision probe; never part of the solver or performance path."""

import argparse
from hashlib import sha256
import agentfem
from agentfem.checkpointing import atomic_write_text
from pathlib import Path
import json
import numpy as np
import mpmath as mp
from agentfem import constitutive

mp.mp.dps = 80
def rot(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array(((c, -s, 0.), (s, c, 0.), (0., 0., 1.)))

def matrix(a):
    return mp.matrix(np.asarray(a).tolist())

def oracle(f, fp, alpha, material):
    f = matrix(f)
    mu, bulk, y, h = map(mp.mpf, (material.shear_modulus, material.bulk_modulus,
                                material.yield_stress, material.hardening_modulus))
    u, s, vt = mp.svd(f * fp**-1)
    logs = [mp.log(x) for x in s]
    volume = sum(logs)
    dev = mp.matrix([2 * mu * (x - volume / 3) for x in logs])
    q = mp.sqrt(mp.mpf('1.5') * sum(x*x for x in dev))
    trial = q - y - h * alpha
    # Same declared binary64 branch tolerance; precision alone changes.
    threshold = mp.mpf(64 * np.finfo(float).eps) * max(mp.mpf(material.young), y, q)
    if trial > threshold:
        increment = trial / (3 * mu + h)
        dev *= max(mp.mpf(0), 1 - 3 * mu * increment / q)
        logs = [volume / 3 + x / (2 * mu) for x in dev]
        # Reconstruct and invert in arbitrary precision, not the optimized
        # inverse-SVD expression under test.
        stretch = u * mp.diag([mp.exp(x) for x in logs]) * u.T
        fp = (stretch * (u * vt))**-1 * f
        alpha += increment
    tau = u * mp.diag([bulk * volume + x for x in dev]) * u.T
    sigma, piola = tau / mp.det(f), tau * (f**-1).T
    return fp, alpha, {"cauchy_stress": np.array(sigma.tolist(), float),
                       "first_piola_stress": np.array(piola.tolist(), float)}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a fresh evidence file.")
    report = {"scope": "Selected ill-conditioned cyclic history points; high-precision arithmetic reference, not performance or external physics validation",
              "mpmath": mp.__version__, "decimal_digits": mp.mp.dps, "numpy": np.__version__,
              "points": []}
    stretches = np.geomspace(1., 100., 1025)
    for hardening in (0., 100.):
        material = constitutive.finite_strain_j2_logarithmic(
            young=1000., poisson=.3, yield_stress=2., hardening_modulus=hardening)
        for point in (0, 512, 1022, 1024):
            s = stretches[point]
            plastic = rot(.37) @ np.diag((s, 1/s, 1.)) @ rot(-.19)
            scalar = material.state_schema.initial_state().copy()
            scalar[:9] = plastic.ravel()
            scalar[9] = 2*np.log(s)
            scalar[10] = material.yield_stress * scalar[9]
            batch = scalar.copy()
            fp, alpha = matrix(plastic), mp.mpf(scalar[9])
            records = []
            for index, shear in enumerate((0., .08, .02, -.10, -.02, .12)):
                elastic = np.eye(3) + np.array(((0., shear, 0.), (0., 0., 0.), (0., 0., 0.)))
                f = (rot(.23 * index) @ elastic) @ plastic
                got = material._integrate_batch(f[None], batch[None])
                old = material._integrate(f, scalar)
                fp, alpha, reference = oracle(f, fp, alpha, material)
                row = {"station": index, "condition_gradient": float(np.linalg.cond(f)), "fields": {}}
                for name, exact in reference.items():
                    a, b = got[name][0], getattr(old, name)
                    row["fields"][name] = {
                        "reference_frobenius_norm": float(np.linalg.norm(exact)),
                        "scalar_frobenius_error": float(np.linalg.norm(b-exact)),
                        "batch_frobenius_error": float(np.linalg.norm(a-exact)),
                        "scalar_batch_frobenius_difference": float(np.linalg.norm(a-b)),
                        "reference_tensor": exact.tolist(), "scalar_tensor": b.tolist(), "batch_tensor": a.tolist(),
                    }
                records.append(row)
                scalar, batch = old.state, got["state"][0]
            report["points"].append({"point_index": point, "hardening": hardening,
                                     "initial_plastic_condition": float(np.linalg.cond(plastic)),
                                     "stations": records})

    report["material_source_sha256"] = sha256((Path(agentfem.__file__).parent / "constitutive/finite_strain_plasticity.py").read_bytes()).hexdigest()
    report["tool_sha256"] = sha256(Path(__file__).read_bytes()).hexdigest()
    report["precision_scope"] = "Binary64 F/material parameters preserved exactly at input; independent 80-digit scalar reconstruction and separately advanced histories. First Piola errors are amplified by F^-T; they do not have Cauchy stress scale."
    for point in report["points"]:
        for station in point["stations"]:
            for name, item in station["fields"].items():
                if name == "cauchy_stress":
                    bound = 2e-9 * item["reference_frobenius_norm"] + 2e-9
                    assert item["scalar_frobenius_error"] < bound
                    assert item["batch_frobenius_error"] < bound
                for tensor in ("reference_tensor", "scalar_tensor", "batch_tensor"):
                    del item[tensor]
    atomic_write_text(args.output, json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "points": len(report["points"]), "dps": mp.mp.dps}))


if __name__ == "__main__":
    main()
