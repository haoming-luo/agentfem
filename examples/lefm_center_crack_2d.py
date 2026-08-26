"""Verify a stationary center crack with the public LEFM evidence workflow."""

from __future__ import annotations

import json

from agentfem import benchmarks


def main() -> None:
    evidence = benchmarks.center_crack_mode_i_benchmark()
    report = evidence.stress_intensity
    verification = evidence.verification
    print(
        json.dumps(
            {
                "status": evidence.status,
                "K_I": report.k_i,
                "K_II": report.k_ii,
                "J": report.j_integral,
                "path_variation": report.path_variation,
                "relative_K_error": verification.relative_k_error,
                "relative_J_error": verification.relative_j_error,
            },
            indent=2,
        )
    )
    if evidence.status != "accepted":
        raise RuntimeError("The LEFM center-crack benchmark was not accepted.")


if __name__ == "__main__":
    main()
