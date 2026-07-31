"""Implicit integration parameters for second-order structural dynamics."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GeneralizedAlphaParameters:
    """Parameters for Newmark/generalized-alpha time integration."""

    alpha_m: float
    alpha_f: float
    beta: float
    gamma: float
    name: str = "generalized-alpha"

    def __post_init__(self) -> None:
        if self.beta <= 0.0:
            raise ValueError("beta must be positive.")
        if not 0.0 < self.gamma <= 1.5:
            raise ValueError("gamma must lie in (0, 1.5].")
        if not 0.0 <= self.alpha_f < 1.0:
            raise ValueError("alpha_f must lie in [0, 1).")
        if not -1.0 <= self.alpha_m < 1.0:
            raise ValueError("alpha_m must lie in [-1, 1).")

    @property
    def method(self) -> str:
        return "newmark" if self.alpha_m == self.alpha_f == 0.0 else "generalized_alpha"

    def summary(self) -> dict[str, object]:
        return {
            "name": self.name,
            "method": self.method,
            "alpha_m": self.alpha_m,
            "alpha_f": self.alpha_f,
            "beta": self.beta,
            "gamma": self.gamma,
        }


def newmark(*, beta: float = 0.25, gamma: float = 0.5):
    """Average-acceleration Newmark by default."""

    return GeneralizedAlphaParameters(
        alpha_m=0.0,
        alpha_f=0.0,
        beta=beta,
        gamma=gamma,
        name="Newmark",
    )


def generalized_alpha(*, spectral_radius: float = 0.8):
    """Second-order generalized-alpha parameters from ``rho_infinity``."""

    rho = float(spectral_radius)
    if not 0.0 <= rho <= 1.0:
        raise ValueError("spectral_radius must lie in [0, 1].")
    alpha_m = (2.0 * rho - 1.0) / (rho + 1.0)
    alpha_f = rho / (rho + 1.0)
    gamma = 0.5 + alpha_f - alpha_m
    beta = 0.25 * (1.0 + alpha_f - alpha_m) ** 2
    return GeneralizedAlphaParameters(
        alpha_m=alpha_m,
        alpha_f=alpha_f,
        beta=beta,
        gamma=gamma,
        name=f"generalized-alpha rho_inf={rho:g}",
    )


__all__ = ["GeneralizedAlphaParameters", "generalized_alpha", "newmark"]
