"""Research-only scalar constrained recovery; no material admission."""

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
from scipy.integrate import solve_ivp

from tools import p92_recoverable_spectrum as spectrum

MODEL_ID = "research.scalar-critical-recovery-spectrum.v1"
STATE_LAYOUT = spectrum.STATE_LAYOUT
INDICES = np.array([1, 2, 4, 5])


@dataclass(frozen=True)
class Material:
    spectrum: spectrum.Material

    def __post_init__(self):
        s = self.spectrum
        if (s.strength_ratios != (1., 1.) or s.hardening_ratios != (1., 1.)
                or s.static_ratios != (1., 1.) or s.static_powers != (0., 0.)
                or s.recoverable_powers != (1., 1.) or s.directional_recovery
                or any(b.saturation_ratio != 1. for b in s.backbone.branches)):
            raise ValueError("Isolate critical recovery from other hypotheses")


def stored_energy(state, material):
    return spectrum.stored_energy(state, material.spectrum)


def coefficients(material):
    branches = material.spectrum.backbone.branches
    return tuple(np.array([v for b in branches for v in getattr(b, name)])
                 for name in ("hardening", "saturation", "static_recovery"))


def unconstrained(state, material):
    h, _, r = coefficients(material)
    flows = []
    for offset, b in zip((1, 4), material.spectrum.backbone.branches, strict=True):
        effective = state[0]-sum(state[offset:offset+2])
        flows.append(b.reference_flow*(max(abs(effective)-b.strength, 0.)/120.)**b.exponent
                     * np.sign(effective))
    return h*np.repeat(flows, 2)-r*state[INDICES], np.asarray(flows)


def rates(state, strain_rate, material, *, stress_control=False, active=None):
    h, a, r = coefficients(material)
    drive, flows = unconstrained(state, material)
    x = state[INDICES]
    if active is None:
        active = (abs(x) >= a) & (x*drive > 0.)
    result = np.zeros(12)
    result[INDICES] = np.where(active, 0., drive)
    result[[3, 6]] = abs(flows)
    s = material.spectrum
    result[10:] = (state[0]/np.asarray(s.moduli)-state[10:])/s.times
    total = sum(flows)+sum(result[10:])
    result[7] = total if stress_control else strain_rate
    result[0] = 0. if stress_control else s.backbone.modulus*(strain_rate-total)
    result[8] = state[0]*result[7]
    result[9] = sum((state[0]-sum(state[o:o+2]))*v for o, v in zip((1, 4), flows))
    result[9] += sum(x*(h*np.repeat(flows, 2)-result[INDICES])/h)
    result[9] += sum(np.asarray(s.moduli)*s.times*result[10:]**2)
    return result


def integrate(old, duration, strain_rate, material, *, stress_control=False,
              method="Radau", rtol=1e-9):
    if np.iscomplexobj(old):
        raise ValueError("Require real state")
    state = np.asarray(old, dtype=float).copy()
    if (state.shape != (12,) or not np.isfinite(state).all()
            or np.any(state[[3, 6, 9]] < 0) or type(stress_control) is not bool
            or method not in {"Radau", "BDF"}
            or not np.isfinite([duration, strain_rate, rtol]).all()
            or duration <= 0 or not 0 < rtol < 1):
        raise ValueError("Invalid critical-recovery state or controls")
    _, bounds, _ = coefficients(material)
    if np.any(abs(state[INDICES]) > bounds+1e-7):
        raise ValueError("Initial backstress outside invariant domain")
    drive, _ = unconstrained(state, material)
    active = (abs(abs(state[INDICES])-bounds) < 1e-9) & (state[INDICES]*drive > 0)
    segments, start, transitions = [], 0., []
    atol = [1e-9, 1e-9, 1e-9, 1e-13, 1e-9, 1e-9, 1e-13, 1e-13, 1e-11, 1e-11, 1e-13, 1e-13]
    while start < duration:
        events, labels = [], []
        mask = active.copy()
        for j, index in enumerate(INDICES):
            if mask[j]:
                sign = float(np.sign(state[index]))

                def release(t, y, j=j, sign=sign):
                    return sign*unconstrained(y, material)[0][j]

                release.terminal = True
                release.direction = -1
                events.append(release)
                labels.append(("release", j, sign))
            else:
                for sign in (-1., 1.):

                    def hit(t, y, j=j, index=index, sign=sign, initial=start):
                        gap = bounds[j]-sign*y[index]
                        return max(gap, 1e-12) if t == initial else gap

                    hit.terminal = True
                    hit.direction = -1
                    events.append(hit)
                    labels.append(("hit", j, sign))
        segment = solve_ivp(
            lambda t, y, mask=mask: rates(y, strain_rate, material, stress_control=stress_control, active=mask),
            (start, duration), state, method=method, rtol=rtol, atol=atol,
            dense_output=True, events=events)
        if not segment.success or not np.isfinite(segment.y).all():
            raise RuntimeError("Critical-recovery integration failed; input retained")
        segments.append(segment)
        state = segment.y[:, -1].copy()
        end = float(segment.t[-1])
        if segment.status == 0:
            break
        event_index = next(i for i, times in enumerate(segment.t_events) if len(times))
        kind, j, sign = labels[event_index]
        active[j] = kind == "hit"
        drive, flows = unconstrained(state, material)
        h, _, recovery = coefficients(material)
        root_scale = 1.+abs(h*np.repeat(flows, 2))+abs(recovery*state[INDICES])
        # Symmetric branches can encounter the same root simultaneously.
        # Update that active set together; do not integrate a zero-length
        # segment or perturb stresses to escape an event.
        if kind == "release":
            active[mask & (abs(drive) <= 1e-10*root_scale)] = False
        else:
            simultaneous = (abs(abs(state[INDICES])-bounds) <= 1e-9) & (state[INDICES]*drive > 0)
            active[simultaneous] = True
        transitions.append((end, kind, j))
        if len(transitions) > 200 or end <= start:
            raise RuntimeError("Critical recovery event stagnation; input retained")
        # Root-located endpoints only: no clipping of an accepted trajectory.
        if kind == "hit":
            if abs(state[INDICES[j]]-sign*bounds[j]) > 1e-7:
                raise RuntimeError("Critical surface root inaccurate")
        start = end
    t = np.concatenate([part.t if i == 0 else part.t[1:] for i, part in enumerate(segments)])
    y = np.concatenate([part.y if i == 0 else part.y[:, 1:] for i, part in enumerate(segments)], axis=1)
    if np.any(abs(y[INDICES]) > bounds[:, None]+1e-6):
        raise RuntimeError("Backstress invariant violated")
    accumulated = y[[3, 6, 9]]
    tolerance = 10*(np.array([1e-13, 1e-13, 1e-11])[:, None]+rtol*abs(accumulated))
    if np.any(np.maximum.accumulate(accumulated, axis=1)-accumulated > tolerance):
        raise RuntimeError("Accumulated state decreased")
    balance = state[8]-old[8]-stored_energy(state, material)+stored_energy(old, material)-state[9]+old[9]
    if abs(balance) > 1e-7+1e-7*abs(state[8]-old[8]):
        raise RuntimeError("Integrated mechanical balance failed")

    def dense(times):
        query = np.asarray(times, dtype=float)
        if not np.isfinite(query).all() or np.any(query < 0) or np.any(query > duration):
            raise ValueError("Dense query outside integrated interval")
        ends = np.array([part.t[-1] for part in segments])
        positions = np.searchsorted(ends, query.ravel(), side="left")
        output = np.empty((12, query.size))
        for i, part in enumerate(segments):
            selected = positions == i
            if np.any(selected):
                output[:, selected] = part.sol(query.ravel()[selected])
        return output[:, 0] if query.ndim == 0 else output.reshape((12, *query.shape))

    return SimpleNamespace(success=True, t=t, y=y, sol=dense, transitions=transitions,
                           nfev=sum(part.nfev for part in segments))
