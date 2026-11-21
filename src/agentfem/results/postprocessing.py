# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Common engineering observations over fields and retained accepted states."""
from dataclasses import dataclass
import numpy as np
from mpi4py import MPI

from .. import fields
from ..kernel import dofs
from .quantities import sample_points, _require_collective_points


@dataclass(frozen=True)
class SectionSample:
    """Point samples on a parametric plane; not a cut-surface quadrature rule.

    Grid order is (v, u), followed by the field's scalar/vector/tensor shape.
    Missing material in a porous section is retained in the coverage masks.
    """
    coordinates: np.ndarray
    values: np.ndarray
    located: np.ndarray
    finite: np.ndarray
    u: np.ndarray
    v: np.ndarray
    normal: np.ndarray

    def summary(self):
        return {"kind": "section_point_samples", "shape": list(self.located.shape),
                "located_count": int(self.located.sum()),
                "finite_count": int((self.located & self.finite).sum()),
                "normal": self.normal.tolist(), "integration": "none"}


def sample_section(field, *, origin, axis_u, axis_v, u, v, missing="raise", padding=1e-10):
    """Sample x=origin+u*axis_u+v*axis_v on a three-dimensional mesh.

    Axes are unit directions, not silently normalized; u/v have length units.
    Every rank supplies identical geometry. Use section_resultant for an actual
    mesh-region force integral; arithmetic sample means are not area averages.
    """
    function = fields.unwrap(field)
    comm = function.function_space.mesh.comm
    error = None
    try:
        if function.function_space.mesh.geometry.dim != 3:
            raise ValueError("requires a three-dimensional field")
        origin, a, b = [np.asarray(item, dtype=float) for item in (origin, axis_u, axis_v)]
        if any(item.shape != (3,) or not np.isfinite(item).all() for item in (origin, a, b)):
            raise ValueError("origin and axes require three finite components")
        if not np.isclose(np.linalg.norm(a), 1.) or not np.isclose(np.linalg.norm(b), 1.) or not np.isclose(a @ b, 0.):
            raise ValueError("axes must be orthonormal unit vectors")
        u, v = [np.asarray(item, dtype=float) for item in (u, v)]
        if any(item.ndim != 1 or len(item) < 2 or not np.isfinite(item).all()
               or np.any(np.diff(item) <= 0) for item in (u, v)):
            raise ValueError("u/v require at least two finite increasing coordinates")
    except (ValueError, TypeError) as exc:
        error = str(exc)
    errors = comm.allgather(error)
    if any(errors):
        raise ValueError("AFM-SECTION-INPUT: " + "; ".join(item for item in errors if item))
    uu, vv = np.meshgrid(u, v, indexing="xy")
    points = origin + uu[..., None] * a + vv[..., None] * b
    sampled = sample_points(function, points.reshape(-1, 3), missing=missing,
                            padding=padding, return_info=True)
    shape = uu.shape
    return SectionSample(points, sampled.values.reshape((*shape, *function.ufl_shape)),
                         sampled.located.reshape(shape), sampled.finite.reshape(shape),
                         u.copy(), v.copy(), np.cross(a, b))


@dataclass(frozen=True)
class FieldComparison:
    """Paired samples and candidate-minus-reference on common physical points."""
    coordinates: np.ndarray
    reference: np.ndarray
    candidate: np.ndarray
    difference: np.ndarray
    valid: np.ndarray
    maximum_absolute_error: float | None
    rms_error: float | None
    relative_l2_error: float | None

    def summary(self):
        return {"kind": "sampled_field_comparison", "point_count": len(self.valid),
                "valid_count": int(self.valid.sum()),
                "maximum_absolute_error": self.maximum_absolute_error,
                "rms_error": self.rms_error, "relative_l2_error": self.relative_l2_error,
                "norm": "unweighted components at requested points; not a volume L2 norm",
                "difference": "candidate - reference"}


def compare_fields(reference, candidate, *, at, missing="raise", padding=1e-10):
    """Compare scalar/vector/tensor Functions on explicit physical points.

    Different meshes are supported on the same ordered MPI communicator. No
    coefficient-array subtraction, mesh projection or unit conversion occurs.
    Field units and tensor/stress conventions must be chosen by the caller.
    Metrics use only points located and finite in both fields; coverage is
    reported, including when missing='nan' is explicitly requested.
    """
    a = fields.unwrap(reference)
    comm = a.function_space.mesh.comm
    error = None
    try:
        b = fields.unwrap(candidate)
        if a.function_space.mesh.geometry.dim != b.function_space.mesh.geometry.dim or a.ufl_shape != b.ufl_shape:
            raise ValueError("AFM-FIELD-SHAPE: field dimensions and value shapes must match.")
        if MPI.Comm.Compare(comm, b.function_space.mesh.comm) not in (MPI.IDENT, MPI.CONGRUENT):
            raise ValueError("AFM-FIELD-COMM: fields require congruent communicators.")
    except (ValueError, TypeError, AttributeError) as exc:
        error = str(exc)
    errors = comm.allgather(error)
    if any(errors):
        raise ValueError("AFM-FIELD-INPUT: " + "; ".join(item for item in errors if item))
    left = sample_points(a, at, missing=missing, padding=padding, return_info=True)
    right = sample_points(b, at, missing=missing, padding=padding, return_info=True)
    valid = left.located & right.located & left.finite & right.finite
    try:
        with np.errstate(over="raise", invalid="ignore"):
            delta = right.values - left.values
    except FloatingPointError as error:
        raise ValueError("AFM-FIELD-DATA: field difference overflows; rescale field units.") from error
    selected = delta[valid].reshape(-1)
    maximum = rms = relative = None
    if len(selected):
        scale = float(np.max(np.abs(selected)))
        maximum = scale
        rms = 0. if scale == 0 else float(scale * np.sqrt(np.mean((selected / scale)**2)))
        baseline = left.values[valid].reshape(-1)
        base_scale = float(np.max(np.abs(baseline)))
        if base_scale > 0:
            with np.errstate(over="ignore", invalid="ignore"):
                relative = float(np.linalg.norm(selected / base_scale) / np.linalg.norm(baseline / base_scale))
            if not np.isfinite(relative):
                raise ValueError("AFM-FIELD-DATA: relative error is not representable.")
    return FieldComparison(left.coordinates.copy(), left.values, right.values, delta,
                           valid, maximum, rms, relative)


@dataclass(frozen=True)
class ReactionForceMoment:
    """Discrete residual force and moment about a reference-mesh point."""
    force: np.ndarray
    moment: np.ndarray | float
    about: np.ndarray
    scalar_dof_count: int

    def as_dict(self):
        return {"force": self.force.tolist(), "moment": np.asarray(self.moment).tolist(),
                "about": self.about.tolist(), "scalar_dof_count": self.scalar_dof_count,
                "source": "unconstrained discrete residual at selected strong-constraint dofs",
                "geometry": "reference mesh coordinates"}


def reaction_force_moment(problem, *, on=None, about, name="RF"):
    """Sum residual nodal forces and (X-about) cross force, without smoothing.

    Scope matches reaction_resultant: strong Dirichlet reactions at converged
    equilibrium. This helper does not invent contact/MPC reactions. Cartesian
    2D/3D nodal vector spaces are required; in 2D the moment is a scalar and
    force/moment retain the problem's out-of-plane thickness convention.
    """
    reaction = problem.reaction_field(name=name)
    V = reaction.function_space
    domain = V.mesh
    dim = domain.geometry.dim
    error = None
    try:
        origin = np.asarray(getattr(about, "coordinates", about), dtype=float)
        if dim not in (2, 3) or reaction.ufl_shape != (dim,) or origin.shape != (dim,) or not np.isfinite(origin).all():
            raise ValueError("AFM-REACTION-GEOMETRY: requires a Cartesian vector and finite matching origin.")
        if on is not None and getattr(on, "domain", domain) is not domain:
            raise ValueError("AFM-REACTION-DOMAIN: boundary belongs to another mesh.")
    except (ValueError, TypeError) as exc:
        error = str(exc)
    errors = domain.comm.allgather(error)
    if any(errors):
        raise ValueError("; ".join(item for item in errors if item))
    _require_collective_points(origin.reshape(1, -1), domain.comm)
    values = dofs.owned_array(reaction)
    force = np.zeros(dim)
    moment = np.zeros(3)
    count = 0
    for component in range(dim):
        scalar, mapping = V.sub(component).collapse()
        # DOLFINx versions return either a flat map or one map per component.
        parent = np.asarray(mapping, dtype=np.int64).reshape(-1)
        selected = parent < len(values)
        if on is not None:
            selected &= np.isin(parent, dofs.locate_component_dofs(V, component, on))
        positions = scalar.tabulate_dof_coordinates()[selected, :dim] - origin
        loads = values[parent[selected]]
        force[component] = loads.sum()
        arm = np.zeros((len(loads), 3))
        arm[:, :dim] = positions
        vectors = np.zeros_like(arm)
        vectors[:, component] = loads
        moment += np.cross(arm, vectors).sum(axis=0)
        count += len(loads)
    force = domain.comm.allreduce(force, op=MPI.SUM)
    moment = domain.comm.allreduce(moment, op=MPI.SUM)
    return ReactionForceMoment(force, float(moment[2]) if dim == 2 else moment,
                               origin.copy(), domain.comm.allreduce(count, op=MPI.SUM))


def accepted_snapshot(step, *, index=None, at=None, coordinate=None, tolerance=1e-10):
    """Select one retained accepted frame by stored index or nearby coordinate.

    Returns the stored frame, including solution and provider-owned fields.
    No field is recomputed/interpolated and no live material state is rewound.
    With at=, tolerance is an absolute coordinate distance. Duplicate equally
    close coordinates require selection by index (e.g. a cyclic load path).
    """
    snapshots = tuple(getattr(step, "snapshots", ()))
    if not snapshots:
        raise ValueError("AFM-SNAPSHOT-MISSING: no retained accepted field snapshots.")
    if (index is None) == (at is None):
        raise ValueError("AFM-SNAPSHOT-SELECTION: supply exactly one of index or at.")
    if index is not None:
        selected = [frame for frame in snapshots if getattr(frame, "index", None) == index]
    else:
        if not np.isfinite(float(at)) or not np.isfinite(tolerance) or tolerance < 0:
            raise ValueError("AFM-SNAPSHOT-SELECTION: finite coordinate and nonnegative tolerance required.")
        key = coordinate or ("time" if all(hasattr(frame, "time") for frame in snapshots) else "load_factor")
        if key not in {"time", "load_factor"}:
            raise ValueError("AFM-SNAPSHOT-COORDINATE: select time or load_factor.")
        if not all(hasattr(frame, key) for frame in snapshots):
            raise ValueError("AFM-SNAPSHOT-COORDINATE: coordinate missing from retained frames.")
        distances = np.abs(np.asarray([getattr(frame, key) for frame in snapshots], dtype=float) - float(at))
        if not np.isfinite(distances).all():
            raise ValueError("AFM-SNAPSHOT-COORDINATE: retained coordinates must be finite.")
        nearest = float(distances.min())
        selected = [frame for frame, distance in zip(snapshots, distances, strict=True)
                    if distance == nearest and distance <= tolerance]
    if len(selected) != 1:
        raise ValueError("AFM-SNAPSHOT-SELECTION: missing or ambiguous retained frame; inspect indices and coordinates.")
    return selected[0]
