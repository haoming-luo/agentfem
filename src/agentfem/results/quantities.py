"""MPI-safe quantities of interest assembled from finite-element expressions."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from math import sqrt

import numpy as np
import basix
import ufl
from dolfinx import fem
from dolfinx import geometry as geometry_api
from mpi4py import MPI

from .. import fields as field_api
from ..kernel import dofs


@dataclass(frozen=True)
class PathSample:
    """Values sampled along one straight physical-space path."""

    coordinates: np.ndarray
    distance: np.ndarray
    values: np.ndarray
    field_name: str

    def add_to(
        self,
        result,
        *,
        name: str | None = None,
        unit: str | None = None,
        distance_unit: str | None = None,
        description: str = "",
    ):
        """Attach the path as a standard result history and return it."""

        selected_name = name or f"{self.field_name}_path"
        selected_description = description or (
            f"{self.field_name} sampled along a straight physical-space path."
        )
        return result.add_history(
            selected_name,
            self.distance,
            self.values,
            unit=unit,
            abscissa_name="distance",
            abscissa_unit=distance_unit,
            description=selected_description,
        )


@dataclass(frozen=True)
class RectilinearGridSample:
    """A finite-element field sampled on a Cartesian observation grid.

    ``shape`` follows array order: ``(ny, nx)`` in two dimensions and
    ``(nz, ny, nx)`` in three. ``inside`` distinguishes the physical domain
    from the surrounding bounding box, so irregular domains can retain honest
    ``NaN`` values without confusing them with failed in-domain evaluations.
    """

    axes: tuple[np.ndarray, ...]
    values: np.ndarray
    inside: np.ndarray
    field_name: str
    reduction: str | None = None

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.inside.shape)

    def summary(self) -> dict[str, object]:
        return {
            "kind": "rectilinear_grid_sample",
            "field": self.field_name,
            "shape": self.shape,
            "inside_points": int(np.count_nonzero(self.inside)),
            "outside_points": int(self.inside.size - np.count_nonzero(self.inside)),
            "reduction": self.reduction,
        }


@dataclass(frozen=True)
class StaticForceBalance:
    """Global algebraic force equilibrium for one linear static solid."""

    external: object
    reaction: object
    residual: object
    absolute_error: float
    relative_error: float

    def as_dict(self) -> dict[str, object]:
        def value(item):
            array = np.asarray(item)
            return float(array) if array.ndim == 0 else array.tolist()

        return {
            "external_force_resultant": value(self.external),
            "reaction_force_resultant": value(self.reaction),
            "force_balance_residual": value(self.residual),
            "absolute_error": float(self.absolute_error),
            "relative_error": float(self.relative_error),
            "definition": "reaction + assembled external force",
            "reaction_scope": "strong Dirichlet constraints",
        }


@dataclass(frozen=True)
class StaticWorkBalance:
    """Energy closure including proportional prescribed boundary motion."""

    strain_energy: float
    natural_load_work: float
    prescribed_motion_work: float
    external_work: float
    balance_error: float
    prescribed_dof_count: int

    def as_dict(self) -> dict[str, object]:
        return {
            "strain_energy": float(self.strain_energy),
            "natural_load_work": float(self.natural_load_work),
            "prescribed_motion_work": float(self.prescribed_motion_work),
            "external_work": float(self.external_work),
            "energy_balance_error": float(self.balance_error),
            "prescribed_dof_count": int(self.prescribed_dof_count),
            "path": "linear proportional ramp from zero",
            "reaction_scope": "strong Dirichlet constraints",
        }


@dataclass(frozen=True)
class ForceMomentResultant:
    """Integrated force and moment about an explicit physical point."""

    force: np.ndarray
    moment: np.ndarray | float
    about: np.ndarray
    measure: float
    source: str

    def as_dict(self) -> dict[str, object]:
        moment = np.asarray(self.moment)
        return {
            "force": np.asarray(self.force).tolist(),
            "moment": float(moment) if moment.ndim == 0 else moment.tolist(),
            "about": np.asarray(self.about).tolist(),
            "measure": float(self.measure),
            "source": self.source,
        }


def integral(expression, *, measure=ufl.dx, comm=None):
    """Return the global integral of a scalar, vector, or tensor expression."""

    shape = tuple(getattr(expression, "ufl_shape", ()))
    selected_comm = comm or _comm_from_measure(measure)
    if not shape:
        return _assemble_component(expression, measure, selected_comm)
    values = np.empty(shape, dtype=float)
    for index in np.ndindex(shape):
        values[index] = _assemble_component(
            expression[index],
            measure,
            selected_comm,
        )
    return values


def average(expression, *, measure=ufl.dx, comm=None):
    """Return the measure-weighted global average of an expression."""

    selected_comm = comm or _comm_from_measure(measure)
    volume = _assemble_component(ufl.as_ufl(1.0), measure, selected_comm)
    if volume <= 0.0:
        raise ValueError("average requires a measure with positive total weight.")
    return integral(expression, measure=measure, comm=selected_comm) / volume


def l2_norm(expression, *, measure=ufl.dx, comm=None) -> float:
    """Return ``sqrt(integral(inner(value, value)))`` globally."""

    selected_comm = comm or _comm_from_measure(measure)
    squared = _assemble_component(
        ufl.inner(expression, expression),
        measure,
        selected_comm,
    )
    return sqrt(max(0.0, squared))


def quadrature_extrema(
    expression,
    domain,
    *,
    degree: int = 4,
) -> tuple[float, float]:
    """Return global min/max sampled at Basix quadrature points.

    This is useful for bounded nonlinear diagnostics such as ``det(F)``.  It is
    a quadrature-point diagnostic, not a mathematical proof of an element-wise
    bound.
    """

    if tuple(getattr(expression, "ufl_shape", ())):
        raise ValueError("quadrature_extrema currently requires a scalar expression.")
    points, _ = basix.make_quadrature(domain.basix_cell(), int(degree))
    evaluator = fem.Expression(expression, points)
    cells = np.arange(
        domain.topology.index_map(domain.topology.dim).size_local,
        dtype=np.int32,
    )
    values = np.asarray(evaluator.eval(domain, cells), dtype=float)
    local_min = float(np.min(values)) if values.size else np.inf
    local_max = float(np.max(values)) if values.size else -np.inf
    return (
        float(domain.comm.allreduce(local_min, op=MPI.MIN)),
        float(domain.comm.allreduce(local_max, op=MPI.MAX)),
    )


def region_integral(expression, *, on):
    """Integrate a scalar, vector, or tensor over a named mesh region."""

    return integral(expression, measure=_region_measure(on))


def region_average(expression, *, on):
    """Return a measure-weighted average over a named mesh region."""

    return average(expression, measure=_region_measure(on))


def region_measure(*, on) -> float:
    """Return the global length, area, or volume of a named region."""

    measure = _region_measure(on)
    comm = _comm_from_measure(measure)
    return float(_assemble_component(ufl.as_ufl(1.0), measure, comm))


def boundary_resultant(traction, *, on):
    """Integrate a traction/flux expression over a named boundary."""

    return integral(traction, measure=_region_measure(on))


def section_resultant(stress, *, on, normal=None, about=None) -> ForceMomentResultant:
    """Integrate section force and moment from a Cauchy/nominal stress field."""

    domain = on.domain
    dimension = int(domain.geometry.dim)
    selected_normal = normal if normal is not None else ufl.FacetNormal(domain)
    traction = ufl.dot(stress, selected_normal)
    x = ufl.SpatialCoordinate(domain)
    measure = region_measure(on=on)
    if about is None:
        selected_about = np.asarray(
            average(x, measure=on.measure, comm=domain.comm), dtype=float
        )
    else:
        selected_about = np.asarray(about, dtype=float).reshape(-1)
    if selected_about.size != dimension:
        raise ValueError(f"section_resultant about requires {dimension} components.")
    reference = ufl.as_vector(tuple(float(value) for value in selected_about))
    arm = x - reference
    force = np.asarray(integral(traction, measure=on.measure, comm=domain.comm))
    if dimension == 3:
        moment_expression = ufl.cross(arm, traction)
    elif dimension == 2:
        moment_expression = arm[0] * traction[1] - arm[1] * traction[0]
    else:
        raise NotImplementedError("section_resultant supports 2D and 3D mechanics.")
    moment = integral(moment_expression, measure=on.measure, comm=domain.comm)
    return ForceMomentResultant(force, moment, selected_about, measure, "section_stress")


def free_body_resultant(
    *,
    boundary_tractions=(),
    body_forces=(),
    about,
) -> ForceMomentResultant:
    """Integrate boundary and volume forces into one free-body resultant."""

    contributions = tuple(boundary_tractions) + tuple(body_forces)
    if not contributions:
        raise ValueError("free_body_resultant requires at least one force contribution.")
    first_expression, first_region = contributions[0]
    domain = first_region.domain
    dimension = int(domain.geometry.dim)
    selected_about = np.asarray(about, dtype=float).reshape(-1)
    if selected_about.size != dimension:
        raise ValueError(f"free_body_resultant about requires {dimension} components.")
    force = np.zeros(dimension)
    moment = np.zeros(3) if dimension == 3 else 0.0
    total_measure = 0.0
    reference = ufl.as_vector(tuple(float(value) for value in selected_about))
    x = ufl.SpatialCoordinate(domain)
    for expression, region in contributions:
        if region.domain is not domain:
            raise ValueError("All free-body contributions must use the same mesh.")
        force += np.asarray(integral(expression, measure=region.measure, comm=domain.comm))
        arm = x - reference
        moment_expression = (
            ufl.cross(arm, expression)
            if dimension == 3
            else arm[0] * expression[1] - arm[1] * expression[0]
        )
        moment += integral(moment_expression, measure=region.measure, comm=domain.comm)
        total_measure += region_measure(on=region)
    return ForceMomentResultant(
        force, moment, selected_about, total_measure, "free_body_forces"
    )


def field_extrema(
    field,
    *,
    magnitude: bool = False,
    location: bool = False,
) -> dict[str, object]:
    """Return MPI-global field extrema, optionally with physical locations.

    Values are sampled at finite-element dofs. For a scalar DG0 field these are
    cell values and the reported location is the cell interpolation point. For
    a vector field ``magnitude=True`` reports nodal vector norms. Tensor
    component extrema retain the compact legacy result and do not claim one
    unambiguous physical location.
    """

    selected_field = (
        field.field
        if hasattr(field, "field") and getattr(field, "field") is not None
        else field
    )
    function = field_api.unwrap(selected_field)
    values = np.asarray(dofs.owned_array(function), dtype=float)
    shape = tuple(getattr(function, "ufl_shape", ()))
    if magnitude:
        if len(shape) != 1:
            raise ValueError("field_extrema(magnitude=True) requires a vector field.")
        components = int(shape[0])
        if values.size % components:
            raise ValueError("Vector dof storage is incompatible with its value shape.")
        values = np.linalg.norm(values.reshape(-1, components), axis=1)
    comm = function.function_space.mesh.comm
    local_min = float(np.min(values)) if values.size else np.inf
    local_max = float(np.max(values)) if values.size else -np.inf
    output = {
        "minimum": float(comm.allreduce(local_min, op=MPI.MIN)),
        "maximum": float(comm.allreduce(local_max, op=MPI.MAX)),
        "magnitude": bool(magnitude),
    }
    if not location:
        return output
    if shape and not magnitude:
        raise ValueError(
            "field_extrema(location=True) requires a scalar field or "
            "magnitude=True for a vector field."
        )

    space = function.function_space
    coordinates = np.asarray(space.tabulate_dof_coordinates(), dtype=float)
    owned_blocks = int(space.dofmap.index_map.size_local)
    coordinates = coordinates[:owned_blocks, : space.mesh.geometry.dim]
    if values.size != owned_blocks:
        raise ValueError(
            "Field storage cannot be mapped one-to-one to interpolation points."
        )
    global_ids = space.dofmap.index_map.local_to_global(
        np.arange(owned_blocks, dtype=np.int32)
    )
    element = getattr(space.element, "basix_element", None)
    is_cellwise = (
        bool(getattr(element, "discontinuous", False))
        and _space_degree(space) == 0
        and not shape
    )
    global_cells = np.full(owned_blocks, -1, dtype=np.int64)
    if is_cellwise:
        cell_map = space.mesh.topology.index_map(space.mesh.topology.dim)
        owned_cells = np.arange(cell_map.size_local, dtype=np.int32)
        cell_ids = cell_map.local_to_global(owned_cells)
        for local_cell, global_cell in zip(owned_cells, cell_ids):
            cell_dofs = space.dofmap.cell_dofs(int(local_cell))
            if len(cell_dofs) == 1 and int(cell_dofs[0]) < owned_blocks:
                global_cells[int(cell_dofs[0])] = int(global_cell)

    def candidate(index: int | None):
        if index is None:
            return None
        return (
            float(values[index]),
            int(comm.rank),
            int(global_ids[index]),
            tuple(float(value) for value in coordinates[index]),
            int(global_cells[index]) if is_cellwise else None,
        )

    local_minimum = candidate(int(np.argmin(values)) if values.size else None)
    local_maximum = candidate(int(np.argmax(values)) if values.size else None)
    minimum_candidates = [item for item in comm.allgather(local_minimum) if item]
    maximum_candidates = [item for item in comm.allgather(local_maximum) if item]
    minimum = min(minimum_candidates, key=lambda item: (item[0], item[1], item[2]))
    maximum = max(maximum_candidates, key=lambda item: (item[0], -item[1], -item[2]))
    output.update(
        {
            "sampling": "cell_values" if is_cellwise else "finite_element_dofs",
            "field_representation": (
                getattr(field, "processing", {}).get(
                    "representation", "finite_element_dofs"
                )
                if getattr(field, "processing", {})
                else "finite_element_dofs"
            ),
            "space_family": str(
                getattr(
                    element,
                    "family",
                    getattr(space.element, "family_name", "finite_element"),
                )
            ),
            "space_degree": _space_degree(space),
            "minimum_location": minimum[3],
            "maximum_location": maximum[3],
            "minimum_rank": minimum[1],
            "maximum_rank": maximum[1],
            "minimum_global_dof": minimum[2],
            "maximum_global_dof": maximum[2],
        }
    )
    if is_cellwise:
        output["entity_kind"] = "cell"
        output["minimum_global_cell"] = minimum[4]
        output["maximum_global_cell"] = maximum[4]
    return output


def reaction_resultant(
    problem,
    *,
    on=None,
    component: int | None = None,
    name: str = "RF",
):
    """Return an MPI-global strong-constraint reaction resultant.

    The problem residual is zero on converged free degrees of freedom, so its
    owned-dof sum gives strong Dirichlet reactions.  ``on=boundary`` restricts
    the sum to a named boundary; ``component=...`` returns one component. This
    is essential for displacement-controlled tests where reactions on several
    constrained boundaries would otherwise cancel globally. Affine MPC, weak,
    and contact reactions require dedicated definitions and remain outside
    this helper.
    """

    if not hasattr(problem, "reaction_field"):
        raise TypeError("reaction_resultant requires a problem with reaction_field().")
    reaction = problem.reaction_field(name=name)
    values = np.asarray(dofs.owned_array(reaction))
    shape = tuple(getattr(reaction, "ufl_shape", ()))
    comm = reaction.function_space.mesh.comm
    marker = on
    if not shape:
        selected = values
        if marker is not None:
            indices = np.asarray(
                dofs.locate_dofs(reaction.function_space, marker),
                dtype=np.int64,
            )
            indices = indices[indices < values.size]
            selected = values[indices]
        if component is not None:
            raise ValueError("Scalar reaction fields do not accept component=....")
        local = float(np.sum(selected))
        return float(comm.allreduce(local, op=MPI.SUM))
    if len(shape) != 1:
        raise NotImplementedError(
            "Reaction resultants currently require scalar or vector fields."
        )
    components = int(shape[0])
    if values.size % components:
        raise ValueError("Reaction dof storage is incompatible with its value shape.")
    if component is not None:
        selected_component = int(component)
        if not 0 <= selected_component < components:
            raise ValueError(
                f"Reaction component must lie in [0, {components - 1}]."
            )
        if marker is None:
            local = float(
                np.sum(values.reshape((-1, components))[:, selected_component])
            )
        else:
            indices = np.asarray(
                dofs.locate_component_dofs(
                    reaction.function_space,
                    selected_component,
                    marker,
                ),
                dtype=np.int64,
            )
            indices = indices[indices < values.size]
            local = float(np.sum(values[indices]))
        return float(comm.allreduce(local, op=MPI.SUM))
    if marker is None:
        local = np.sum(values.reshape((-1, components)), axis=0)
    else:
        local = np.zeros(components, dtype=values.dtype)
        for selected_component in range(components):
            indices = np.asarray(
                dofs.locate_component_dofs(
                    reaction.function_space,
                    selected_component,
                    marker,
                ),
                dtype=np.int64,
            )
            indices = indices[indices < values.size]
            local[selected_component] = np.sum(values[indices])
    global_values = np.empty_like(local)
    comm.Allreduce(local, global_values, op=MPI.SUM)
    return global_values.reshape(shape)


def external_force_resultant(problem):
    """Return the MPI-global resultant of a linear problem's assembled RHS.

    The result includes every body, boundary, and other contribution contained
    in the system force operator. It is algebraic evidence for the solved
    system, not an attempt to infer the provenance of individual load terms.
    """

    if not hasattr(problem, "system") or not hasattr(problem.system, "rhs_form"):
        raise TypeError(
            "external_force_resultant requires a linear system problem with rhs_form()."
        )
    import dolfinx.fem.petsc as fem_petsc
    from petsc4py import PETSc

    solution = problem._solution()
    vector = fem_petsc.assemble_vector(fem.form(problem.system.rhs_form()))
    try:
        vector.ghostUpdate(
            addv=PETSc.InsertMode.ADD,
            mode=PETSc.ScatterMode.REVERSE,
        )
        shape = tuple(getattr(solution, "ufl_shape", ()))
        space = solution.function_space
        owned = int(space.dofmap.index_map.size_local) * int(
            space.dofmap.index_map_bs
        )
        values = np.asarray(vector.array_r[:owned])
        comm = space.mesh.comm
        if not shape:
            return float(comm.allreduce(float(np.sum(values)), op=MPI.SUM))
        if len(shape) != 1:
            raise NotImplementedError(
                "External resultants currently require scalar or vector fields."
            )
        components = int(shape[0])
        if values.size % components:
            raise ValueError(
                "External-force storage is incompatible with its value shape."
            )
        local = np.sum(values.reshape((-1, components)), axis=0)
        global_values = np.empty_like(local)
        comm.Allreduce(local, global_values, op=MPI.SUM)
        return global_values.reshape(shape)
    finally:
        vector.destroy()


def static_force_balance(problem) -> StaticForceBalance:
    """Evaluate ``R + F = 0`` for a converged linear static solid.

    Reactions are the unconstrained residual at strong Dirichlet dofs. Affine
    MPC, weak, contact, and multiplier reactions need dedicated definitions
    and are intentionally not folded into this diagnostic.
    """

    external = external_force_resultant(problem)
    reaction = reaction_resultant(problem)
    residual = np.asarray(external) + np.asarray(reaction)
    absolute = float(np.linalg.norm(np.atleast_1d(residual)))
    scale = max(
        float(np.linalg.norm(np.atleast_1d(external))),
        float(np.linalg.norm(np.atleast_1d(reaction))),
    )
    relative = 0.0 if scale == 0.0 else absolute / scale
    if residual.ndim == 0:
        residual = float(residual)
    return StaticForceBalance(
        external=external,
        reaction=reaction,
        residual=residual,
        absolute_error=absolute,
        relative_error=relative,
    )


def static_work_balance(problem, *, constraints=()) -> StaticWorkBalance:
    """Evaluate linear-static work including nonzero strong Dirichlet data.

    Natural loads and prescribed values are assumed to ramp proportionally
    from zero.  The prescribed-motion contribution is the trapezoidal path
    integral ``0.5 * R_c dot ubar`` on uniquely constrained owned dofs.  MPC,
    weak, contact, and multiplier constraints need their own dual variables
    and are deliberately rejected by this helper.
    """

    if not hasattr(problem, "system") or not hasattr(problem, "reaction_field"):
        raise TypeError("static_work_balance requires a linear system problem.")
    from .. import constraints as constraint_api
    from .. import operators

    solution = problem._solution()
    strain = 0.5 * operators.quadratic_form(problem.system.K, solution)
    natural = 0.5 * operators.dual_product(problem.system.F, solution)
    reaction = problem.reaction_field()
    reaction_values = np.asarray(reaction.x.array)
    block_size = int(solution.function_space.dofmap.index_map_bs)
    prescribed: dict[int, float] = {}
    unsupported = []
    for item in constraint_api.dirichlet_constraints(constraints):
        bc = getattr(item, "bc", None)
        if bc is None:
            unsupported.append(type(item).__name__)
            continue
        value = getattr(item, "value", None)
        if value is None:
            value = getattr(item, "constant", None)
        if value is None:
            # A raw backend BC has no inspectable prescribed-motion contract;
            # do not silently assume that its value and work are zero.
            unsupported.append(f"{type(item).__name__}:uninspectable_value")
            continue
        dof_indices, first_ghost = bc.dof_indices()
        selected_dofs = np.asarray(dof_indices[:first_ghost], dtype=np.int64)
        if hasattr(value, "x") and hasattr(value.x, "array"):
            selected_values = np.asarray(value.x.array)[selected_dofs]
        else:
            raw = np.asarray(value.value if hasattr(value, "value") else value)
            flat = raw.reshape(-1)
            if flat.size == 1:
                selected_values = np.full(selected_dofs.shape, float(flat[0]))
            elif flat.size == block_size:
                selected_values = flat[selected_dofs % block_size]
            else:
                raise ValueError(
                    "Prescribed value shape does not match the constrained field."
                )
        for dof, selected_value in zip(selected_dofs, selected_values):
            value_float = float(selected_value)
            previous = prescribed.get(int(dof))
            if previous is not None and not np.isclose(previous, value_float):
                raise ValueError(
                    f"Conflicting prescribed values were found at local dof {dof}."
                )
            prescribed[int(dof)] = value_float
    if unsupported:
        raise NotImplementedError(
            "static_work_balance only accepts strong Dirichlet constraints; "
            f"unsupported={tuple(unsupported)}."
        )
    local_generalized = sum(
        reaction_values[dof] * value
        for dof, value in prescribed.items()
    )
    comm = solution.function_space.mesh.comm
    generalized = float(comm.allreduce(float(local_generalized), op=MPI.SUM))
    prescribed_work = 0.5 * generalized
    external = float(natural + prescribed_work)
    return StaticWorkBalance(
        strain_energy=float(strain),
        natural_load_work=float(natural),
        prescribed_motion_work=float(prescribed_work),
        external_work=external,
        balance_error=float(external - strain),
        prescribed_dof_count=int(comm.allreduce(len(prescribed), op=MPI.SUM)),
    )


def probe(field, *, at, padding: float = 1.0e-10):
    """Return one scalar, vector, or tensor field value at a physical point."""

    function = field_api.unwrap(field)
    point = _single_point(at, function.function_space.mesh)
    value = sample_points(function, point, padding=padding)[0]
    return value.item() if np.asarray(value).ndim == 0 else np.asarray(value).copy()


def sample_points(
    field,
    points,
    *,
    padding: float = 1.0e-10,
    missing: str = "raise",
) -> np.ndarray:
    """Evaluate a finite-element field at common physical points under MPI.

    Every rank must call this function with identical point coordinates.  One
    deterministic rank evaluates each point in an owned cell, then the values
    are shared collectively.  For discontinuous fields, a point exactly on an
    interelement boundary uses the lowest-rank, lowest-local-cell candidate;
    sample inside the intended cell when a one-sided value is required.
    """

    function = field_api.unwrap(field)
    domain = function.function_space.mesh
    comm = domain.comm
    selected_padding = float(padding)
    if selected_padding < 0.0:
        raise ValueError("sample_points padding must be non-negative.")
    selected_missing = str(missing).lower()
    if selected_missing not in {"raise", "nan"}:
        raise ValueError("sample_points missing must be 'raise' or 'nan'.")
    coordinates = _point_array(points, domain)
    _require_collective_points(coordinates, comm)
    point_count = int(coordinates.shape[0])
    if point_count == 0:
        shape = tuple(getattr(function, "ufl_shape", ()))
        return np.empty((0, *shape), dtype=function.x.array.dtype)

    topology = domain.topology
    owned_count = int(topology.index_map(topology.dim).size_local)
    local_cells = np.full(point_count, -1, dtype=np.int32)
    if owned_count:
        owned_cells = np.arange(owned_count, dtype=np.int32)
        tree = geometry_api.bb_tree(
            domain,
            topology.dim,
            padding=selected_padding,
            entities=owned_cells,
        )
        candidates = geometry_api.compute_collisions_points(tree, coordinates)
        collisions = geometry_api.compute_colliding_cells(
            domain,
            candidates,
            coordinates,
        )
        for index in range(point_count):
            cells = np.asarray(collisions.links(index), dtype=np.int32)
            cells = cells[cells < owned_count]
            if cells.size:
                local_cells[index] = int(np.min(cells))

    local_owner = np.where(local_cells >= 0, comm.rank, comm.size).astype(np.int32)
    owner = np.empty_like(local_owner)
    comm.Allreduce(local_owner, owner, op=MPI.MIN)
    missing_indices = np.flatnonzero(owner == comm.size)
    if missing_indices.size and selected_missing == "raise":
        listed = ", ".join(str(int(index)) for index in missing_indices[:8])
        raise ValueError(
            "sample_points could not locate point indices "
            f"[{listed}] in the mesh."
        )

    value_shape = tuple(getattr(function, "ufl_shape", ()))
    value_size = int(np.prod(value_shape, dtype=int)) if value_shape else 1
    local_values = np.zeros(
        (point_count, value_size),
        dtype=function.x.array.dtype,
    )
    selected = np.flatnonzero(owner == comm.rank)
    if selected.size:
        evaluated = np.asarray(
            function.eval(coordinates[selected], local_cells[selected])
        ).reshape(selected.size, value_size)
        local_values[selected] = evaluated
    values = np.empty_like(local_values)
    comm.Allreduce(local_values, values, op=MPI.SUM)
    if missing_indices.size:
        values[missing_indices] = np.nan
    if value_shape:
        return values.reshape((point_count, *value_shape))
    return values[:, 0]


def sample_path(
    field,
    *,
    start,
    end,
    count: int = 101,
    padding: float = 1.0e-10,
    missing: str = "raise",
) -> PathSample:
    """Sample a field along the straight segment from ``start`` to ``end``."""

    function = field_api.unwrap(field)
    domain = function.function_space.mesh
    selected_count = int(count)
    if selected_count < 2:
        raise ValueError("sample_path count must be at least two.")
    start_point = _single_point(start, domain)[0, : domain.geometry.dim]
    end_point = _single_point(end, domain)[0, : domain.geometry.dim]
    coordinates = np.linspace(start_point, end_point, selected_count)
    values = sample_points(
        function,
        coordinates,
        padding=padding,
        missing=missing,
    )
    distance = np.linalg.norm(coordinates - coordinates[0], axis=1)
    return PathSample(
        coordinates=coordinates,
        distance=distance,
        values=values,
        field_name=str(getattr(function, "name", "field")),
    )


def sample_rectilinear_grid(
    field,
    *,
    bbox,
    shape,
    reduction: str | None = None,
    component: int | None = None,
    padding: float = 1.0e-10,
) -> RectilinearGridSample:
    """Sample a scalar or vector field on a 2D/3D rectilinear grid.

    The grid may extend beyond an irregular finite-element domain. Such points
    are represented by ``NaN`` and recorded by ``inside``. Vector values may be
    reduced with ``reduction="magnitude"`` or by selecting ``component=...``.
    The operation is collective and returns the same arrays on every MPI rank.
    """

    function = field_api.unwrap(field)
    domain = function.function_space.mesh
    dimension = int(domain.geometry.dim)
    selected_shape = tuple(int(value) for value in shape)
    selected_bbox = tuple(float(value) for value in bbox)
    if dimension not in {2, 3}:
        raise ValueError("sample_rectilinear_grid supports 2D and 3D meshes.")
    if len(selected_shape) != dimension or any(value < 2 for value in selected_shape):
        raise ValueError(f"shape must contain {dimension} entries, each at least two.")
    if len(selected_bbox) != 2 * dimension:
        raise ValueError(f"bbox must contain {2 * dimension} coordinates.")
    if reduction is not None and component is not None:
        raise ValueError("Pass either reduction=... or component=..., not both.")
    selected_reduction = None if reduction is None else str(reduction).lower()
    if selected_reduction not in {None, "magnitude"}:
        raise ValueError("reduction must be None or 'magnitude'.")

    axes = tuple(
        np.linspace(selected_bbox[2 * axis], selected_bbox[2 * axis + 1], selected_shape[axis])
        for axis in range(dimension)
    )
    indexing_axes = tuple(reversed(axes))
    grids = np.meshgrid(*indexing_axes, indexing="ij")
    points = np.zeros((int(np.prod(selected_shape)), 3), dtype=float)
    for axis in range(dimension):
        points[:, axis] = grids[dimension - 1 - axis].reshape(-1)

    raw = np.asarray(sample_points(function, points, padding=padding, missing="nan"))
    field_shape = tuple(getattr(function, "ufl_shape", ()))
    if field_shape:
        inside = np.all(np.isfinite(raw.reshape(raw.shape[0], -1)), axis=1)
        if component is not None:
            selected_component = int(component)
            if selected_component < 0 or selected_component >= raw.shape[1]:
                raise ValueError(
                    f"component={selected_component} is outside vector size {raw.shape[1]}."
                )
            raw = raw[:, selected_component]
        elif selected_reduction == "magnitude":
            raw = np.linalg.norm(raw, axis=1)
    else:
        inside = np.isfinite(raw)

    array_shape = tuple(reversed(selected_shape))
    trailing = tuple(raw.shape[1:])
    values = raw.reshape((*array_shape, *trailing))
    return RectilinearGridSample(
        axes=axes,
        values=values,
        inside=inside.reshape(array_shape),
        field_name=str(getattr(function, "name", "field")),
        reduction=selected_reduction if component is None else f"component_{component}",
    )


def _assemble_component(expression, measure, comm) -> float:
    local = fem.assemble_scalar(fem.form(expression * measure))
    return float(comm.allreduce(local, op=MPI.SUM))


def _single_point(point, domain) -> np.ndarray:
    selected = np.asarray(point, dtype=domain.geometry.x.dtype)
    if selected.ndim != 1:
        raise ValueError("A probe point must be one coordinate vector.")
    return _point_array(selected.reshape(1, -1), domain)


def _point_array(points, domain) -> np.ndarray:
    selected = np.asarray(points, dtype=domain.geometry.x.dtype)
    if selected.ndim == 1 and domain.geometry.dim == 1:
        selected = selected.reshape(-1, 1)
    if selected.ndim != 2:
        raise ValueError("Point coordinates must have shape (number, dimension).")
    geometric_dimension = int(domain.geometry.dim)
    storage_dimension = int(domain.geometry.x.shape[1])
    if selected.shape[1] == storage_dimension:
        return np.ascontiguousarray(selected)
    if selected.shape[1] != geometric_dimension:
        raise ValueError(
            "Point coordinates must match the mesh geometric dimension "
            f"{geometric_dimension}."
        )
    coordinates = np.zeros(
        (selected.shape[0], storage_dimension),
        dtype=domain.geometry.x.dtype,
    )
    coordinates[:, :geometric_dimension] = selected
    return coordinates


def _require_collective_points(points: np.ndarray, comm) -> None:
    digest = sha256(np.ascontiguousarray(points).tobytes()).hexdigest()
    identity = (tuple(int(value) for value in points.shape), digest)
    identities = comm.allgather(identity)
    if any(item != identities[0] for item in identities[1:]):
        raise ValueError(
            "sample_points requires identical coordinates on every MPI rank."
        )


def _space_degree(space) -> int:
    degree = getattr(space.element, "degree", None)
    if degree is None:
        degree = space.ufl_element().degree
        if callable(degree):
            degree = degree()
    if isinstance(degree, tuple):
        degree = max(degree)
    return int(degree)


def _comm_from_measure(measure):
    domain = measure.ufl_domain()
    cargo = None if domain is None else domain.ufl_cargo()
    comm = getattr(cargo, "comm", None)
    if comm is None:
        raise ValueError(
            "Could not infer an MPI communicator from the integration measure; "
            "pass comm=... explicitly."
        )
    return comm


def _region_measure(region):
    measure = getattr(region, "measure", None)
    if measure is None:
        raise ValueError("A named cell or boundary region with a measure is required.")
    return measure
