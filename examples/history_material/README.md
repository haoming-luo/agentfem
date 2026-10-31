# Local history material → structural simulation

This educational project keeps `case.py` as the engineering model and
`material.py` as two local constitutive updates. There is no project Newton
loop, quadrature manager, Step provider, mesh writer or checkpoint adapter.
Use an AgentFEM build exposing `mechanics.FiniteStrainMaterialStep(checkpoint=...)` and
`constitutive.PlaneStressReduction`; an older
development package with the same version string is not sufficient.

```sh
agentfem check
agentfem run --name viscous -- --material viscoelastic
agentfem run --name damage -- --material damage
agentfem run --name plane-stress -- --material viscoelastic --assumption plane_stress
```

Both runs use the same finite-strain plane-strain plate, hold/reversal path,
physical duration and framework solver. Only the local material changes.
The optional `--assumption plane_stress` retains the same structure and local
material, adding an explicit reflection-symmetric reduction. It solves positive
thickness stretch from transverse equilibrium and condenses the full discrete
material tangent. It is not a truncated plane-strain stiffness or a shell.
Forces and energies in these 2D examples are per unit reference thickness.
These are objective Green-strain reference laws with synthetic constants,
not calibrated hydrogel models or robust large-strain rubber models. The
damage law is a local demonstration, not a mesh-objective fracture model.

Open the run's `results.xdmf` in ParaView. Choose `Displacement` in Warp By
Vector (scale 1) and color by `S_CELL` or a state field. The reference mesh,
displacement and recovered cell fields share one dataset at each saved time.
Keep the accompanying `results-data` directory; all HDF5 references are
relative. `results.xdmf` is updated after each accepted frame and refers only
to closed immutable HDF5 files, so it can be opened/refreshed during the run.
Topology/reference geometry are stored once. One frame file per saved station
does not duplicate the reference mesh; checkpoints remain separate artifacts.

The example writes 16 accepted frames unless additional declared path knots
are needed. Failure never fabricates missing frames. `final.npz` is the full
same-layout serial restart; a resumed run writes a new visualization segment,
not a second copy of unavailable old fields. The core continuation view can
join compatible segments without copying heavy data.

## Pause, inspect, and continue

In a fresh project copy, these two ordinary runs pause halfway and then finish:

```sh
agentfem run --name pause -- --material viscoelastic --assumption plane_stress --until 0.5
agentfem run --name resume -- --material viscoelastic --assumption plane_stress --restart outputs/001-pause/accepted.npz
```

Use the actual numbered pause directory if other runs already exist. The first
run publishes eight frames and `accepted.npz`; its result is explicitly
`partial`, not a completed physical path. `0.5` is the normalized path position
(physical time 1 s in this example), not half a second. The command itself exits
successfully because the requested pause was reached. Inspect its `results.xdmf`
immediately, without waiting for continuation. The second run publishes the
restored boundary plus the remaining eight new frames and `final.npz`. A joined
view verifies and deduplicates the shared boundary; it has 16 unique frames.
Keep the same material, assumption,
parameters and mesh when restoring. Pausing on an existing station preserves
the exact original increment partition; inserting an extra station can change
the time-discretized response of a rate-dependent material.

The framework also saves an accepted-state checkpoint every four accepted
increments under the run's `checkpoints/` directory, keeping the latest two
scheduled saves. A planned pause or normal completion saves its endpoint even
off that cadence. `accepted.npz`/`final.npz` is an additional convenient final
archive, not a replacement for intermediate recovery. On unexpected failure,
restart from the latest successfully written `.npz` under `checkpoints/`;
unsaved increments must be recomputed. Failed trial states are never saved.
Numerical acceptance precedes writing; a disk error raises without numerical
cutback or deletion of the previous good archive. Cadence continues from the
restored accepted-increment count. A new run writes to its own directory and
does not delete previous runs' checkpoints. No checkpoint callback is needed.
To join compatible output segments without copying HDF5 data, use
`results.join_xdmf_segments`; this does not recreate unsaved frames.
When restarting from an older scheduled archive, old output can extend beyond
the restart time. Use its explicit `overlap="identical"` option to verify and
deduplicate the recomputed interval. Conflicting fields or unmatched overlapping
times are rejected; never concatenate different branches by ignoring warnings.

## Check a local update before solving the structure

With a build exposing `constitutive.check_material_rotation`, the same public
tools can check either local law without running a global solve. Run this Python
snippet from the example directory:

```python
import numpy as np
from agentfem import constitutive as c
from material import HistoryMaterial

material = HistoryMaterial(mechanism="viscoelastic")  # or "damage"
point = c.MaterialPointInput(
    np.eye(3), np.diag((1.1, 0.98, 1.02)), 0.1, 0.1, [],
    material.state_schema.initial_state(), material.state_schema,
)
Q = np.array(((0., -1., 0.), (1., 0., 0.), (0., 0., 1.)))
derivative = c.check_material_tangent(material, point)
rotated = c.check_material_rotation(
    material, point, Q,
    state_frames={v.name: "invariant" for v in material.state_schema.variables},
)
print(derivative.summary())
print(rotated.summary())
assert derivative.accepted and rotated.accepted
```

Here `Q` history is a reference Green-strain tensor, `KAPPA` a scalar and `AGE`
physical elapsed time; all are invariant under the selected observer rotation.
Do not copy that declaration to spatial history tensors in a different law.
Repeat at unloading/hold states and use meaningful characteristic scales near
zero responses. These checks complement, not replace, independent material and
structural references. They add no checks to each production integration point.

## Scientific and numerical boundaries

- `duration=2.` is physical time in the user's consistent units. The amplitude
  argument is the normalized coordinate 0–1. Cutback subdivides both.
- Every candidate evaluates from committed old F and state. Only accepted
  equilibrium advances material history. Initialization/restart do not invoke
  the constitutive update with fictitious time.
- Material stress is Cauchy; the tangent is the declared reference 9×9 dP/dF,
  including the discrete history update. The driver converts stress to P.
- Plane stress requires `constitutive.PlaneStressReduction` and a 3D law whose
  history preserves the declared reflection symmetry. `F_CELL[2,2]` records
  thickness stretch; `S_CELL` is actual 3D Cauchy stress, not thickness-scaled.
  Positive transverse algorithmic stiffness is required. General director/
  transverse-shear reduction, bending shells and singular softening are not
  admitted. Scalar local solves currently use the material's `update` method;
  no batch speedup is claimed for this reduction.
- No automatic plane-stress inference, contact/MPC, MPI, temperature-history input or
  mixed pressure formulation is admitted by this experimental route.
- Raw integration-point fields remain in `SimulationResult`; `*_CELL` is an
  explicitly recovered cell average, not a nodal stress or raw quadrature plot.
- Optional stored energy is not total mechanical work. Missing energy stays
  unavailable; convergence is not independent scientific validation.

The regression tests exercise material tangents, rotations, uniform structural
patches, nonuniform dead-load equilibrium, assembled tangent directions,
physical-duration sensitivity, cutback, atomic failures/restart and real VTK
reading/warping. See `docs/history_material_workflow.md` for the public contract.
