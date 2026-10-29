# Local history material → structural simulation

This educational project keeps `case.py` as the engineering model and
`material.py` as two local constitutive updates. There is no project Newton
loop, quadrature manager, Step provider, mesh writer or checkpoint adapter.
Use an AgentFEM build exposing `mechanics.FiniteStrainMaterialStep`; an older
development package with the same version string is not sufficient.

```sh
agentfem check
agentfem run --name viscous -- --material viscoelastic
agentfem run --name damage -- --material damage
```

Both runs use the same finite-strain plane-strain plate, hold/reversal path,
physical duration and framework solver. Only the local material changes.
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

## Scientific and numerical boundaries

- `duration=2.` is physical time in the user's consistent units. The amplitude
  argument is the normalized coordinate 0–1. Cutback subdivides both.
- Every candidate evaluates from committed old F and state. Only accepted
  equilibrium advances material history. Initialization/restart do not invoke
  the constitutive update with fictitious time.
- Material stress is Cauchy; the tangent is the declared reference 9×9 dP/dF,
  including the discrete history update. The driver converts stress to P.
- No plane-stress inference, contact/MPC, MPI, temperature-history input or
  mixed pressure formulation is admitted by this experimental route.
- Raw integration-point fields remain in `SimulationResult`; `*_CELL` is an
  explicitly recovered cell average, not a nodal stress or raw quadrature plot.
- Optional stored energy is not total mechanical work. Missing energy stays
  unavailable; convergence is not independent scientific validation.

The regression tests exercise material tangents, rotations, uniform structural
patches, nonuniform dead-load equilibrium, assembled tangent directions,
physical-duration sensitivity, cutback, atomic failures/restart and real VTK
reading/warping. See `docs/history_material_workflow.md` for the public contract.
