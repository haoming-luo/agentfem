# Digital-twin direction

AgentFEM's digital-twin opportunity starts with a strict observation contract,
not with a dashboard or an IoT connector. A useful engineering twin must keep
the relationship among physical measurements, finite-element state, learned
predictions, uncertainty, and model identity explicit throughout the asset
lifecycle.

## What AgentFEM should own

AgentFEM should own the scientific part of a future twin:

- coordinates, components, units, frames, and time identity of observations;
- deterministic FEM-to-sensor and FEM-to-grid observation operators;
- model, material, mesh, procedure, checkpoint, and result provenance;
- parameterized high-fidelity campaigns and reviewed learning datasets;
- applicability and quality evidence for surrogate predictions;
- an explicit route back to FEM when a learned model is outside its domain.

The core package should not become a SCADA, MQTT, historian, dashboard, or
asset-management product. Those systems can call AgentFEM through the same
structured project/result boundary used by a CLI, agent, or future GUI.

## Implemented foundation

The current foundation consists of:

- MPI-safe physical point and path probes;
- reusable Cartesian `ObservationGrid` coordinates;
- structured-grid FEM sampling with explicit array order and geometry masks;
- field roles, units, components, mesh policy, and `NeuralOperatorSpec`;
- parameterized campaigns, scientific datasets, PyTorch export, validation,
  applicability guards, and FEM fallback;
- transient histories, structured progress, and integrity-checked restart
  checkpoints with automatic accepted-increment cadence.

These pieces support offline twin development now: simulate an operating
envelope, sample the FEM solution exactly where sensors or a learned operator
will observe it, train externally, and retain the scientific lineage.

## Intended closed loop

The future loop is:

1. an external connector supplies timestamped, calibrated measurements;
2. an observation specification aligns sensor identity, position, component,
   unit, coordinate frame, and uncertainty with an AgentFEM model;
3. an estimator updates model parameters or internal state;
4. a guarded surrogate supplies fast in-domain predictions;
5. AgentFEM reruns the deterministic model when evidence is insufficient or
   the requested state lies outside the learned domain;
6. the twin records the updated state, uncertainty, model/checkpoint identity,
   and decision evidence.

For high-temperature plant equipment, this could connect wall-temperature and
strain measurements to thermo-mechanical stress, creep state, fatigue usage,
and inspection planning. For wave problems, sparse transient sensors could
support defect identification while the FEM model supplies controlled training
families and independent checks.

## Contracts still required

The next digital-twin-specific records should be designed before an online
service is built:

- `ObservationSpec`: sensor identity, coordinates, frame, component, unit,
  sampling time, calibration, noise, and missing-data policy;
- `TwinState`: physical timestamp, model identity, accepted parameters,
  internal-state/checkpoint identity, uncertainty, and validity domain;
- `AssimilationRun`: observations used, estimator, priors, residuals,
  identifiability checks, posterior, and rejected data;
- `PredictionRoute`: learned-model identity, applicability decision,
  uncertainty, FEM fallback, and resulting evidence.

The governing principle is that real-time speed must not erase the distinction
between measurement, inference, and deterministic computation.

## Technical references

- [NIST human-centered framework for updating digital twins](https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=936651)
- [DOLFINx finite-element expression and function evaluation](https://docs.fenicsproject.org/dolfinx/main/python/generated/dolfinx.fem.html)
- [NeuralOperator 2.0 API: grid embeddings and arbitrary-coordinate neighbor search](https://neuraloperator.github.io/dev/modules/api.html)
- [Abaqus field and history output concepts](https://docs.software.vt.edu/abaqusv2024/English/SIMACAEOUTRefMap/simaout-c-output.htm)
