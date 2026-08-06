# Result Provenance Seal

Every `SimulationResult.write_manifest(...)` now adds a lightweight provenance
seal automatically. The seal binds three things:

1. the canonical scientific result record;
2. the AgentFEM version that produced it;
3. the byte content and size of every registered artifact.

The seal also carries a compact origin block naming AgentFEM, Haoming Luo, the
canonical repository, the open-source date, the Apache-2.0 license, and the
project citation file. Thus attribution travels with ordinary result bundles
rather than living only on the GitHub home page.

No account, server, key, optional dependency, or extra case code is required.
The numerical fields are never modified. A user, agent, CI job, or future GUI
can check a result directory with:

```bash
agentfem verify path/to/result.json
agentfem verify --project path/to/project --json
```

The second form follows the project's `latest.json` pointer. The command
returns one of four explicit states:

- `verified`: the manifest and every registered artifact match the seal;
- `modified`: sealed content has changed or disappeared;
- `incomplete`: the seal is consistent, but an artifact was unavailable when
  it was created;
- `unsealed`: a legacy or external manifest has no AgentFEM seal.

## Integrity is not scientific validation

The provenance seal answers, “Is this still the same recorded result?” It does
not answer, “Is the mesh adequate, did the nonlinear solve converge, or is the
model validated for this engineering claim?” Those questions remain in the
scientific verification report and its `computed`, `converged`, `verified`,
and `validated` vocabulary.

The current SHA-256 seal is deterministic integrity evidence, not proof against
an adversary who can rewrite both the result and its seal. This is deliberate:
it makes provenance ubiquitous without infrastructure or workflow cost. If
release, regulatory, or industrial use requires non-forgeable authorship, a
later optional layer can sign the stable `seal_id` with a maintainer identity
and publish it to a transparency log. Existing manifests and user commands do
not need to change.

No technical marker in openly editable source code is literally impossible to
remove. Durable credit comes from several reinforcing records: retained
license/NOTICE obligations, public Git and release history, result origin
blocks, and—later—signed release and result identities. This gives an
independent chronology and evidence chain without contaminating scientific
fields with hidden numerical watermarks.

Official tagged wheel and source distributions add the next level of this
chain: the release workflow publishes a GitHub artifact attestation before the
same files are sent to PyPI. A downloaded distribution can therefore be checked
against the canonical repository with GitHub's attestation verifier. The
attestation proves the official build origin; the result seal proves the later
integrity of a particular simulation bundle.

## Artifact discipline

Only artifacts registered on `SimulationResult` are sealed. Writers should
therefore attach XDMF, HDF5, CSV, checkpoint, report, and dataset files before
publishing the result. An intentionally deferred path is recorded as
`incomplete`; it is never silently presented as verified.

The XDMF index and HDF5 heavy-data file are separate files and both must be
registered. Their individual hashes prevent a valid index from disguising a
replaced numerical payload.

When a `SimulationResult` becomes a campaign sample, its compact software
origin block is also copied into sample provenance. This keeps attribution and
lineage attached when numerical results move from FEM into NPZ datasets and
learning workflows; it does not add AgentFEM-specific requirements to the
user's neural network.
