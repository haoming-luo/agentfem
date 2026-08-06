# Decision 0007: Make provenance ubiquitous before adding signatures

## Decision

Every published AgentFEM result manifest carries a deterministic provenance
seal over its scientific record and registered artifacts. The core mechanism
uses SHA-256 and the Python standard library. It does not alter numerical data,
require a network service, or claim cryptographic authorship.

Each seal also retains the canonical AgentFEM origin, initiator, repository,
open-source date, license, and citation route. Modified open-source work remains
possible; deleting that local block cannot erase the independently published
chronology and leaves a conflicting provenance trail.

The stable seal identity is the future extension point for optional maintainer
signatures and transparency logs. Those services must remain outside the
finite-element solve and must not create a second result format.

Tagged Python distributions are independently covered by GitHub build
attestations. Distribution attestation and result sealing are complementary:
the former proves the official build source, while the latter detects later
changes to one simulation record and its artifacts.

## Why

Most practical provenance failures are accidental: a heavy-data file is
replaced, an output directory is partially copied, a result is paired with the
wrong metadata, or an agent trains on a stale artifact. Automatic local sealing
addresses these failures at almost zero workflow cost. Numerical watermarking
would contaminate scientific data, while mandatory signing infrastructure would
make ordinary local simulation unnecessarily fragile.

## Consequences

- Humans, agents, CI, and GUIs use one `agentfem verify` contract.
- Integrity status remains separate from scientific trust and validation.
- Legacy unsealed results remain readable and are reported honestly.
- Missing artifacts produce an incomplete seal instead of blocking result
  publication or pretending completeness.
- A future signature signs `seal_id`; it does not redesign simulation output.
