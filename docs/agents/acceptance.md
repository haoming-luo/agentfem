# Agent acceptance contract

An AI-native claim should be tested from the package that a user receives,
not from an editable source checkout or a maintainer's memory. AgentFEM's
deterministic acceptance route starts in a clean directory and requires the
installed wheel to expose and complete this sequence:

```text
doctor -> capabilities -> init -> check -> run -> inspect -> verify
```

The release gate exercises every installed project template through that
sequence. It also rejects a constitutive maturity claim when the benchmark
registry does not provide the minimum evidence appropriate to that claim.
Run it against a candidate distribution with:

```bash
python release_gate.py \
  --dist dist \
  --smoke \
  --report agent-acceptance.json
```

The report records the runtime fingerprint, capability discovery, evidence
audit, and result lifecycle for each template. It can be produced on Linux,
macOS, or WSL2 without changing the scientific case.

This deterministic gate proves that the machine interfaces needed by an agent
are present and coherent. A fresh-agent trial remains a separate behavioral
test: the agent must still choose an applicable model, preserve visible
scientific choices, and explain why the result is or is not supported by its
evidence. Successful execution alone is not scientific validation.
