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
are present and coherent. It does **not** impersonate an unfamiliar AI agent.
A fresh-agent trial remains a separate behavioral test: the agent must choose
an applicable model, preserve visible scientific choices, and explain why the
result is or is not supported by its evidence. Successful execution alone is
not scientific validation.

Prepare an immutable trial bundle from the exact release candidate first:

```bash
python tools/prepare_agent_trial.py \
  --wheel dist/agentfem-*.whl \
  --output fresh-agent-trial
```

The bundle contains one bounded mechanics task, an empty project directory,
the exact wheel, its SHA-256 digest, the source commit and an independent
review checklist. Give `TASK.md` and the bundle to a genuinely fresh agent;
the maintainer who developed the candidate must not silently complete or
repair the project.

After a genuinely fresh task has completed, retain its transcript and final
scientific explanation beside the project, then record the trial with:

```bash
python tools/agent_trial_acceptance.py \
  --project fresh-agent-project \
  --agent "Codex/<model identity>" \
  --transcript fresh-agent-project/agent-transcript.md \
  --explanation fresh-agent-project/explanation.md \
  --fresh-context --human-interventions 0 --reviewed-explanation \
  --source-commit <commit-from-trial-contract> \
  --wheel fresh-agent-trial/agentfem-*.whl \
  --contract fresh-agent-trial/trial-contract.json \
  --report fresh-agent-project/agent-trial-acceptance.json
```

The recorder independently reruns `doctor`, `capabilities`, `check`, `inspect`
and `verify`. It refuses a source checkout, inherited project context, missing
transcript, human repair intervention, unverified result, or unreviewed
explanation. The reviewer confirms scientific adequacy; the recorder never
pretends that prose quality can be inferred from a successful solve.
Candidate version, source commit, wheel digest, transcript digest and
explanation digest are retained and cross-checked against the immutable trial
contract, so an older or substituted successful trial cannot promote a newer
release candidate.
