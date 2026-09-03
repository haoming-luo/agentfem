# Reliability Feedback and Private Support

AgentFEM uses a minimal anonymous reliability signal to improve the free,
open-source software. It learns which platforms and workflows succeed or fail
in real use while remaining independent of the finite-element model and
numerical result lifecycle.

## What basic reporting contains

Basic reporting is on by default and can be disabled at any time:

```bash
agentfem telemetry status
agentfem telemetry off
agentfem telemetry on
```

Turning reporting off also deletes every unsent automatic event. Local
scientific results and user-created support bundles are unaffected.

One event contains only:

- AgentFEM, Python, DOLFINx and petsc4py versions;
- operating-system family, CPU architecture and AgentFEM platform route;
- MPI vendor and rank count;
- CLI command, completed/failed status and a coarse duration bucket;
- for a failure, its stable AFM code, execution stage, exception class and an
  anonymous failure fingerprint.

It never contains model definitions, meshes, material parameters, source
code, project or file names, filesystem paths, result values or fields,
exception messages, or tracebacks. Events are capped at 8 KiB, a maximum of
64 events is queued locally, successful use is sampled at most once per day,
and network failure never delays or fails a simulation. The event does not
contain the client's activity timestamp; the collector supplies only the UTC
day used for aggregate reliability counts.

Use `agentfem telemetry show-last` to inspect the exact latest event, including
after successful delivery. This local snapshot follows the same strict schema;
`agentfem telemetry off` deletes it together with every unsent event. If the
first project-owned endpoint is temporarily unavailable, AgentFEM tries the
next reviewed route within the same bounded delivery budget. It sends an event
through the first successful route only. If all routes are unavailable,
AgentFEM keeps only the bounded local queue and never fails a simulation.

## Repeated failure and Agent assistance

Failure repetition is counted locally by a message-free fingerprint. After
the same failure occurs three times in one AgentFEM version, the CLI suggests:

```bash
agentfem diagnose
agentfem assist --force
```

`diagnose` explains the latest execution locally. `assist` creates a private
directory containing a task for Codex or another AI agent, sanitized execution
structure, runtime inventory and integrity manifest. Nothing is uploaded.

For a report that the user can inspect and send manually:

```bash
agentfem feedback --output agentfem-feedback.zip
```

Creating a GitHub issue is a separate, explicit action and requires an
authenticated GitHub CLI:

```bash
agentfem feedback --github
```

AgentFEM first searches for an open issue carrying the same fingerprint to
reduce duplicates. The issue contains only the sanitized report; confidential
model evidence must never be attached automatically.

This escalation benefits the reporting user as well as the project: one
confirmed report can lead directly to a diagnosis, bug fix, clearer message,
or generally useful capability. It is never an automatic extension of basic
reporting.

## Collector boundary

The reference collector in `services/reliability-collector/` validates the
same strict schema and stores daily aggregate counts. Random delivery IDs are
retained for seven days only for retry idempotency. It does not store raw
events, source IP addresses, user identities, exception text, or scientific
data. The 0.3.1 endpoint is project-owned and passed health, rejection and
aggregation smoke tests before publication; an unowned placeholder remains
forbidden.

## Global and mainland-China ingress

The packaged endpoint manifest can declare more than one reviewed HTTPS route.
The global route reaches the aggregate collector directly. The mainland-China
route is a Tencent Cloud SCF privacy relay: it validates the exact same schema,
does not store a request, and forwards only the validated JSON body. Function
URL headers, including the transport address supplied by SCF, are not copied.
The aggregate collector therefore sees the relay rather than the originating
client.

The relay is intentionally not a second analytics implementation. Client,
relay, and collector retain one event schema and one privacy boundary. Route
failure changes only delivery; it never changes, duplicates, enriches, or
reinterprets an event. The deployable reference and contract tests live in
`services/reliability-relay-tencent/`.
