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
agentfem telemetry route auto       # or: global / china
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
and delivery has a 1.5-second total network budget followed by exponential
backoff. Network failure never fails a simulation. The event does not
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

## Independent delivery routes

The packaged endpoint manifest can declare more than one reviewed HTTPS route.
`auto` uses the declared order and remembers the last successful route;
`global` or `china` moves that reviewed route first. AgentFEM does not infer a
user's location from an IP address, locale, or model path. One batch stops after
the first accepted response, so failover does not intentionally duplicate it.

The global Cloudflare route and Tencent Cloud route are independent aggregate
collectors. Both validate the same exact schema and immediately reduce valid
events to daily counters. The Tencent function never reads or stores Function
URL headers such as the transport address, and its private COS bucket accepts
only aggregate objects through an SCF runtime role. Route failure changes only
delivery; it never changes, enriches, or reinterprets a scientific result. The
deployable Tencent reference and contract tests live in
`services/reliability-relay-tencent/`.
