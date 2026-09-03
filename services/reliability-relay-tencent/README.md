# AgentFEM mainland reliability relay

This dependency-free Tencent Cloud SCF event function provides a mainland
HTTPS ingress for AgentFEM's existing anonymous reliability channel. It is a
privacy relay, not a second telemetry system:

- the exact AgentFEM batch and event allowlists are validated before forwarding;
- function-URL headers, including `x-scf-remote-addr`, are never copied;
- no event, IP address, user identifier, model, mesh, parameter, code, path or
  result is stored by the function;
- only the validated JSON body reaches the project-owned aggregate collector;
- a failed relay returns `503`, allowing the client to use its next reviewed route.

## Tencent Cloud deployment

Create an **event function** in a mainland region with a current Node.js
runtime, upload this directory, and set the handler to `index.main_handler`.
Set one environment variable:

```text
UPSTREAM_RELIABILITY_ENDPOINT=https://agentfem-reliability.horming-luo.workers.dev/v1/reliability
```

Create an open public **Function URL** with HTTPS access. API Gateway is not
required. Do not enable request-body logging. Set a small concurrency and
invocation quota because the route is public.

Before adding the returned URL to `src/agentfem/feedback-endpoint.json`, verify:

1. `GET /health` reports `storage=none` and `forwarding=schema_only`;
2. a valid fixture returns `202`;
3. a fixture containing any undeclared field returns `400`;
4. the aggregate count changes without any raw event or IP record;
5. failure of this route causes the Python client to use the global fallback.

Run the local contract tests with:

```bash
node --test services/reliability-relay-tencent/index.test.cjs
```
