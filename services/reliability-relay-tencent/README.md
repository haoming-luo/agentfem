# AgentFEM Tencent reliability collector

This dependency-light Tencent Cloud SCF function is the independent regional
route for AgentFEM's privacy-bounded reliability channel. It is not a proxy to
the global Cloudflare collector.

- the exact AgentFEM batch and event allowlists are validated before use;
- request headers and source addresses are never passed to storage;
- models, meshes, parameters, source, paths, messages, tracebacks and results
  are not accepted by the schema;
- valid events are immediately reduced to daily counters in a private COS
  object;
- random delivery IDs are retained only inside the daily record for
  idempotency; raw event records are not stored;
- a storage failure returns `503`, allowing the client to retain its local
  queue and try another reviewed route later.

## Tencent Cloud deployment

Create an event function in the same region as a private COS bucket. The
single-file function uses Node.js built-ins and Tencent COS V5 request signing,
so no runtime package or long-lived credential is bundled. Use a current
Node.js runtime and set the handler to `index.main_handler`. Configure:

```text
AGGREGATE_BUCKET=agentfem-reliability-aggregate-<APPID>
AGGREGATE_REGION=ap-guangzhou
```

Assign an SCF runtime role with only `GetObject` and `PutObject`
on this prefix:

```text
qcs::cos:ap-guangzhou:uid/<APPID>:agentfem-reliability-aggregate-<APPID>/daily/*
```

Keep the bucket private, disable access logging for this route, and cap the
function at one concurrent instance because a daily aggregate is updated with
a read-modify-write transaction. The endpoint itself stores no credential;
SCF supplies short-lived role credentials at runtime. The implementation also
caps daily delivery IDs and aggregate dimensions so an unauthenticated public
route cannot grow one COS object without bound.

Create an HTTPS Function URL without authentication. Before adding it to the
packaged endpoint manifest, verify:

1. `GET /health` reports `storage=daily_aggregates_only`;
2. a valid fixture returns `202` and increments one daily counter;
3. delivering the same event twice increments the duplicate count, not the
   event count;
4. a fixture containing any undeclared field returns `400`;
5. the private COS record contains counters and delivery IDs, but no source
   address, request header, or raw event object;
6. a disabled Tencent route leaves the Python client's local queue intact.

Run the local contract tests with:

```bash
node --test services/reliability-relay-tencent/index.test.cjs
```
