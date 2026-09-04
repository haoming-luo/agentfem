# AgentFEM reliability collector

This optional Cloudflare Worker receives only AgentFEM's strict basic-event
schema. It stores daily aggregate counts. Random event IDs are retained for
seven days solely to make retries idempotent; raw events, source IPs, user IDs,
paths, exception messages, models, meshes, parameters, and results are not
stored.

Deployment is deliberately separate from package publication:

1. create a D1 database;
2. apply `schema.sql`;
3. copy `wrangler.jsonc.example` to the deployment configuration and insert
   the database ID;
4. deploy behind a project-owned HTTPS Workers URL or custom domain;
5. set that reviewed URL in `src/agentfem/feedback-endpoint.json` before a
   release.

The manual GitHub workflow `Deploy reliability collector` performs steps 2--4
after these repository secrets are configured:

- `CLOUDFLARE_API_TOKEN`;
- `CLOUDFLARE_ACCOUNT_ID`;
- `CLOUDFLARE_D1_DATABASE_ID`.

The deployment workflow never edits the Python package automatically. The
returned HTTPS URL must first pass `/health` and a schema-rejection test, then
be reviewed in an ordinary pull request before it becomes a package default.

Until step 5 is complete, clients keep a bounded local queue and report
`delivery_available: false`. A release must never contain a guessed or
unowned endpoint.

The reviewed 0.3.1 deployment is available at
`https://agentfem-reliability.horming-luo.workers.dev`. Its client submission
route is `/v1/reliability`; `/health` exposes only the aggregate-storage
contract.

The Tencent reference in `services/reliability-relay-tencent/` is an
independent aggregate-only collector using the same exact schema and privacy
contract. It does not make the global service a hidden dependency of a regional
route.

The dependency-free contract tests run with:

```bash
node --test services/reliability-collector/worker.test.mjs
```

They prove that the health endpoint is minimal, a valid event reaches only
aggregate SQL, the transient rate-limit address is not bound to storage, and
an undeclared scientific-looking field is rejected before any database call.
