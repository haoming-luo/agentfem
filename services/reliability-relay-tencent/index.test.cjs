const assert = require("node:assert/strict");
const test = require("node:test");

const {createHandler} = require("./index.js");

function event(overrides = {}) {
  return {
    schema: "agentfem.reliability-event",
    schema_version: "0.1.0",
    event_id: "00000000-0000-0000-0000-000000000001",
    agentfem_version: "0.3.2",
    command: "run",
    outcome: "failed",
    duration_bucket: "1-10m",
    runtime: {
      system: "Linux", route: "WSL2", machine: "x86_64", python: "3.11",
      dolfinx: "0.11.0", petsc4py: "3.25.2", mpi_vendor: "MPICH",
      mpi_ranks: 2, installation: "installed",
    },
    failure: {
      code: "AFM-SOLVE-007", stage: "solve", kind: "RuntimeError",
      fingerprint: "AFM-FP-0123456789AB",
    },
    ...overrides,
  };
}

function request(payload, headers = {}) {
  return {
    httpMethod: "POST",
    path: "/v1/reliability",
    headers,
    body: JSON.stringify({
      schema: "agentfem.reliability-batch",
      schema_version: "0.1.0",
      events: [payload],
    }),
  };
}

test("health response declares a storage-free schema relay", async () => {
  const response = await createHandler()({httpMethod: "GET", path: "/health"});
  assert.equal(response.statusCode, 200);
  assert.deepEqual(JSON.parse(response.body), {
    status: "ok", storage: "none", forwarding: "schema_only",
  });
});

test("valid batch forwards only its reviewed JSON body", async () => {
  const calls = [];
  const handler = createHandler({
    endpoint: "https://collector.example/v1/reliability",
    post: async (endpoint, body) => calls.push({endpoint, body}),
  });
  const response = await handler(request(event(), {
    "x-scf-remote-addr": "192.0.2.7",
    "authorization": "private",
  }));
  assert.equal(response.statusCode, 202);
  assert.equal(calls.length, 1);
  assert.equal(calls[0].endpoint, "https://collector.example/v1/reliability");
  assert.equal(calls[0].body.includes("192.0.2.7"), false);
  assert.equal(calls[0].body.includes("authorization"), false);
});

test("undeclared scientific data is rejected before forwarding", async () => {
  let calls = 0;
  const handler = createHandler({
    endpoint: "https://collector.example/v1/reliability",
    post: async () => { calls += 1; },
  });
  const response = await handler(request(event({material: {young: 210e9}})));
  assert.equal(response.statusCode, 400);
  assert.equal(calls, 0);
});

test("upstream failure remains outside the simulation lifecycle", async () => {
  const handler = createHandler({
    endpoint: "https://collector.example/v1/reliability",
    post: async () => { throw new Error("offline"); },
  });
  const response = await handler(request(event()));
  assert.equal(response.statusCode, 503);
  assert.equal(JSON.parse(response.body).status, "upstream_unavailable");
});
