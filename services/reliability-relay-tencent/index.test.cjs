const assert = require("node:assert/strict");
const test = require("node:test");

const {cosAuthorization, createCosStore, createHandler} = require("./index.js");

function event(overrides = {}) {
  return {
    schema: "agentfem.reliability-event",
    schema_version: "0.1.0",
    event_id: "00000000-0000-4000-8000-000000000001",
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

function request(events, headers = {}) {
  return {
    httpMethod: "POST",
    path: "/v1/reliability",
    headers,
    body: JSON.stringify({
      schema: "agentfem.reliability-batch",
      schema_version: "0.1.0",
      events: Array.isArray(events) ? events : [events],
    }),
  };
}

function memoryStore() {
  let record = null;
  return {
    async load() { return record; },
    async save(_day, value) { record = JSON.parse(JSON.stringify(value)); },
    get record() { return record; },
  };
}

test("health response declares independent aggregate-only storage", async () => {
  const response = await createHandler({store: memoryStore()})({httpMethod: "GET", path: "/health"});
  assert.equal(response.statusCode, 200);
  assert.deepEqual(JSON.parse(response.body), {
    status: "ok", storage: "daily_aggregates_only", provider: "tencent_cos",
  });
});

test("valid batches become daily counters without request metadata or raw events", async () => {
  const storage = memoryStore();
  const handler = createHandler({store: storage, now: () => new Date("2026-09-03T08:00:00Z")});
  const response = await handler(request(event(), {
    "x-scf-remote-addr": "192.0.2.7",
    "authorization": "private",
  }));
  assert.equal(response.statusCode, 202);
  assert.equal(storage.record.event_count, 1);
  assert.equal(storage.record.seen_event_ids.length, 1);
  const encoded = JSON.stringify(storage.record);
  assert.equal(encoded.includes("192.0.2.7"), false);
  assert.equal(encoded.includes("authorization"), false);
  assert.equal(encoded.includes("events"), false);
  assert.equal(encoded.includes("runtime"), false);
});

test("duplicate delivery is idempotent within a daily aggregate", async () => {
  const storage = memoryStore();
  const handler = createHandler({store: storage, now: () => new Date("2026-09-03T08:00:00Z")});
  assert.equal((await handler(request(event()))).statusCode, 202);
  const response = await handler(request(event()));
  assert.deepEqual(JSON.parse(response.body), {status: "accepted", accepted: 0, duplicates: 1});
  assert.equal(storage.record.event_count, 1);
  assert.equal(storage.record.duplicate_count, 1);
});

test("undeclared scientific data is rejected before storage", async () => {
  const storage = memoryStore();
  const response = await createHandler({store: storage})(request(event({material: {young: 210e9}})));
  assert.equal(response.statusCode, 400);
  assert.equal(storage.record, null);
});

test("delivery IDs must be random UUIDs rather than user-controlled text", async () => {
  const storage = memoryStore();
  const response = await createHandler({store: storage})(
    request(event({event_id: "customer-name-or-project"}))
  );
  assert.equal(response.statusCode, 400);
  assert.equal(storage.record, null);
});

test("storage failure remains outside the simulation lifecycle", async () => {
  const storage = {
    async load() { throw new Error("offline"); },
    async save() { throw new Error("offline"); },
  };
  const response = await createHandler({store: storage})(request(event()));
  assert.equal(response.statusCode, 503);
  assert.equal(JSON.parse(response.body).status, "storage_unavailable");
});

test("COS signing includes the runtime security token without exposing its value", () => {
  const authorization = cosAuthorization({
    method: "GET",
    pathname: "/daily/2026-09-03.json",
    host: "bucket.cos.ap-guangzhou.myqcloud.com",
    secretId: "temporary-id",
    secretKey: "temporary-key",
    token: "temporary-token",
    now: new Date("2026-09-03T08:00:00Z"),
  });
  assert.match(authorization, /q-header-list=host;x-cos-security-token/);
  assert.match(authorization, /q-signature=[a-f0-9]{40}/);
  assert.equal(authorization.includes("temporary-key"), false);
  assert.equal(authorization.includes("temporary-token"), false);
});

test("COS store limits reads and writes to one daily aggregate prefix", async () => {
  const calls = [];
  const store = createCosStore({
    bucket: "aggregate-123",
    region: "ap-guangzhou",
    request: async (options) => {
      calls.push(options);
      return options.method === "GET"
        ? {statusCode: 404, body: ""}
        : {statusCode: 200, body: ""};
    },
  });
  assert.equal(await store.load("2026-09-03"), null);
  await store.save("2026-09-03", {schema: "agentfem.daily-reliability"});
  assert.deepEqual(calls.map(({method, bucket, region, key}) => ({method, bucket, region, key})), [
    {method: "GET", bucket: "aggregate-123", region: "ap-guangzhou", key: "daily/2026-09-03.json"},
    {method: "PUT", bucket: "aggregate-123", region: "ap-guangzhou", key: "daily/2026-09-03.json"},
  ]);
});
