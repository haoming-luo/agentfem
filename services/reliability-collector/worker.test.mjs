import assert from "node:assert/strict";
import test from "node:test";

import worker from "./worker.js";

function event(overrides = {}) {
  return {
    schema: "agentfem.reliability-event",
    schema_version: "0.1.0",
    event_id: "00000000-0000-4000-8000-000000000001",
    agentfem_version: "0.3.1",
    command: "run",
    outcome: "failed",
    duration_bucket: "1-10m",
    runtime: {
      system: "Darwin",
      route: "native macOS",
      machine: "arm64",
      python: "3.11.15",
      dolfinx: "0.11.0",
      petsc4py: "3.25.2",
      mpi_vendor: "MPICH",
      mpi_ranks: 1,
      installation: "installed",
    },
    failure: {
      code: "AFM-SOLVE-007",
      stage: "case_execution",
      kind: "RuntimeError",
      fingerprint: "AFM-FP-0123456789AB",
    },
    ...overrides,
  };
}

function environment() {
  const statements = [];
  const rateKeys = [];
  return {
    statements,
    rateKeys,
    env: {
      RELIABILITY_RATE_LIMITER: {
        async limit({ key }) {
          rateKeys.push(key);
          return { success: true };
        },
      },
      DB: {
        prepare(sql) {
          const statement = { sql, values: [] };
          statements.push(statement);
          return {
            bind(...values) {
              statement.values = values;
              return {
                async run() {
                  return { success: true, meta: { changes: 1 } };
                },
              };
            },
            async run() {
              return { success: true, meta: { changes: 1 } };
            },
          };
        },
      },
    },
  };
}

function post(payload) {
  return new Request("https://reliability.example/v1/reliability", {
    method: "POST",
    headers: {
      "content-type": "application/json",
      "cf-connecting-ip": "192.0.2.1",
    },
    body: JSON.stringify(payload),
  });
}

test("health exposes only the aggregate storage contract", async () => {
  const response = await worker.fetch(
    new Request("https://reliability.example/health"),
    environment().env,
  );
  assert.equal(response.status, 200);
  assert.deepEqual(await response.json(), {
    status: "ok",
    storage: "daily_aggregates_only",
  });
});

test("one exact event is aggregated without persisting the client address", async () => {
  const fixture = environment();
  const response = await worker.fetch(
    post({
      schema: "agentfem.reliability-batch",
      schema_version: "0.1.0",
      events: [event()],
    }),
    fixture.env,
  );
  assert.equal(response.status, 200);
  assert.equal((await response.json()).accepted, 1);
  assert.deepEqual(fixture.rateKeys, ["192.0.2.1"]);
  assert.equal(
    fixture.statements.flatMap((item) => item.values).includes("192.0.2.1"),
    false,
  );
});

test("an undeclared scientific-looking field is rejected before storage", async () => {
  const fixture = environment();
  const response = await worker.fetch(
    post({
      schema: "agentfem.reliability-batch",
      schema_version: "0.1.0",
      events: [event({ project: "confidential-model" })],
    }),
    fixture.env,
  );
  assert.equal(response.status, 400);
  assert.equal(fixture.statements.length, 0);
});
