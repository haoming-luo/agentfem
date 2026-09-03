"use strict";

const https = require("node:https");

const EVENT_KEYS = new Set([
  "schema", "schema_version", "event_id", "agentfem_version",
  "command", "outcome", "duration_bucket", "runtime", "failure",
]);
const RUNTIME_KEYS = new Set([
  "system", "route", "machine", "python", "dolfinx", "petsc4py",
  "mpi_vendor", "mpi_ranks", "installation",
]);
const FAILURE_KEYS = new Set(["code", "stage", "kind", "fingerprint"]);
const OUTCOMES = new Set(["completed", "failed", "cancelled"]);
const DURATIONS = new Set(["<1s", "1-10s", "10-60s", "1-10m", "10-60m", ">=1h", "unknown"]);

function exactKeys(value, expected) {
  if (!value || typeof value !== "object" || Array.isArray(value)) return false;
  const keys = Object.keys(value);
  return keys.length === expected.size && keys.every((key) => expected.has(key));
}

function bounded(value, size = 80) {
  return typeof value === "string" && value.length > 0 && value.length <= size;
}

function safeText(value, size = 80) {
  return bounded(String(value), size) && /^[A-Za-z0-9 ._+:/()<>-]+$/.test(String(value));
}

function validEvent(event) {
  if (!exactKeys(event, EVENT_KEYS) || event.schema !== "agentfem.reliability-event") return false;
  if (!exactKeys(event.runtime, RUNTIME_KEYS)) return false;
  if (!bounded(event.event_id, 64) || !bounded(event.agentfem_version, 32)) return false;
  if (!bounded(event.command, 64) || !OUTCOMES.has(event.outcome)) return false;
  if (!DURATIONS.has(event.duration_bucket)) return false;
  if (event.outcome === "failed" && !exactKeys(event.failure, FAILURE_KEYS)) return false;
  if (event.outcome !== "failed" && event.failure !== null) return false;
  if (event.failure !== null) {
    if (!safeText(event.failure.code) || !safeText(event.failure.stage) || !safeText(event.failure.kind)) return false;
    if (!/^AFM-FP-[A-F0-9]{12}$/.test(event.failure.fingerprint || "")) return false;
  }
  return Object.entries(event.runtime).every(([key, value]) =>
    key === "mpi_ranks"
      ? Number.isInteger(value) && value >= 1 && value <= 65536
      : value === null || safeText(value)
  );
}

function validBatch(payload) {
  return Boolean(
    payload && payload.schema === "agentfem.reliability-batch" &&
    Object.keys(payload).length === 3 &&
    Object.hasOwn(payload, "schema_version") &&
    Array.isArray(payload.events) && payload.events.length >= 1 &&
    payload.events.length <= 8 && payload.events.every(validEvent)
  );
}

function reply(statusCode, payload) {
  return {
    statusCode,
    headers: {"Content-Type": "application/json", "Cache-Control": "no-store"},
    body: JSON.stringify(payload),
  };
}

function postJson(url, body, timeoutMilliseconds = 600) {
  return new Promise((resolve, reject) => {
    const target = new URL(url);
    const request = https.request({
      protocol: target.protocol,
      hostname: target.hostname,
      port: target.port || 443,
      path: `${target.pathname}${target.search}`,
      method: "POST",
      timeout: timeoutMilliseconds,
      headers: {
        "Content-Type": "application/json",
        "Content-Length": Buffer.byteLength(body),
        "User-Agent": "AgentFEM mainland privacy relay",
      },
    }, (response) => {
      response.resume();
      response.on("end", () => {
        if (response.statusCode >= 200 && response.statusCode < 300) resolve(response.statusCode);
        else reject(new Error(`upstream status ${response.statusCode}`));
      });
    });
    request.on("timeout", () => request.destroy(new Error("upstream timeout")));
    request.on("error", reject);
    request.end(body);
  });
}

function createHandler({post = postJson, endpoint = process.env.UPSTREAM_RELIABILITY_ENDPOINT} = {}) {
  return async function handler(event = {}) {
    const method = String(event.httpMethod || "GET").toUpperCase();
    const path = String(event.path || "/");
    if (method === "GET" && (path === "/" || path === "/health")) {
      return reply(200, {status: "ok", storage: "none", forwarding: "schema_only"});
    }
    if (method !== "POST") return reply(405, {status: "method_not_allowed"});
    const encoded = Boolean(event.isBase64Encoded);
    const raw = encoded
      ? Buffer.from(String(event.body || ""), "base64").toString("utf8")
      : String(event.body || "");
    if (Buffer.byteLength(raw) > 65536) return reply(413, {status: "payload_too_large"});
    let payload;
    try {
      payload = JSON.parse(raw);
    } catch {
      return reply(400, {status: "invalid_json"});
    }
    if (!validBatch(payload)) return reply(400, {status: "invalid_reliability_schema"});
    if (!endpoint || !endpoint.startsWith("https://")) {
      return reply(503, {status: "relay_unconfigured"});
    }
    try {
      // Only the validated body is forwarded. Function-URL request headers,
      // including x-scf-remote-addr, are deliberately not read or copied.
      await post(endpoint, JSON.stringify(payload));
    } catch {
      return reply(503, {status: "upstream_unavailable"});
    }
    return reply(202, {status: "accepted"});
  };
}

exports.main_handler = createHandler();
exports.createHandler = createHandler;
exports.validBatch = validBatch;
