"use strict";

const crypto = require("node:crypto");
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
const MAX_DAILY_EVENTS = 20000;
const MAX_COUNTER_BUCKETS = 4096;

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
  if (!/^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i.test(event.event_id || "")) return false;
  if (!bounded(event.agentfem_version, 32)) return false;
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

function text(value, fallback = "none") {
  return value === null || value === undefined ? fallback : String(value).slice(0, 80);
}

function counterKey(event) {
  const runtime = event.runtime;
  const failure = event.failure || {};
  return JSON.stringify([
    text(event.agentfem_version), text(event.command), text(event.outcome),
    text(event.duration_bucket), text(runtime.system), text(runtime.route),
    text(runtime.machine), text(runtime.python), text(runtime.dolfinx),
    text(runtime.petsc4py), text(runtime.mpi_vendor), text(runtime.mpi_ranks),
    text(runtime.installation), text(failure.code), text(failure.stage),
    text(failure.kind), text(failure.fingerprint),
  ]);
}

function emptyAggregate(day) {
  return {
    schema: "agentfem.daily-reliability",
    schema_version: "0.1.0",
    day,
    event_count: 0,
    duplicate_count: 0,
    seen_event_ids: [],
    counters: {},
  };
}

function aggregateBatch(existing, events, day) {
  const record = existing && existing.schema === "agentfem.daily-reliability" && existing.day === day
    ? existing : emptyAggregate(day);
  const seen = new Set(Array.isArray(record.seen_event_ids) ? record.seen_event_ids : []);
  const counters = record.counters && typeof record.counters === "object" ? {...record.counters} : {};
  let accepted = 0;
  let duplicates = 0;
  for (const event of events) {
    if (seen.has(event.event_id)) {
      duplicates += 1;
      continue;
    }
    if (seen.size >= MAX_DAILY_EVENTS) throw new Error("daily event capacity reached");
    seen.add(event.event_id);
    const key = counterKey(event);
    if (!Object.hasOwn(counters, key) && Object.keys(counters).length >= MAX_COUNTER_BUCKETS) {
      throw new Error("daily counter capacity reached");
    }
    counters[key] = Number(counters[key] || 0) + 1;
    accepted += 1;
  }
  return {
    record: {
      ...emptyAggregate(day),
      event_count: Number(record.event_count || 0) + accepted,
      duplicate_count: Number(record.duplicate_count || 0) + duplicates,
      seen_event_ids: [...seen],
      counters,
    },
    accepted,
    duplicates,
  };
}

function sha1(value) {
  return crypto.createHash("sha1").update(value).digest("hex");
}

function hmacSha1(key, value) {
  return crypto.createHmac("sha1", key).update(value).digest("hex");
}

function encode(value) {
  return encodeURIComponent(String(value)).replace(/[!'()*]/g, (character) =>
    `%${character.charCodeAt(0).toString(16).toUpperCase()}`
  );
}

function cosAuthorization({method, pathname, host, secretId, secretKey, token, now = new Date()}) {
  const start = Math.floor(now.getTime() / 1000) - 60;
  const keyTime = `${start};${start + 900}`;
  const headers = {host};
  if (token) headers["x-cos-security-token"] = token;
  const headerNames = Object.keys(headers).sort();
  const headerList = headerNames.map(encode).join(";");
  const canonicalHeaders = headerNames
    .map((name) => `${encode(name)}=${encode(headers[name].trim())}`)
    .join("&");
  const httpString = `${method.toLowerCase()}\n${pathname}\n\n${canonicalHeaders}\n`;
  const stringToSign = `sha1\n${keyTime}\n${sha1(httpString)}\n`;
  const signature = hmacSha1(hmacSha1(secretKey, keyTime), stringToSign);
  return [
    "q-sign-algorithm=sha1",
    `q-ak=${encode(secretId)}`,
    `q-sign-time=${keyTime}`,
    `q-key-time=${keyTime}`,
    `q-header-list=${headerList}`,
    "q-url-param-list=",
    `q-signature=${signature}`,
  ].join("&");
}

function cosRequest({method, bucket, region, key, body = null, now = new Date()}) {
  const secretId = process.env.TENCENTCLOUD_SECRETID;
  const secretKey = process.env.TENCENTCLOUD_SECRETKEY;
  const token = process.env.TENCENTCLOUD_SESSIONTOKEN;
  if (!secretId || !secretKey) throw new Error("SCF runtime role credentials are required");
  const host = `${bucket}.cos.${region}.myqcloud.com`;
  const pathname = `/${key.split("/").map(encode).join("/")}`;
  const authorization = cosAuthorization({
    method, pathname, host, secretId, secretKey, token, now,
  });
  return new Promise((resolve, reject) => {
    const headers = {host, authorization};
    if (token) headers["x-cos-security-token"] = token;
    if (body !== null) {
      headers["content-type"] = "application/json";
      headers["content-length"] = Buffer.byteLength(body);
    }
    const request = https.request({method, host, path: pathname, headers, timeout: 5000}, (response) => {
      const chunks = [];
      response.on("data", (chunk) => chunks.push(chunk));
      response.on("end", () => resolve({
        statusCode: Number(response.statusCode || 0),
        body: Buffer.concat(chunks).toString("utf8"),
      }));
    });
    request.on("timeout", () => request.destroy(new Error("COS request timed out")));
    request.on("error", reject);
    if (body !== null) request.write(body);
    request.end();
  });
}

function createCosStore({
  bucket = process.env.AGGREGATE_BUCKET,
  region = process.env.AGGREGATE_REGION || "ap-guangzhou",
  request = cosRequest,
} = {}) {
  if (!bucket) throw new Error("AGGREGATE_BUCKET is required");
  return {
    async load(day) {
      const response = await request({
        method: "GET", bucket, region, key: `daily/${day}.json`,
      });
      if (response.statusCode === 404) return null;
      if (response.statusCode !== 200) throw new Error(`COS read failed: ${response.statusCode}`);
      return JSON.parse(response.body);
    },
    async save(day, record) {
      const body = JSON.stringify(record);
      const response = await request({
        method: "PUT", bucket, region, key: `daily/${day}.json`, body,
      });
      if (response.statusCode < 200 || response.statusCode >= 300) {
        throw new Error(`COS write failed: ${response.statusCode}`);
      }
    },
  };
}

function createHandler({store, now = () => new Date()} = {}) {
  return async function handler(event = {}) {
    const method = String(event.httpMethod || "GET").toUpperCase();
    const path = String(event.path || "/");
    if (method === "GET" && (path === "/" || path === "/health")) {
      return reply(200, {status: "ok", storage: "daily_aggregates_only", provider: "tencent_cos"});
    }
    if (method !== "POST") return reply(405, {status: "method_not_allowed"});
    const raw = event.isBase64Encoded
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
    try {
      // Request headers, source addresses and raw bodies are deliberately not
      // passed to storage. Only the reviewed schema is reduced to counters.
      const selectedStore = store || createCosStore();
      const day = now().toISOString().slice(0, 10);
      const previous = await selectedStore.load(day);
      const aggregate = aggregateBatch(previous, payload.events, day);
      await selectedStore.save(day, aggregate.record);
      return reply(202, {
        status: "accepted",
        accepted: aggregate.accepted,
        duplicates: aggregate.duplicates,
      });
    } catch {
      return reply(503, {status: "storage_unavailable"});
    }
  };
}

exports.main_handler = createHandler();
exports.createHandler = createHandler;
exports.createCosStore = createCosStore;
exports.cosAuthorization = cosAuthorization;
exports.aggregateBatch = aggregateBatch;
exports.validBatch = validBatch;
