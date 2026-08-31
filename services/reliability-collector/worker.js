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

function text(value, fallback = "none") {
  return value === null || value === undefined ? fallback : String(value).slice(0, 80);
}

async function aggregate(env, event, day) {
  const seen = await env.DB.prepare(
    "INSERT OR IGNORE INTO seen_events(event_id, received_day) VALUES (?, ?)"
  ).bind(event.event_id, day).run();
  if (!seen.meta?.changes) return false;
  const runtime = event.runtime;
  const failure = event.failure || {};
  await env.DB.prepare(`
    INSERT INTO daily_reliability (
      day, agentfem_version, command, outcome, duration_bucket, system, route,
      machine, python, dolfinx, petsc4py, mpi_vendor, mpi_ranks, installation,
      failure_code, failure_stage, failure_kind, failure_fingerprint, event_count
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
    ON CONFLICT DO UPDATE SET event_count = event_count + 1
  `).bind(
    day, text(event.agentfem_version), text(event.command), text(event.outcome),
    text(event.duration_bucket), text(runtime.system), text(runtime.route),
    text(runtime.machine), text(runtime.python), text(runtime.dolfinx),
    text(runtime.petsc4py), text(runtime.mpi_vendor), text(runtime.mpi_ranks),
    text(runtime.installation), text(failure.code), text(failure.stage),
    text(failure.kind), text(failure.fingerprint)
  ).run();
  return true;
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    if (request.method === "GET" && url.pathname === "/health") {
      return Response.json({ status: "ok", storage: "daily_aggregates_only" });
    }
    if (request.method !== "POST" || url.pathname !== "/v1/reliability") {
      return new Response("Not found", { status: 404 });
    }
    const declared = Number(request.headers.get("content-length") || 0);
    if (declared > 65536) return new Response("Payload too large", { status: 413 });
    if (env.RELIABILITY_RATE_LIMITER) {
      const client = request.headers.get("cf-connecting-ip") || "unknown";
      const decision = await env.RELIABILITY_RATE_LIMITER.limit({ key: client });
      if (!decision.success) return new Response("Rate limited", { status: 429 });
    }
    let payload;
    try {
      const raw = await request.text();
      if (raw.length > 65536) return new Response("Payload too large", { status: 413 });
      payload = JSON.parse(raw);
    } catch {
      return new Response("Invalid JSON", { status: 400 });
    }
    if (
      !payload || payload.schema !== "agentfem.reliability-batch" ||
      !Array.isArray(payload.events) || payload.events.length < 1 ||
      payload.events.length > 8 || !payload.events.every(validEvent)
    ) return new Response("Invalid reliability schema", { status: 400 });
    const day = new Date().toISOString().slice(0, 10);
    let accepted = 0;
    for (const event of payload.events) accepted += await aggregate(env, event, day) ? 1 : 0;
    // Event identifiers are random delivery ids, retained only for seven-day
    // idempotency. No source IP, user id, raw payload, message, or path is stored.
    await env.DB.prepare("DELETE FROM seen_events WHERE received_day < date('now', '-7 day')").run();
    return Response.json({ status: "accepted", accepted, duplicates: payload.events.length - accepted });
  },
};
