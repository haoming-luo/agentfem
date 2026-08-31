CREATE TABLE IF NOT EXISTS seen_events (
  event_id TEXT PRIMARY KEY,
  received_day TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS daily_reliability (
  day TEXT NOT NULL,
  agentfem_version TEXT NOT NULL,
  command TEXT NOT NULL,
  outcome TEXT NOT NULL,
  duration_bucket TEXT NOT NULL,
  system TEXT NOT NULL,
  route TEXT NOT NULL,
  machine TEXT NOT NULL,
  python TEXT NOT NULL,
  dolfinx TEXT NOT NULL,
  petsc4py TEXT NOT NULL,
  mpi_vendor TEXT NOT NULL,
  mpi_ranks TEXT NOT NULL,
  installation TEXT NOT NULL,
  failure_code TEXT NOT NULL,
  failure_stage TEXT NOT NULL,
  failure_kind TEXT NOT NULL,
  failure_fingerprint TEXT NOT NULL,
  event_count INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (
    day, agentfem_version, command, outcome, duration_bucket, system, route,
    machine, python, dolfinx, petsc4py, mpi_vendor, mpi_ranks, installation,
    failure_code, failure_stage, failure_kind, failure_fingerprint
  )
);

CREATE INDEX IF NOT EXISTS seen_events_day ON seen_events(received_day);
