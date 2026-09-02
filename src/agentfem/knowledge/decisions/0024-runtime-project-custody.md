# Runtime and project custody are separate

## Decision

AgentFEM treats a numerical runtime as replaceable and a scientific project as
durable user data. On Windows through WSL, the conventional
`~/AgentFEMProjects` entry resolves to a host-owned Windows directory by
default. The same `agentfem workspace` contract applies whether AgentFEM came
from the Complete Runtime, conda-forge, Mamba, PyPI, or a source checkout.

Migration is fail-closed: copy files, verify their content identity, retain the
pre-migration directory, and only then switch the conventional path. Project
and output custody are recorded in execution evidence. Runtime removal first
requires protected custody and retains a complete recovery snapshot unless the
user explicitly declines that runtime backup.

## Reason

A WSL distribution contains both software and an ext4 filesystem, and
unregistering it permanently deletes both. A novice should not lose models,
results, or checkpoints merely because AgentFEM is upgraded or removed.
Conversely, package environments must remain replaceable so exact numerical
runtimes can evolve without becoming the owner of scientific work.

## Consequences

- Complete Runtime and Mamba users see one project path and one safety check;
- projects and default outputs remain together as a portable Windows folder;
- paths intentionally kept inside WSL are warned about rather than silently
  treated as durable;
- high-frequency recomputable scratch may remain on the Linux filesystem for
  performance, but it is not the only accepted copy of a result;
- raw `wsl --unregister` is outside the AgentFEM lifecycle contract.

## Evidence

- workspace migration, conflict, symlink, and installed-CLI regression tests;
- project and output custody fields in project/run records;
- PowerShell parser checks for install and removal scripts;
- full serial and two-rank MPI regression;
- installed-wheel release smoke including the workspace machine entrypoint.
