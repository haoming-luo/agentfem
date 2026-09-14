# AgentFEM 0.3.6

AgentFEM 0.3.6 is a focused installation-reliability release. It does not
change the finite-element formulations or promote any scientific capability.

## One explicit mirror contract

People may tell an AI agent **"走镜像通道 / use the mirror channel"**. The
agent then follows the documented single-source mirror route for the complete
FEniCSx/PETSc/MPI stack. AgentFEM does not infer this decision from IP or VPN
location, and it does not mix numerical package channels.

## Windows first-install path

The Windows Complete Runtime keeps one entry command:

```powershell
powershell -ExecutionPolicy Bypass -File .\Install-AgentFEM.ps1
```

When WSL is absent or outdated, the installer now requests Windows
administrator approval and prepares the platform without downloading an
unrelated Ubuntu distribution. The automatic policy tries Microsoft's direct
web route and then the Microsoft Store route if the first command fails.
Explicit Web, Store, and offline-MSI paths remain available.

A required Windows restart is an explicit continuation state: restart, then
run the same command again. Unsupported architecture, obsolete Windows/WSL,
and host virtualization failures remain distinct from AgentFEM solver errors.

## Data and trust boundary

The runtime image, project-storage contract, numerical dependencies, and
scientific implementation are unchanged from 0.3.5. Project inputs, outputs,
and checkpoints continue to live under Windows
`Documents\AgentFEMProjects`, outside the replaceable WSL distribution.

The Windows artifact remains a Preview until its exact released bundle is
accepted on a real Windows/WSL2 host. Package tests, PowerShell parsing,
runtime-image acceptance, checksums, and release provenance remain mandatory
before publication.

## Known issue: first-solve C compiler

The published v0.3.6 Windows Preview may not expose a C compiler when FEniCSx
JIT-compiles the first new form. Existing installations can be repaired in
place, without changing project data:

```bash
sudo apt-get update
sudo apt-get install -y gcc g++
agentfem doctor
```

The corrected milestone build contract installs the compiler in the exported
root filesystem and accepts the final `.wsl` artifact after a clean re-import.
The [first successful Windows field report](https://github.com/haoming-luo/agentfem/discussions/3)
records the clean-host WSL2 + TUNA route through a completed solve; TUNA has
since synchronized AgentFEM 0.3.6.
