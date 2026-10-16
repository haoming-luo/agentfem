# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Read-only run advisories, never scientific controls or automatic tuning."""
import os
from pathlib import Path
import platform
import re


def _read(path):
    try:
        return Path(path).read_text()
    except (OSError, UnicodeError):
        return ""


def _unescape_mount(value):
    return re.sub(r"\\([0-7]{3})", lambda m: chr(int(m[1], 8)), value)


def _storage(path, mountinfo):
    """Longest containing mount, including custom/bind-mounted DrvFS paths."""
    path = Path(path).expanduser().resolve()
    selected = None
    for line in mountinfo.splitlines():
        before, sep, after = line.partition(" - ")
        fields, tail = before.split(), after.split()
        if not sep or len(fields) < 6 or len(tail) < 3:
            continue
        mount = Path(_unescape_mount(fields[4]))
        if not path.is_relative_to(mount):
            continue
        if selected is None or len(mount.parts) > selected[0]:
            windows = tail[0] == "drvfs" or (tail[0] == "9p" and "drvfs" in after.lower())
            selected = (len(mount.parts), str(mount), tail[0], windows)
    # Only a fallback when the mount table cannot establish the filesystem.
    conventional = bool(re.match(r"^/mnt/[a-zA-Z](?:/|$)", str(path)))
    return {"path": str(path), "mount": None if selected is None else selected[1],
            "filesystem": None if selected is None else selected[2],
            "windows_mount": conventional if selected is None else selected[3],
            "detection": "path_fallback" if selected is None else "mountinfo"}


def execution_environment(*, project, output):
    """Cheap snapshot at preflight/run boundaries; no subprocess or benchmark."""
    wsl = platform.system() == "Linux" and (
        "microsoft" in platform.release().lower() or bool(os.environ.get("WSL_DISTRO_NAME")))
    mountinfo = _read('/proc/self/mountinfo') if wsl else ""
    storage = {name: _storage(path, mountinfo) for name, path in
               (("project", project), ("output", output))} if wsl else {}
    try:
        cpus = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        cpus = os.cpu_count()
    memory = {}
    if platform.system() == "Linux":
        for line in _read('/proc/meminfo').splitlines():
            key, _, value = line.partition(':')
            if key in ('MemTotal', 'MemAvailable', 'SwapTotal', 'SwapFree'):
                parts = value.split()
                if len(parts) == 2 and parts[0].isdigit() and parts[1] == 'kB':
                    memory[key] = int(parts[0]) * 1024
    threads = {name: os.environ[name] for name in
               ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'BLIS_NUM_THREADS')
               if name in os.environ}
    findings = []
    if any(item['windows_mount'] for item in storage.values()):
        findings.append({"code": "AFM-PERF-WSL-STORAGE", "severity": "advisory",
            "message": "Windows-mounted project/output storage may slow Linux I/O. For heavy runs, "
                       "use a WSL-native working copy and export results to Windows afterward; "
                       "back it up before removing the WSL distribution. No files were moved."})
    if memory.get('MemTotal', 0) and memory.get('MemAvailable', memory['MemTotal']) < memory['MemTotal'] * .1:
        findings.append({"code": "AFM-PERF-MEMORY-PRESSURE", "severity": "advisory",
            "message": "Less than 10% of visible memory is available at startup. "
                       "Check competing workloads and WSL memory limits before a large run."})
    if cpus and any(value.isdigit() and int(value) > cpus for value in threads.values()):
        findings.append({"code": "AFM-PERF-THREAD-BUDGET", "severity": "advisory",
            "message": "A numerical-library thread setting exceeds the visible CPU affinity. "
                       "Review it together with material/contact workers; settings were not changed."})
    return {"schema": "agentfem.execution-environment.v1", "under_wsl": wsl,
            "visible_cpus": cpus, "memory_bytes": memory, "thread_environment": threads,
            "storage": storage, "findings": findings, "settings_modified": False,
            "scope": "startup_snapshot_not_execution_route_or_speed_prediction"}
