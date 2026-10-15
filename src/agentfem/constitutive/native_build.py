# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Content-addressed builds of explicitly trusted project Fortran libraries.

This is build infrastructure, not a material ABI or a constitutive adapter.
The manifest declares every source/include and ordered link target. No shell
commands, downloads, package installation or automatic material loading occur.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import platform
import re
import shutil
import subprocess
import tempfile
import tomllib


def _hash(data):
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class NativeLibraryBundle:
    directory: Path
    record_json: str

    def summary(self):
        return json.loads(self.record_json)

    def library(self, name):
        record = self.summary()
        if name not in record["targets"]:
            raise ValueError("AFM-NATIVE-TARGET: unknown library")
        path = self.directory / record["targets"][name]
        if _hash(path.read_bytes()) != record["libraries"][path.name]:
            raise ValueError("AFM-NATIVE-CACHE: compiled library changed")
        return path


def build_native_libraries(manifest, *, cache_directory, trusted=False,
                           compiler="gfortran", timeout=180):
    """Build a reviewed GNU Fortran bundle on macOS/Linux/WSL2.

    A TOML manifest uses schema ``agentfem.native-libraries.v1``, ``files``
    (all sources/headers), optional relative ``include_directories`` and
    ``flags``, and ``targets`` mapping names to ordered source lists. All
    inputs are snapshotted before compilation and hashed with compiler identity.
    Trust covers executable source, not merely the integrity of a download.
    No thread safety, kinematics, stable increment or material ABI is inferred.
    """
    if trusted is not True:
        raise ValueError("AFM-NATIVE-TRUST: source review and trusted=True are required")
    if platform.system() not in ("Darwin", "Linux"):
        raise RuntimeError("AFM-NATIVE-PLATFORM: use macOS, Linux or WSL2")
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not 0 < timeout <= 3600:
        raise ValueError("AFM-NATIVE-TIMEOUT: require 0 < timeout <= 3600")
    source = Path(manifest).resolve()
    spec = tomllib.loads(source.read_text())
    if set(spec) - {"schema", "files", "bundled_sources", "include_directories", "flags", "targets"} or spec.get("schema") != "agentfem.native-libraries.v1":
        raise ValueError("AFM-NATIVE-MANIFEST: unsupported fields or schema")
    root = source.parent

    def relative(value):
        if not isinstance(value, str) or not value or "\\" in value:
            raise ValueError("AFM-NATIVE-PATH: require relative POSIX paths")
        p = Path(value)
        if p.is_absolute() or ".." in p.parts or not (root / p).resolve().is_relative_to(root):
            raise ValueError("AFM-NATIVE-PATH: source must stay inside the project material directory")
        return p.as_posix()

    names = spec.get("files", [])
    if not isinstance(names, list):
        raise ValueError("AFM-NATIVE-FILES: declare every source and include")
    names = [relative(n) for n in names]
    if len(set(names)) != len(names):
        raise ValueError("AFM-NATIVE-FILES: duplicate input")
    payload = {name: (root / name).read_bytes() for name in names}
    bundled = spec.get("bundled_sources", {})
    if not isinstance(bundled, dict):
        raise ValueError("AFM-NATIVE-BUNDLED: expected a mapping")
    for name, asset in bundled.items():
        name = relative(name)
        if asset not in ("polar_series.f90", "polar_wider.f90") or name in payload:
            raise ValueError("AFM-NATIVE-BUNDLED: unknown or overlapping source")
        payload[name] = (Path(__file__).parent / "native" / asset).read_bytes()
    if not payload:
        raise ValueError("AFM-NATIVE-FILES: no source inputs")
    targets = spec.get("targets")
    if not isinstance(targets, dict) or not targets:
        raise ValueError("AFM-NATIVE-TARGET: declare at least one library")
    for name, inputs in targets.items():
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", name) or not isinstance(inputs, list) or not inputs:
            raise ValueError("AFM-NATIVE-TARGET: invalid target name or sources")
        if any(not isinstance(p, str) or p not in payload for p in inputs):
            raise ValueError("AFM-NATIVE-TARGET: target sources must be declared in files")
    includes = spec.get("include_directories", ["."])
    if not isinstance(includes, list):
        raise ValueError("AFM-NATIVE-INCLUDE: expected a list")
    includes = [relative(p) for p in includes]
    # Fortran INCLUDE dependencies must be part of the frozen input set.
    # This is dependency admission, not a sandbox for trusted executable code.
    for name, data in payload.items():
        for declaration in re.findall(r"(?im)^\s*(?:include|#include)\s*['\"]([^'\"]+)['\"]", data.decode("utf-8")):
            include = relative(declaration)
            candidates = [(Path(name).parent / include).as_posix(),
                          *[(Path(p) / include).as_posix() for p in includes]]
            if not any(p in payload for p in candidates):
                raise ValueError(f"AFM-NATIVE-INCLUDE: undeclared include {declaration}")
    allowed = {"-O0", "-O1", "-O2", "-O3", "-fno-fast-math", "-ffp-contract=off",
               "-fPIC", "-frecursive", "-ffree-line-length-none", "-ffixed-line-length-none",
               "-fdefault-real-8", "-fdefault-double-8", "-shared"}
    flags = spec.get("flags", ["-O3", "-fno-fast-math", "-ffp-contract=off", "-fPIC", "-shared"])
    if not isinstance(flags, list) or any(not isinstance(f, str) or f not in allowed for f in flags):
        raise ValueError("AFM-NATIVE-FLAGS: unsupported compiler option")
    if "-shared" not in flags:
        raise ValueError("AFM-NATIVE-FLAGS: -shared is required")
    executable = shutil.which(str(compiler))
    if executable is None:
        raise RuntimeError("AFM-NATIVE-COMPILER: GNU Fortran is required; no automatic installation")
    version = subprocess.check_output([executable, "--version"], text=True, timeout=10).splitlines()[0]
    if "GNU Fortran" not in version:
        raise ValueError("AFM-NATIVE-COMPILER: only GNU Fortran is supported")
    identity = dict(schema=spec["schema"], sources={n: _hash(b) for n, b in sorted(payload.items())},
                    targets=targets, include_directories=includes, flags=flags,
                    compiler=version, compiler_sha256=_hash(Path(executable).read_bytes()),
                    system=platform.system(), machine=platform.machine())
    key = _hash(json.dumps(identity, sort_keys=True).encode())
    cache = Path(cache_directory).expanduser().resolve()
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / key
    suffix = ".dylib" if platform.system() == "Darwin" else ".so"
    libraries = {n: n + suffix for n in targets}
    import fcntl
    with (cache / "build.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not target.exists():
            stage = Path(tempfile.mkdtemp(prefix="pending-", dir=cache))
            inputs_root = stage / "inputs"
            for name, data in payload.items():
                path = inputs_root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
            try:
                for name, inputs in targets.items():
                    command = [executable, *flags, *["-I" + str(inputs_root / p) for p in includes],
                               *[str(inputs_root / p) for p in inputs], "-o", str(stage / libraries[name])]
                    with (stage / (name + ".log")).open("x") as log:
                        subprocess.run(command, cwd=stage, stdout=log, stderr=subprocess.STDOUT,
                                       check=True, timeout=timeout)
                record = {"identity": identity, "targets": libraries,
                          "libraries": {p: _hash((stage / p).read_bytes()) for p in libraries.values()}}
                (stage / "build.json").write_text(json.dumps(record, sort_keys=True, indent=2))
                stage.rename(target)
            except (OSError, subprocess.SubprocessError) as error:
                raise RuntimeError(f"AFM-NATIVE-BUILD: failed; diagnostics retained in {stage}") from error
        try:
            record = json.loads((target / "build.json").read_text())
            valid = record["identity"] == identity and record["targets"] == libraries
            valid = valid and set(record["libraries"]) == set(libraries.values())
            valid = valid and all(_hash((target / p).read_bytes()) == record["libraries"][p] for p in libraries.values())
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise ValueError("AFM-NATIVE-CACHE: unreadable build record") from error
        if not valid:
            raise ValueError("AFM-NATIVE-CACHE: compiled library integrity mismatch")
    return NativeLibraryBundle(target, json.dumps(record, sort_keys=True))
