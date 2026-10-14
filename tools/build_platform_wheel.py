# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Build one platform wheel from the immutable source archive, never a checkout.

Linux publication uses auditwheel repair before any acceptance/hash recording.
The caller owns environment provisioning; this tool does not install libraries.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile

from packaging.tags import sys_tags
from packaging.utils import parse_wheel_filename


def select_source(directory: Path) -> Path:
    sources = sorted(directory.glob("agentfem-*.tar.gz"))
    if len(sources) != 1:
        raise ValueError("Expected exactly one AgentFEM source distribution")
    return sources[0]


def inspect_wheel(wheel: Path, *, require_portable_linux: bool = False) -> dict:
    _, _, _, tags = parse_wheel_filename(wheel.name)
    if not tags.intersection(sys_tags()):
        raise ValueError("Wheel is not compatible with the acceptance host")
    if require_portable_linux and any(
        not tag.platform.startswith(("manylinux", "musllinux")) for tag in tags
    ):
        raise ValueError("Linux publication requires a repaired portable platform tag")
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
    modules = {
        name: any(path.startswith(f"agentfem/{name}.") and path.endswith((".so", ".pyd"))
                  for path in names)
        for name in ("_hex_native", "_body_pipeline", "_q1_native")
    }
    if not all(modules[name] for name in ("_hex_native", "_body_pipeline")):
        raise ValueError("Platform candidate silently omitted required body accelerators")
    return {"tags": sorted(str(tag) for tag in tags), "packaged_native_modules": modules}


def build(source_directory: Path, output: Path, *, repair: str = "none") -> dict:
    source = select_source(source_directory).resolve()
    if repair not in ("none", "linux") or (repair == "linux" and sys.platform != "linux"):
        raise ValueError("Unsupported repair mode for this host")
    # Refuse to replace existing candidate artifacts, even on a failed retry.
    output.mkdir(parents=True, exist_ok=False)
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    with tempfile.TemporaryDirectory(prefix="agentfem-platform-build-") as temporary:
        raw = Path(temporary) / "raw"
        raw.mkdir()
        environment = dict(os.environ, AGENTFEM_BUILD_NATIVE="1")
        subprocess.run([
            sys.executable, "-m", "pip", "wheel", str(source), "--no-deps",
            "--no-cache-dir", "--wheel-dir", str(raw),
        ], env=environment, check=True)
        wheels = list(raw.glob("agentfem-*.whl"))
        if len(wheels) != 1:
            raise RuntimeError("Platform build did not produce exactly one wheel")
        selected = wheels[0]
        if repair == "linux":
            repaired = Path(temporary) / "repaired"
            subprocess.run([
                sys.executable, "-m", "auditwheel", "repair", str(selected),
                "--wheel-dir", str(repaired),
            ], check=True)
            wheels = list(repaired.glob("agentfem-*.whl"))
            if len(wheels) != 1:
                raise RuntimeError("Repair did not produce exactly one portable wheel")
            selected = wheels[0]
        record = inspect_wheel(selected, require_portable_linux=repair == "linux")
        if hashlib.sha256(source.read_bytes()).hexdigest() != source_hash:
            raise RuntimeError("Source distribution changed during the build")
        shutil.copy2(source, output / source.name)
        shutil.copy2(selected, output / selected.name)
        record.update(
            source=source.name, source_sha256=source_hash, wheel=selected.name,
            wheel_sha256=hashlib.sha256(selected.read_bytes()).hexdigest(),
            repair=repair, host=sys.platform,
            scope="packaging only; installed numerical acceptance remains required",
        )
        (output / "platform-build.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repair", choices=("none", "linux"), default="none")
    args = parser.parse_args()
    print(json.dumps(build(args.source_dir, args.output, repair=args.repair)))


if __name__ == "__main__":
    main()
