#!/usr/bin/env python3
"""Build self-contained AgentFEM desktop runtime artifacts.

The ordinary wheel remains the canonical AgentFEM code artifact. Runtime
installers add a pinned scientific stack around that wheel; they never rebuild
or fork the finite-element implementation.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import tomllib
from typing import Any
from urllib.request import urlopen
import zipfile


ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / "packaging" / "runtime"
BUILD = ROOT / "build" / "runtime"
SCHEMA = "agentfem.runtime-release"
GMSH_RELEASES = {
    "4.15.2": {
        "license_url": "https://gmsh.info/LICENSE.txt",
        "license_sha256": "32085d8c954e2e22dc667c089f360049e8f1af955f946fb0a18f29d70390276a",
        "source_url": "https://gmsh.info/src/gmsh-4.15.2-source.tgz",
        "source_sha256": "be3f66f225d27ba9fa014f07e83169285da8a051b0e8ab7103d88066b39bdd3e",
        "feedstock_commit": "e86636d7217dbfd016d712cb60ab2b846d12c7e2",
        "feedstock_url": "https://github.com/conda-forge/gmsh-feedstock/archive/e86636d7217dbfd016d712cb60ab2b846d12c7e2.tar.gz",
        "feedstock_sha256": "7b4aca9f1bbae13bedf59067e24d62d8994e41fe5031eaf7cb94a125ac99e4b3",
    }
}


def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
    print("+", " ".join(command), flush=True)
    return subprocess.run(command, check=True, text=True, **kwargs)


def run_with_retries(
    command: list[str], *, attempts: int = 4, **kwargs: Any
) -> subprocess.CompletedProcess[str]:
    last_error: subprocess.CalledProcessError | None = None
    for attempt in range(1, attempts + 1):
        try:
            return run(command, **kwargs)
        except subprocess.CalledProcessError as exc:
            last_error = exc
            if attempt == attempts:
                break
            print(
                f"Command failed during network-backed packaging; retrying "
                f"({attempt}/{attempts})...",
                flush=True,
            )
            time.sleep(2 * attempt)
    assert last_error is not None
    raise last_error


def project_version() -> str:
    with (ROOT / "pyproject.toml").open("rb") as stream:
        return str(tomllib.load(stream)["project"]["version"])


def git_value(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
    ).strip()


def source_identity() -> dict[str, Any]:
    return {
        "commit": git_value("rev-parse", "HEAD"),
        "tag": git_value("describe", "--tags", "--exact-match")
        if subprocess.run(
            ["git", "describe", "--tags", "--exact-match"],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        == 0
        else None,
        "dirty": bool(git_value("status", "--porcelain")),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_verified(url: str, destination: Path, expected_sha256: str) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and sha256(destination) == expected_sha256:
        return destination
    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.unlink(missing_ok=True)
    last_error: Exception | None = None
    for attempt in range(1, 6):
        try:
            with urlopen(url, timeout=60) as response, temporary.open("wb") as stream:
                shutil.copyfileobj(response, stream, length=1024 * 1024)
            actual = sha256(temporary)
            if actual != expected_sha256:
                raise RuntimeError(
                    f"Checksum mismatch for {url}: expected {expected_sha256}, got {actual}"
                )
            temporary.replace(destination)
            return destination
        except Exception as exc:  # network failures are retried deterministically
            last_error = exc
            temporary.unlink(missing_ok=True)
            if attempt < 5:
                time.sleep(attempt)
    raise RuntimeError(f"Could not retrieve verified release input {url}") from last_error


def prepare_legal_materials(profile: str, lock: Path, output: Path) -> tuple[Path, Path]:
    notices = BUILD / "THIRD_PARTY_NOTICES.txt"
    record_path = BUILD / "third-party-components.json"
    gmsh_license_target = BUILD / "GMSH-LICENSE.txt"
    record: dict[str, Any] = {
        "schema": "agentfem.third-party-components",
        "schema_version": 1,
        "profile": profile,
        "components": [],
    }
    text = [
        "AgentFEM Runtime — Third-Party Notices",
        "",
        "AgentFEM is Apache-2.0. Components in the pinned runtime retain their own licenses.",
    ]
    if profile == "complete":
        version = "4.15.2"
        metadata = GMSH_RELEASES[version]
        gmsh_urls = [
            line.strip()
            for line in lock.read_text(encoding="utf-8").splitlines()
            if f"/gmsh-{version}-" in line
        ]
        if len(gmsh_urls) != 1:
            raise RuntimeError(f"Expected one locked Gmsh {version} binary, found {gmsh_urls}")
        third_party = BUILD / "third-party" / "gmsh"
        license_path = fetch_verified(
            str(metadata["license_url"]),
            third_party / "LICENSE.txt",
            str(metadata["license_sha256"]),
        )
        shutil.copy2(license_path, gmsh_license_target)
        source_name = f"Gmsh-{version}-corresponding-source.tar.gz"
        source_path = fetch_verified(
            str(metadata["source_url"]),
            output / source_name,
            str(metadata["source_sha256"]),
        )
        recipe_name = f"Gmsh-{version}-conda-forge-recipe-{str(metadata['feedstock_commit'])[:12]}.tar.gz"
        recipe_path = fetch_verified(
            str(metadata["feedstock_url"]),
            output / recipe_name,
            str(metadata["feedstock_sha256"]),
        )
        record["components"].append(
            {
                "name": "Gmsh",
                "version": version,
                "license": "GPL-2.0-or-later",
                "binary": gmsh_urls[0],
                "license_sha256": sha256(license_path),
                "corresponding_source": {
                    "filename": source_path.name,
                    "sha256": sha256(source_path),
                    "upstream_url": metadata["source_url"],
                },
                "redistribution_recipe": {
                    "filename": recipe_path.name,
                    "sha256": sha256(recipe_path),
                    "commit": metadata["feedstock_commit"],
                },
            }
        )
        text.extend(
            [
                "",
                "Gmsh 4.15.2",
                "Copyright (C) 1997-2026 C. Geuzaine and J.-F. Remacle.",
                "License: GNU GPL version 2 or later, with the exception in the bundled license.",
                f"License file: share/agentfem/GMSH-LICENSE.txt",
                f"Corresponding source: {source_name} (published beside this installer)",
                f"Conda-forge build recipe: {recipe_name} (published beside this installer)",
            ]
        )
    else:
        gmsh_license_target.write_text(
            "Gmsh is not included in the AgentFEM Core runtime.\n",
            encoding="utf-8",
        )
    notices.write_text("\n".join(text) + "\n", encoding="utf-8")
    record_path.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return notices, record_path


def ensure_release_source(*, allow_dirty: bool) -> dict[str, Any]:
    identity = source_identity()
    if identity["dirty"] and not allow_dirty:
        raise SystemExit(
            "Refusing to build a release runtime from a dirty checkout. "
            "Commit the candidate or pass --allow-dirty for a local prototype."
        )
    return identity


def build_wheel(candidate: str | None = None) -> Path:
    wheel_dir = BUILD / "wheels"
    wheel_dir.mkdir(parents=True, exist_ok=True)
    for existing in wheel_dir.glob("*.whl"):
        existing.unlink()
    if candidate:
        supplied = Path(candidate).expanduser().resolve()
        if not supplied.is_file():
            raise SystemExit(f"AgentFEM wheel does not exist: {supplied}")
        if not supplied.name.startswith("agentfem-") or not supplied.name.endswith(
            ".whl"
        ):
            raise SystemExit(
                "The supplied release wheel must retain its canonical "
                "agentfem-<version>-<tags>.whl filename."
            )
        copied = wheel_dir / supplied.name
        shutil.copy2(supplied, copied)
        return copied
    run(
        [sys.executable, "-m", "pip", "wheel", ".", "--no-deps", "--no-build-isolation", "--wheel-dir", str(wheel_dir)],
        cwd=ROOT,
    )
    wheels = list(wheel_dir.glob("*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(f"Expected one wheel, found {wheels}")
    return wheels[0]


def profile_specs(profile: str) -> Path:
    suffix = "" if profile == "core" else f"-{profile}"
    path = RUNTIME / f"runtime-specs{suffix}.txt"
    if not path.is_file():
        raise SystemExit(f"Unknown or incomplete runtime profile: {profile}")
    return path


def _direct_spec_names(specs: Path) -> set[str]:
    names: set[str] = set()
    for raw_line in specs.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        names.add(line.split("=", 1)[0].strip())
    return names


def _linked_records(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the complete solved environment, independent of local caches."""

    return list(payload.get("actions", {}).get("LINK", []))


def resolve_lock(target_platform: str, *, profile: str) -> Path:
    lock_dir = BUILD / "locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock = lock_dir / f"{target_platform}-{profile}.txt"
    with tempfile.TemporaryDirectory(prefix="agentfem-solve-", dir=BUILD) as prefix:
        # A private empty cache makes FETCH contain every solved artifact,
        # including its real .conda/.tar.bz2 filename and cryptographic hash.
        cache = Path(prefix) / "pkgs"
        cache.mkdir()
        solver_env = {**os.environ, "CONDA_PKGS_DIRS": str(cache)}
        # Cross-solving a Linux environment from macOS has no native
        # ``__glibc`` virtual package.  Declare the oldest runtime ABI that
        # AgentFEM's WSL2 image supports so conda can resolve Linux packages
        # without accidentally targeting the host operating system.
        if target_platform == "linux-64":
            solver_env["CONDA_OVERRIDE_GLIBC"] = "2.17"
        completed = run_with_retries(
            [
                "conda",
                "create",
                "--dry-run",
                "--json",
                "--yes",
                "--platform",
                target_platform,
                "--prefix",
                prefix,
                "--channel",
                "conda-forge",
                "--override-channels",
                "--file",
                str(profile_specs(profile)),
            ],
            cwd=ROOT,
            capture_output=True,
            env=solver_env,
        )
    payload = json.loads(completed.stdout)
    # FETCH is cache-dependent and can omit direct requirements already held by
    # the solver (notably pip/conda on hosted runners). LINK is the complete
    # solved environment and therefore the only sound basis for an explicit,
    # portable runtime lock.
    records = _linked_records(payload)
    urls: list[str] = []
    locked_records: list[dict[str, Any]] = []
    for record in records:
        url = record.get("url")
        if not url:
            base = record.get("base_url") or record.get("channel")
            filename = record.get("fn")
            if base and filename:
                url = f"{str(base).rstrip('/')}/{filename}"
        if not url:
            raise RuntimeError(f"Cannot recover an exact URL for {record}")
        package_sha256 = record.get("sha256")
        exact_url = str(url)
        if package_sha256 and "#" not in exact_url:
            exact_url = f"{exact_url}#{package_sha256}"
        urls.append(exact_url)
        locked_records.append(
            {
                key: record.get(key)
                for key in (
                    "name",
                    "version",
                    "build",
                    "build_number",
                    "subdir",
                    "license",
                    "sha256",
                    "md5",
                    "size",
                )
            }
            | {"url": str(url)}
        )
    if not urls:
        raise RuntimeError("Conda returned no locked packages")
    locked_names = {str(record.get("name")) for record in locked_records}
    missing_direct = _direct_spec_names(profile_specs(profile)) - locked_names
    if missing_direct:
        raise RuntimeError(
            "Conda's solved environment omitted direct runtime requirements: "
            + ", ".join(sorted(missing_direct))
        )
    lock.write_text("@EXPLICIT\n" + "\n".join(urls) + "\n", encoding="utf-8")
    lock.with_suffix(".json").write_text(
        json.dumps(
            {
                "schema": "agentfem.conda-lock",
                "schema_version": 1,
                "platform": target_platform,
                "profile": profile,
                "packages": locked_records,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return lock


def write_sbom(lock: Path, *, target: str, profile: str, output: Path) -> Path:
    lock_record = json.loads(lock.with_suffix(".json").read_text(encoding="utf-8"))
    components = []
    for package in lock_record["packages"]:
        name = str(package["name"])
        version = str(package["version"])
        build = str(package["build"])
        subdir = str(package["subdir"])
        component: dict[str, Any] = {
            "type": "library",
            "name": name,
            "version": version,
            "purl": f"pkg:conda/{name}@{version}?build={build}&subdir={subdir}",
            "properties": [
                {"name": "agentfem:conda:url", "value": str(package["url"])},
                {"name": "agentfem:conda:build", "value": build},
                {"name": "agentfem:conda:subdir", "value": subdir},
            ],
        }
        if package.get("sha256"):
            component["hashes"] = [
                {"alg": "SHA-256", "content": package["sha256"]}
            ]
        if package.get("license"):
            component["licenses"] = [
                {"expression": str(package["license"])}
            ]
        components.append(component)
    product = "Complete" if profile == "complete" else "Core"
    path = output / f"AgentFEM-{product}-{project_version()}-{target}-SBOM.cdx.json"
    path.write_text(
        json.dumps(
            {
                "bomFormat": "CycloneDX",
                "specVersion": "1.6",
                "version": 1,
                "metadata": {
                    "component": {
                        "type": "application",
                        "name": f"AgentFEM {product} Runtime",
                        "version": project_version(),
                    },
                    "properties": [
                        {"name": "agentfem:target", "value": target},
                        {"name": "agentfem:profile", "value": profile},
                    ],
                },
                "components": components,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def publish_runtime_evidence(
    *,
    lock: Path,
    notices: Path,
    components: Path,
    output: Path,
    target: str,
    profile: str,
) -> dict[str, Path]:
    """Publish inspectable evidence beside, as well as inside, an installer."""

    product = "Complete" if profile == "complete" else "Core"
    stem = f"AgentFEM-{product}-{project_version()}-{target}"
    published = {
        "lock": output / f"{stem}-conda-lock.txt",
        "lock_record": output / f"{stem}-conda-lock.json",
        "components": output / f"{stem}-third-party.json",
        "notices": output / f"AgentFEM-{product}-{project_version()}-THIRD_PARTY_NOTICES.txt",
    }
    shutil.copy2(lock, published["lock"])
    shutil.copy2(lock.with_suffix(".json"), published["lock_record"])
    shutil.copy2(components, published["components"])
    shutil.copy2(notices, published["notices"])
    if profile == "complete":
        published["gmsh_license"] = output / "GMSH-LICENSE.txt"
        shutil.copy2(BUILD / "GMSH-LICENSE.txt", published["gmsh_license"])
    return published


def release_record(
    *, wheel: Path, lock: Path, target: str, source: dict[str, Any], profile: str
) -> dict[str, Any]:
    packages = [
        line.strip()
        for line in lock.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("@")
    ]
    return {
        "schema": SCHEMA,
        "schema_version": 1,
        "agentfem_version": project_version(),
        "profile": profile,
        "target": target,
        "source": source,
        "wheel": {"filename": wheel.name, "sha256": sha256(wheel)},
        "packages": packages,
        "included_optional_components": ["gmsh"] if profile == "complete" else [],
        "optional_packs_excluded": (
            ["learning", "visualization"]
            if profile == "complete"
            else ["gmsh", "learning", "visualization"]
        ),
    }


def build_macos(args: argparse.Namespace) -> Path:
    if sys.platform != "darwin" or platform.machine() != "arm64":
        raise SystemExit("The first macOS runtime must be built on Apple Silicon.")
    source = ensure_release_source(allow_dirty=args.allow_dirty)
    wheel = build_wheel(args.wheel)
    lock = resolve_lock("osx-arm64", profile=args.profile)
    input_dir = BUILD / "macos-input"
    if input_dir.exists():
        shutil.rmtree(input_dir)
    input_dir.mkdir(parents=True)
    shutil.copy2(lock, BUILD / "runtime-lock.txt")
    shutil.copy2(wheel, input_dir / wheel.name)
    for name in ("README.txt", "post_install.sh"):
        shutil.copy2(RUNTIME / "macos" / name, input_dir / name)
    record = release_record(
        wheel=wheel,
        lock=lock,
        target="macos-arm64",
        source=source,
        profile=args.profile,
    )
    (input_dir / "runtime-release.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    app_identity = bool(os.environ.get("AGENTFEM_APP_IDENTITY"))
    installer_identity = bool(os.environ.get("AGENTFEM_INSTALLER_IDENTITY"))
    if app_identity != installer_identity:
        raise SystemExit(
            "A macOS release requires both Developer ID Application and "
            "Developer ID Installer identities."
        )
    signed = app_identity and installer_identity
    notarized = signed and bool(os.environ.get("AGENTFEM_NOTARY_PROFILE"))
    suffix = "" if notarized else "-unsigned-preview"
    product = "AgentFEM-Complete" if args.profile == "complete" else "AgentFEM-Core"
    filename = f"{product}-{project_version()}-macOS-arm64{suffix}.pkg"
    template = (RUNTIME / "macos" / "construct.yaml.in").read_text(
        encoding="utf-8"
    )
    template = template.replace("@VERSION@", project_version()).replace(
        "@INSTALLER_FILENAME@", filename
    )
    template = template.replace("@WHEEL_FILENAME@", wheel.name)
    (input_dir / "construct.yaml").write_text(template, encoding="utf-8")
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    notices, components = prepare_legal_materials(args.profile, lock, output)
    publish_runtime_evidence(
        lock=lock,
        notices=notices,
        components=components,
        output=output,
        target="macOS-arm64",
        profile=args.profile,
    )
    sbom = write_sbom(
        lock,
        target="macOS-arm64",
        profile=args.profile,
        output=output,
    )
    shutil.copy2(notices, input_dir / notices.name)
    shutil.copy2(components, input_dir / components.name)
    shutil.copy2(sbom, input_dir / "runtime-sbom.cdx.json")
    shutil.copy2(BUILD / "GMSH-LICENSE.txt", input_dir / "GMSH-LICENSE.txt")
    # Do not let Finder metadata or resource forks become ``._*`` payload
    # files in the scientific runtime archive.
    run(["xattr", "-cr", str(input_dir)], cwd=ROOT)
    local_constructor = BUILD / "constructor" / "bin" / "constructor"
    constructor = shutil.which("constructor") or (
        str(local_constructor) if local_constructor.exists() else None
    )
    if not constructor:
        raise SystemExit(
            "constructor is not installed. Create an isolated build environment "
            "with `conda create -p build/runtime/constructor -c conda-forge constructor`."
        )
    constructor_command = [
        constructor,
        str(input_dir),
        "--output-dir",
        str(output),
    ]
    # An explicit override must be constructor-compatible (conda-standalone
    # or micromamba), not an ordinary activated conda frontend.  Constructor
    # otherwise supplies its own standalone executable.
    conda_executable = os.environ.get("CONSTRUCTOR_CONDA_EXE")
    if conda_executable:
        constructor_command.extend(["--conda-exe", conda_executable])
    run_with_retries(
        constructor_command,
        cwd=ROOT,
        env={**os.environ, "COPYFILE_DISABLE": "1"},
    )
    artifact = output / filename
    if not artifact.exists():
        candidates = sorted(output.glob("*.pkg"), key=lambda item: item.stat().st_mtime)
        if not candidates:
            raise RuntimeError("constructor did not create a PKG")
        candidates[-1].replace(artifact)
    if notarized:
        run(
            [
                str(RUNTIME / "macos" / "sign_and_notarize.sh"),
                str(artifact),
                os.environ["AGENTFEM_NOTARY_PROFILE"],
            ],
            cwd=ROOT,
        )
    return artifact


def build_wsl(args: argparse.Namespace) -> Path:
    source = ensure_release_source(allow_dirty=args.allow_dirty)
    wheel = build_wheel(args.wheel)
    lock = resolve_lock("linux-64", profile=args.profile)
    shutil.copy2(lock, BUILD / "runtime-lock-linux-64.txt")
    # Docker COPY uses the stable name below and the repository root as context.
    docker_lock = BUILD / "runtime-lock.txt"
    shutil.copy2(lock, docker_lock)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    notices, components = prepare_legal_materials(args.profile, lock, output)
    published = publish_runtime_evidence(
        lock=lock,
        notices=notices,
        components=components,
        output=output,
        target="WSL2-x86_64",
        profile=args.profile,
    )
    sbom = write_sbom(
        lock,
        target="WSL2-x86_64",
        profile=args.profile,
        output=output,
    )
    shutil.copy2(sbom, BUILD / "runtime-sbom.cdx.json")
    record = release_record(
        wheel=wheel,
        lock=lock,
        target="wsl2-x86_64",
        source=source,
        profile=args.profile,
    )
    embedded_record = BUILD / "runtime-release.json"
    embedded_record.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    tag = f"agentfem-runtime:{project_version()}-{args.profile}-wsl2-amd64"
    run(
        [
            "docker",
            "build",
            "--platform",
            "linux/amd64",
            "--build-arg",
            f"AGENTFEM_VERSION={project_version()}",
            "--tag",
            tag,
            "--file",
            str(RUNTIME / "wsl" / "Dockerfile"),
            ".",
        ],
        cwd=ROOT,
    )
    container = run(["docker", "create", tag], capture_output=True).stdout.strip()
    raw_tar = BUILD / "agentfem-wsl-rootfs.tar"
    try:
        run(["docker", "export", "--output", str(raw_tar), container])
    finally:
        subprocess.run(["docker", "rm", container], check=False)
    product = "AgentFEM-Complete" if args.profile == "complete" else "AgentFEM-Core"
    artifact = output / f"{product}-{project_version()}-WSL2-x86_64.wsl"
    with raw_tar.open("rb") as source_stream, gzip.open(
        artifact, "wb", compresslevel=9
    ) as target_stream:
        shutil.copyfileobj(source_stream, target_stream, length=1024 * 1024)
    raw_tar.unlink()
    record_path = output / "runtime-release-wsl2-x86_64.json"
    record_path.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    installer = output / "Install-AgentFEM.ps1"
    installer_template = (RUNTIME / "wsl" / "Install-AgentFEM.ps1").read_text(
        encoding="utf-8"
    )
    installer.write_text(
        installer_template.replace("@IMAGE_FILENAME@", artifact.name).replace(
            "@IMAGE_SHA256@", sha256(artifact)
        ),
        encoding="utf-8",
    )
    bundle = output / f"{product}-{project_version()}-WSL2-x86_64-preview-offline.zip"
    bundle_files = [
        artifact,
        installer,
        record_path,
        sbom,
        published["lock"],
        published["lock_record"],
        published["notices"],
        published["components"],
    ]
    if args.profile == "complete":
        version = "4.15.2"
        metadata = GMSH_RELEASES[version]
        bundle_files.extend(
            [
                published["gmsh_license"],
                output / f"Gmsh-{version}-corresponding-source.tar.gz",
                output
                / (
                    f"Gmsh-{version}-conda-forge-recipe-"
                    f"{str(metadata['feedstock_commit'])[:12]}.tar.gz"
                ),
            ]
        )
    with zipfile.ZipFile(bundle, "w") as archive:
        for path in bundle_files:
            # The WSL image, source archive and recipe archive are already
            # compressed.  Storing them avoids an expensive second pass and
            # makes the offline bundle deterministic and quick to assemble.
            compression = (
                zipfile.ZIP_STORED
                if path.suffix in {".wsl", ".gz"}
                else zipfile.ZIP_DEFLATED
            )
            archive.write(path, path.name, compress_type=compression)
    return bundle


def write_manifest(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    ignored = {"SHA256SUMS", "runtime-artifacts.json"}
    artifacts = []
    for path in sorted(item for item in output_dir.iterdir() if item.is_file()):
        if path.name in ignored:
            continue
        artifacts.append(
            {"filename": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
        )
    manifest = {
        "schema": "agentfem.runtime-artifacts",
        "schema_version": 1,
        "agentfem_version": project_version(),
        "source": source_identity(),
        "artifacts": artifacts,
    }
    target = output_dir / "runtime-artifacts.json"
    target.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    checksums = output_dir / "SHA256SUMS"
    checksums.write_text(
        "".join(f"{item['sha256']}  {item['filename']}\n" for item in artifacts),
        encoding="utf-8",
    )
    return target


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("target", choices=("macos", "wsl", "manifest"))
    result.add_argument(
        "--profile",
        choices=("core", "complete"),
        default="complete",
        help="complete is the recommended runtime and includes separately licensed Gmsh",
    )
    result.add_argument(
        "--wheel",
        help="embed an existing release wheel instead of rebuilding it from the checkout",
    )
    result.add_argument("--output-dir", default=str(ROOT / "dist" / "runtime"))
    result.add_argument(
        "--allow-dirty",
        action="store_true",
        help="build a visibly non-promotable local prototype from a dirty checkout",
    )
    return result


def main() -> int:
    args = parser().parse_args()
    if args.target == "macos":
        artifact = build_macos(args)
    elif args.target == "wsl":
        artifact = build_wsl(args)
    else:
        artifact = write_manifest(Path(args.output_dir).resolve())
    print(artifact)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
