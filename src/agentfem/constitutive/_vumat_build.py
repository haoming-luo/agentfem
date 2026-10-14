# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Private, content-addressed build of an explicitly trusted legacy VUMAT."""
import hashlib
import json
from pathlib import Path
import platform
import re
import shutil
import subprocess
import tempfile

from .user_material import inspect_abaqus_user_material


_WRAPPER = """subroutine agentfem_vumat_batch(n,ns,np,nf,st,tt,dt,props,rho,xyz,length,de,spin, &
    to,uo,fo,fvo,so,qo,eo,io,tn,un,fn,fvn,sn,qn,en,inn,cname) bind(C)
  use iso_c_binding
  implicit none
  integer(c_int), value :: n,ns,np,nf
  real(c_double), value :: st,tt,dt
  real(c_double) :: props(np),rho(n),xyz(n,3),length(n),de(n,6),spin(n,3)
  real(c_double) :: to(n),uo(n,6),fo(n,9),fvo(n,nf),so(n,6),qo(n,ns),eo(n),io(n)
  real(c_double) :: tn(n),un(n,6),fn(n,9),fvn(n,nf),sn(n,6),qn(n,ns),en(n),inn(n)
  character(kind=c_char) :: cname(80)
  character(len=80) :: name
  integer :: i
  do i=1,80
    name(i:i)=cname(i)
  end do
  call vumat(n,3,3,ns,nf,np,0,st,tt,dt,name,xyz,length,props,rho,de,spin, &
    to,uo,fo,fvo,so,qo,eo,io,tn,un,fn,fvn,sn,qn,en,inn)
end subroutine
"""
_SIGNATURE = ("nblock ndir nshr nstatev nfieldv nprops lanneal steptime totaltime dt "
              "cmname coordmp charlength props density straininc relspininc tempold "
              "stretchold defgradold fieldold stressold stateold enerinternold "
              "enerinelasold tempnew stretchnew defgradnew fieldnew stressnew "
              "statenew enerinternnew enerinelasnew").split()
_FLAGS = ("-O3", "-fno-fast-math", "-ffp-contract=off", "-fPIC", "-shared",
          "-frecursive", "-fdefault-real-8", "-fdefault-double-8",
          "-ffree-line-length-none", "-ffixed-line-length-none")


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _check_signature(source):
    text = source.read_text()
    if source.suffix.lower() in (".f", ".for"):
        lines = [line[6:] for line in text.splitlines()
                 if len(line) > 6 and line[0] not in "cC*!" ]
    else:
        lines = text.splitlines()
    code = " ".join(line.split("!", 1)[0] for line in lines).replace("&", " ").lower()
    match = re.search(r"\bsubroutine\s+vumat\s*\(([^)]*)\)", code)
    if not match or [v.strip() for v in match[1].split(",")] != _SIGNATURE:
        raise ValueError("AFM-VUMAT-ABI: only the reviewed legacy scalar-dt signature is supported")


def build_vumat(source, *, cache_directory, trusted, compiler):
    if trusted is not True:
        raise ValueError("AFM-VUMAT-TRUST: compilation/loading requires trusted=True after source review")
    if platform.system() not in ("Darwin", "Linux"):
        raise RuntimeError("AFM-VUMAT-PLATFORM: use macOS, Linux or WSL2")
    import fcntl
    report = inspect_abaqus_user_material(source, kind="VUMAT")
    if report.status != "adapter_candidate" or report.external_calls:
        raise ValueError("AFM-VUMAT-SOURCE: unresolved source dependencies; " + report.format())
    root = Path(source).resolve()
    _check_signature(root)
    files = {}
    for item in report.source_graph.files:
        relative = Path(item.logical_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("AFM-VUMAT-INCLUDE: keep reviewed includes inside the source directory")
        if relative.parts[0].lower() in {
            "agentfem-wrapper.f90", "build.json", "build.log", "vumat.so", "vumat.dylib"
        }:
            raise ValueError("AFM-VUMAT-SOURCE: source path conflicts with generated build artifacts")
        files[relative.as_posix()] = item.path.read_bytes()
        if _digest(files[relative.as_posix()]) != item.source_sha256:
            raise ValueError("AFM-VUMAT-SOURCE: source changed during inspection")
    if set(report.source_graph.runtime_includes) - {"VABA_PARAM.INC"}:
        raise ValueError("AFM-VUMAT-INCLUDE: unsupported vendor runtime header")
    executable = shutil.which(str(compiler))
    if executable is None:
        raise RuntimeError("AFM-VUMAT-COMPILER: GNU Fortran is required; no compiler will be installed automatically")
    version = subprocess.check_output([executable, "--version"], text=True, timeout=10).splitlines()[0]
    if "GNU Fortran" not in version:
        raise ValueError("AFM-VUMAT-COMPILER: only GNU Fortran is admitted")
    header = b"      implicit real*8(a-h,o-z)\n"
    identity = {"schema": "agentfem.legacy-vumat-build.v1", "interface": "legacy_3d_binary64",
                "sources": {k: _digest(v) for k, v in sorted(files.items())},
                "wrapper": _digest(_WRAPPER.encode()), "parameter_header": _digest(header),
                "compiler": version, "flags": list(_FLAGS),
                "system": platform.system(), "machine": platform.machine()}
    key = _digest(json.dumps(identity, sort_keys=True).encode())
    cache = Path(cache_directory).expanduser().resolve()
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / key
    library_name = "vumat.dylib" if platform.system() == "Darwin" else "vumat.so"
    with (cache / "build.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not target.exists():
            stage = Path(tempfile.mkdtemp(prefix="pending-", dir=cache))
            for relative, data in files.items():
                destination = stage / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(data)
            for edge in report.source_graph.edges:
                if edge.status == "runtime_provided":
                    declaration = Path(edge.declaration)
                    if declaration.is_absolute() or ".." in declaration.parts:
                        raise ValueError("AFM-VUMAT-INCLUDE: unsupported runtime header path")
                    destination = stage / Path(edge.source).parent / declaration
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(header)
            (stage / "agentfem-wrapper.f90").write_text(_WRAPPER)
            command = [executable, *_FLAGS, "-I" + str(stage),
                       str(stage / "agentfem-wrapper.f90"), str(stage / root.name),
                       "-o", str(stage / library_name)]
            try:
                with (stage / "build.log").open("w") as log:
                    subprocess.run(command, cwd=stage, stdout=log, stderr=subprocess.STDOUT,
                                   check=True, timeout=180)
            except (subprocess.SubprocessError, OSError) as error:
                raise RuntimeError(f"AFM-VUMAT-BUILD: failed; diagnostic files retained in {stage}") from error
            record = {"identity": identity, "library_sha256": _digest((stage / library_name).read_bytes())}
            (stage / "build.json").write_text(json.dumps(record, sort_keys=True, indent=2))
            stage.rename(target)
        library = target / library_name
        try:
            record = json.loads((target / "build.json").read_text())
            valid = (record["identity"] == identity
                     and record["library_sha256"] == _digest(library.read_bytes()))
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise ValueError("AFM-VUMAT-CACHE: unreadable compiled material record") from error
        if not valid:
            raise ValueError("AFM-VUMAT-CACHE: compiled material integrity mismatch")
    return library, record
