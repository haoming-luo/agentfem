# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
import ctypes
import shutil
import pytest
from agentfem.constitutive import build_native_libraries


def setup_bundle(root):
    (root / "law.f90").write_text('subroutine twice(x) bind(C)\nuse iso_c_binding\nreal(c_double)::x\nx=2*x\nend subroutine\n')
    manifest = root / "material.toml"
    manifest.write_text('schema="agentfem.native-libraries.v1"\nfiles=["law.f90"]\n[targets]\nlaw=["law.f90"]\n')
    return manifest


def test_trust_and_paths_before_compilation(tmp_path):
    manifest = setup_bundle(tmp_path)
    with pytest.raises(ValueError, match="TRUST"):
        build_native_libraries(manifest, cache_directory=tmp_path / "cache")
    manifest.write_text('schema="agentfem.native-libraries.v1"\nfiles=["../escape.for"]\n')
    with pytest.raises(ValueError, match="PATH"):
        build_native_libraries(manifest, trusted=True, cache_directory=tmp_path / "cache")


@pytest.mark.skipif(shutil.which("gfortran") is None, reason="GNU Fortran required")
def test_real_compile_reuse_invalidation_and_integrity(tmp_path):
    manifest = setup_bundle(tmp_path)
    args = dict(trusted=True, cache_directory=tmp_path / "cache")
    bundle = build_native_libraries(manifest, **args)
    lib = ctypes.CDLL(str(bundle.library("law")))
    lib.twice.argtypes = [ctypes.POINTER(ctypes.c_double)]
    lib.twice.restype = None
    value = ctypes.c_double(3)
    lib.twice(ctypes.byref(value))
    assert value.value == 6
    stamp = bundle.library("law").stat().st_mtime_ns
    assert build_native_libraries(manifest, **args).library("law").stat().st_mtime_ns == stamp
    (tmp_path / "law.f90").write_text((tmp_path / "law.f90").read_text().replace("2*x", "3*x"))
    changed = build_native_libraries(manifest, **args)
    assert changed.directory != bundle.directory
    changed.library("law").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="CACHE"):
        build_native_libraries(manifest, **args)


@pytest.mark.skipif(shutil.which("gfortran") is None, reason="GNU Fortran required")
def test_failed_build_keeps_diagnostics_not_completed_target(tmp_path):
    manifest = setup_bundle(tmp_path)
    (tmp_path / "law.f90").write_text("not a Fortran subroutine")
    with pytest.raises(RuntimeError, match="diagnostics retained"):
        build_native_libraries(manifest, trusted=True, cache_directory=tmp_path / "cache")
    assert list((tmp_path / "cache").glob("pending-*/law.log"))
    assert not list((tmp_path / "cache").glob("*/build.json"))


@pytest.mark.skipif(shutil.which("gfortran") is None, reason="GNU Fortran required")
@pytest.mark.parametrize("asset", ["polar_series.f90", "polar_wider.f90"])
def test_bundled_corotational_native_and_reference_repair(tmp_path, asset):
    import numpy as np
    from agentfem.constitutive import CompiledCorotationalIncrement, polar_strain_increment
    manifest = tmp_path / "material.toml"
    manifest.write_text('schema="agentfem.native-libraries.v1"\n'
        '[bundled_sources]\n"polar.f90"="'+asset+'"\n[targets]\npolar=["polar.f90"]\n')
    bundle = build_native_libraries(manifest, trusted=True, cache_directory=tmp_path / "cache")
    native = CompiledCorotationalIncrement(bundle, "polar")
    rng = np.random.default_rng(21)
    old = np.repeat(np.eye(3)[None], 65, axis=0) + rng.normal(scale=.002, size=(65,3,3))
    new = old + rng.normal(scale=.001, size=old.shape)
    # Last point is outside both fast gates and must take exact reference repair.
    old[-1] = np.diag([1.8,.9,1.1]); new[-1] = old[-1] + np.diag([.002,.001,-.001])
    expected = polar_strain_increment(old, new)
    actual = native(old, new)
    for name in ("strain_increment", "frame_old", "frame_new"):
        np.testing.assert_allclose(getattr(actual,name), getattr(expected,name), atol=2e-14, rtol=2e-11)
        assert not getattr(actual,name).flags.writeable
    invalid = old.copy(); invalid[0,0,0] = -1
    with pytest.raises(ValueError):
        native(invalid, new)


def test_undeclared_include_is_rejected(tmp_path):
    manifest = setup_bundle(tmp_path)
    (tmp_path / "law.f90").write_text('include "missing.inc"\n')
    with pytest.raises(ValueError, match="INCLUDE"):
        build_native_libraries(manifest, trusted=True, cache_directory=tmp_path / "cache")
