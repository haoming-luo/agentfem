# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
from dataclasses import replace
import json
import shutil

import numpy as np
import pytest

from agentfem import constitutive as c, elements
from agentfem.constitutive.material_array_batch import MaterialPointArrayBatchInput
from test_vumat_bridge import SOURCE


def manifest(tmp_path, ns=3, **changes):
    spec = dict(schema="agentfem.strain-history.v1", source="law.f90",
        property_names=["lambda", "mu"], state_count=ns,
        unused_channels=["coordinates", "characteristic_length", "relative_spin", "temperature",
            "stretch", "deformation_gradient", "fields", "absolute_time", "material_name", "old_energy_for_response"],
        kinematics="linear_right_stretch_path", energy="incremental_internal_work",
        maximum_stretch_increment=.01, increment_ceiling=.1, thread_safe=True)
    spec.update(changes)
    (tmp_path / "law.f90").write_text(SOURCE)
    path = tmp_path / "material.toml"
    path.write_text("\n".join(k + " = " + json.dumps(v) for k, v in spec.items()))
    return path


def runtime(path, **kw):
    pytest.importorskip("agentfem._body_pipeline")
    if shutil.which("gfortran") is None:
        pytest.skip("GNU Fortran unavailable")
    return c.StrainHistoryRuntime(path, parameters={"lambda": 4., "mu": 3.}, density=2.,
        cache_directory=path.parent/"cache", trusted=True, **kw)


def request(rt, n=11):
    old = np.tile(np.eye(3), (n, 1, 1))
    new = old.copy()
    new[:, 0, 0] += .001
    new[:, 0, 1] = .0003
    return MaterialPointArrayBatchInput(old, new, np.zeros((n, rt.state_schema.size)),
        rt.state_schema, .01, .001)


@pytest.mark.parametrize("ns", [0, 1, 7, 29])
@pytest.mark.parametrize("workers", [1, 3])
def test_fused_generic_history_matches_reference(tmp_path, ns, workers):
    with runtime(manifest(tmp_path, ns), workers=workers) as rt:
        r = request(rt)
        rt.increment_limit(r)
        expected = rt.update(r)
        rng = np.random.default_rng(15)
        g = elements.ExplicitHexGeometry(rng.normal(size=(11,8,3)),
            rng.normal(size=(11,8,4)), np.ones(11), np.full(11, .01))
        displacement = rng.normal(size=(11,8,3))
        actual = rt.kernel.evaluate(rt.material, r, g, displacement).material_response
        for key in ("cauchy_stress", "state_new", "internal_work_density_increment"):
            np.testing.assert_allclose(getattr(actual, key), getattr(expected, key), rtol=1e-12, atol=1e-14)
        np.testing.assert_array_equal(r.state_old, 0)
        np.testing.assert_allclose(expected.state_new[:, :ns], np.tile(.001*np.arange(1, ns+1), (11, 1)))
        # Repeated trial evaluation must not advance hidden state.
        np.testing.assert_array_equal(rt.update(r).state_new, expected.state_new)
        assert rt.summary()["spec"]["state_count"] == ns


def test_limits_channels_and_failure_do_not_change_state(tmp_path):
    with runtime(manifest(tmp_path)) as rt:
        r = request(rt)
        f = r.deformation_gradient_new.copy(); f[:, 0, 0] = 1.1
        with pytest.raises(c.ExplicitMaterialCutbackRequest, match="AFM-HISTORY-INCREMENT"):
            rt.increment_limit(replace(r, deformation_gradient_new=f))
        with pytest.raises(ValueError, match="AFM-HISTORY-CHANNELS"):
            rt.update(replace(r, temperature=np.ones(11)))
        np.testing.assert_array_equal(r.state_old, 0)
        np.testing.assert_allclose(rt.increment_limit(r).maximum_time_increment, .1)


@pytest.mark.parametrize("change", [dict(state_count=True), dict(unused_channels=[]),
    dict(property_names=["mu", "mu"]), dict(energy="stored_energy"), dict(source="../law.f90"),
    dict(maximum_stretch_increment=0), dict(increment_ceiling=-1), dict(thread_safe="yes")])
def test_bad_declaration_rejected_before_compilation(tmp_path, change):
    with pytest.raises(ValueError, match="AFM-HISTORY"):
        runtime(manifest(tmp_path, **change))


def test_trust_and_thread_safety(tmp_path):
    path = manifest(tmp_path, thread_safe=False)
    with pytest.raises(ValueError, match="AFM-HISTORY-TRUST"):
        c.StrainHistoryRuntime(path, parameters={}, density=1., cache_directory=tmp_path/"cache")
    with pytest.raises(ValueError, match="AFM-HISTORY-THREADS"):
        runtime(path, workers=2)
    with pytest.raises(ValueError, match="AFM-HISTORY-THREADS"):
        runtime(path, workers=1)


def test_bound_and_missing_work_reject_unpublished_response(tmp_path):
    path = manifest(tmp_path)
    with path.open("a") as f:
        f.write('\nstate_bounds = [{index=1, minimum=0.0, maximum=0.001, tolerance=0.0}]\n')
    with runtime(path) as rt:
        r = request(rt)
        with pytest.raises(ValueError, match="AFM-HISTORY-STATE"):
            rt.update(r)
        np.testing.assert_array_equal(r.state_old, 0)
    path = manifest(tmp_path)
    (tmp_path / "law.f90").write_text(SOURCE.replace(
        "enerinternnew(k)=enerinternold(k)+work/density(k)", "! unavailable"))
    with runtime(path) as rt:
        with pytest.raises(ValueError, match="AFM-HISTORY-ENERGY"):
            rt.update(request(rt))
        r = request(rt)
        geometry = elements.ExplicitHexGeometry(np.zeros((11,8,3)), np.zeros((11,8,4)),
                                               np.ones(11), np.zeros(11))
        with pytest.raises(ValueError):
            rt.kernel.evaluate(rt.material, r, geometry, np.zeros((11,8,3)))
        np.testing.assert_array_equal(r.state_old, 0)


def test_repair_and_resume_material_history(tmp_path):
    with runtime(manifest(tmp_path)) as rt:
        r = request(rt)
        # A far-from-identity accepted deformation forces the polar reference repair.
        old = np.tile(np.diag([2., 1., 1.]), (11,1,1))
        new = old.copy(); new[:, 0, 0] += .001
        r = replace(r, deformation_gradient_old=old, deformation_gradient_new=new)
        rng = np.random.default_rng(3)
        g = elements.ExplicitHexGeometry(rng.normal(size=(11,8,3)),
            rng.normal(size=(11,8,4)), np.ones(11), np.full(11,.01))
        u = np.zeros((11,8,3))
        actual = rt.kernel.evaluate(rt.material, r, g, u).material_response
        expected = rt.update(r)
        np.testing.assert_array_equal(actual.state_new, expected.state_new)
        np.testing.assert_array_equal(actual.cauchy_stress, expected.cauchy_stress)
        np.savez(tmp_path/"accepted.npz", state=actual.state_new, gradient=new)
        with np.load(tmp_path/"accepted.npz", allow_pickle=False) as data:
            resumed = replace(r, state_old=data["state"], deformation_gradient_old=data["gradient"])
        next_request = replace(r, state_old=expected.state_new, deformation_gradient_old=new)
        np.testing.assert_array_equal(rt.update(resumed).state_new, rt.update(next_request).state_new)


def test_mutated_metadata_rejected_before_native_pointer_call(tmp_path):
    with runtime(manifest(tmp_path)) as rt:
        r = request(rt)
        r.deformation_gradient_old.dtype = np.int64
        with pytest.raises(ValueError, match="AFM-HISTORY-STATE"):
            rt.increment_limit(r)
        r = request(rt)
        r.state_old.shape = (1, -1)
        with pytest.raises(ValueError, match="AFM-HISTORY-STATE"):
            rt.increment_limit(r)


def test_parameter_domain_is_declared_not_inferred(tmp_path):
    path = manifest(tmp_path)
    with path.open("a") as f:
        f.write('\nparameter_bounds = [{name="mu", maximum=3.0, exclusive_maximum=true}]\n')
    with pytest.raises(ValueError, match="AFM-HISTORY-PARAMETERS"):
        runtime(path)
