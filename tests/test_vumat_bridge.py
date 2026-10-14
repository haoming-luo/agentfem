# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
from dataclasses import replace
import json
import shutil

import numpy as np
import pytest

from agentfem import constitutive as c


# Independent, deliberately simple public test law. No confidential source.
SOURCE = """subroutine vumat(nblock,ndir,nshr,nstatev,nfieldv,nprops,lanneal, &
    steptime,totaltime,dt,cmname,coordmp,charlength,props,density,straininc,relspininc, &
    tempold,stretchold,defgradold,fieldold,stressold,stateold,enerinternold,enerinelasold, &
    tempnew,stretchnew,defgradnew,fieldnew,stressnew,statenew,enerinternnew,enerinelasnew)
  include 'vaba_param.inc'
  integer nblock,ndir,nshr,nstatev,nfieldv,nprops,lanneal,k,j
  character*80 cmname
  dimension coordmp(nblock,3),charlength(nblock),props(nprops),density(nblock)
  dimension straininc(nblock,6),relspininc(nblock,3),tempold(nblock),stretchold(nblock,6)
  dimension defgradold(nblock,9),fieldold(nblock,nfieldv),stressold(nblock,6)
  dimension stateold(nblock,nstatev),enerinternold(nblock),enerinelasold(nblock)
  dimension tempnew(nblock),stretchnew(nblock,6),defgradnew(nblock,9),fieldnew(nblock,nfieldv)
  dimension stressnew(nblock,6),statenew(nblock,nstatev),enerinternnew(nblock),enerinelasnew(nblock)
  do k=1,nblock
    tr=straininc(k,1)+straininc(k,2)+straininc(k,3)
    do j=1,6
      stressnew(k,j)=stressold(k,j)+2*props(2)*straininc(k,j)
      if (j.le.3) stressnew(k,j)=stressnew(k,j)+props(1)*tr
    end do
    do j=1,nstatev
      statenew(k,j)=stateold(k,j)+dt*j
    end do
    work=0
    do j=1,6
      fac=1
      if (j.gt.3) fac=2
      work=work+fac*(stressnew(k,j)+stressold(k,j))*straininc(k,j)/2
    end do
    enerinternnew(k)=enerinternold(k)+work/density(k)
    enerinelasnew(k)=enerinelasold(k)
  end do
end subroutine
"""


def request(n=4, ns=2, nf=0):
    stretch = np.tile([1., 1., 1., 0., 0., 0.], (n, 1))
    gradient = np.pad(stretch, ((0, 0), (0, 3)))
    strain = np.arange(n*6).reshape(n, 6)*1e-4
    return c.VumatBatchInput(step_time=0., total_time=0., time_increment=.1,
        density=np.ones(n)*2, coordinates=np.zeros((n, 3)), characteristic_length=np.ones(n),
        strain_increment=strain, relative_spin_increment=np.zeros((n, 3)),
        temperature_old=np.ones(n)*293., temperature_new=np.ones(n)*294.,
        stretch_old=stretch, stretch_new=stretch, deformation_gradient_old=gradient,
        deformation_gradient_new=gradient, field_old=np.zeros((n, nf)), field_new=np.ones((n, nf)),
        stress_old=np.zeros((n, 6)), state_old=np.zeros((n, ns)),
        internal_energy_old=np.zeros(n), inelastic_energy_old=np.zeros(n))


def build(tmp_path, source=SOURCE, ns=2, **options):
    if shutil.which("gfortran") is None:
        pytest.skip("GNU Fortran is required for native VUMAT acceptance")
    path = tmp_path / "material.f90"
    path.write_text(source)
    schema = c.MaterialStateSchema("test", (c.MaterialStateVariable("history", shape=(ns,)),))
    properties = options.pop("properties", [4., 3.])
    return c.compile_vumat(path, state_schema=schema, properties=properties,
                           cache_directory=tmp_path / "cache", trusted=True, **options)


def test_native_elastic_response_arbitrary_state_and_tensor_shear(tmp_path):
    material = build(tmp_path, ns=7)
    r = request(ns=7)
    out = material.update(r)
    expected = 6*r.strain_increment.copy()
    expected[:, :3] += 4*np.sum(r.strain_increment[:, :3], axis=1)[:, None]
    np.testing.assert_allclose(out.stress_new, expected, rtol=2e-15, atol=1e-16)
    np.testing.assert_allclose(out.state_new, np.tile(.1*np.arange(1,8), (4,1)))
    weights = [1,1,1,2,2,2]
    np.testing.assert_allclose(out.internal_energy_new, np.sum(expected*r.strain_increment*weights,axis=1)/4)
    assert not out.state_new.flags.writeable
    np.testing.assert_array_equal(r.state_old, 0)
    for i in range(4):
        selected = {k: getattr(r,k)[i:i+1] for k in r.__dataclass_fields__ if isinstance(getattr(r,k),np.ndarray)}
        point = material.update(replace(r, **selected))
        np.testing.assert_array_equal(point.stress_new, out.stress_new[i:i+1])


def test_all_context_channels_arrive_without_fabrication(tmp_path):
    source = SOURCE.replace("statenew(k,j)=stateold(k,j)+dt*j", """statenew(k,j)=stateold(k,j)+dt*j
      statenew(k,j)=statenew(k,j)+steptime+totaltime+coordmp(k,2)+charlength(k)
      statenew(k,j)=statenew(k,j)+tempnew(k)-tempold(k)+relspininc(k,2)
      statenew(k,j)=statenew(k,j)+stretchnew(k,2)+defgradnew(k,7)+fieldnew(k,1)""")
    material = build(tmp_path, source)
    r = replace(request(nf=1), step_time=.2,total_time=.3)
    out = material.update(r)
    np.testing.assert_allclose(out.state_new, np.tile([4.6,4.7],(4,1)))


def test_cache_reuse_and_tamper_rejection(tmp_path):
    first = build(tmp_path)
    second = build(tmp_path)
    assert first.summary() == second.summary()
    record = next((tmp_path / "cache").glob("*/build.json"))
    data = json.loads(record.read_text())
    data["library_sha256"] = "bad"
    record.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="AFM-VUMAT-CACHE"):
        build(tmp_path)


def test_missing_optional_energy_is_not_zero(tmp_path):
    source = SOURCE.replace("enerinternnew(k)=enerinternold(k)+work/density(k)", "! unavailable")
    out = build(tmp_path, source).update(request())
    assert out.internal_energy_new is None
    np.testing.assert_array_equal(out.inelastic_energy_new, 0)


def test_corrupt_cache_record_is_structured_failure(tmp_path):
    build(tmp_path)
    record = next((tmp_path / "cache").glob("*/build.json"))
    record.write_text("not json")
    with pytest.raises(ValueError, match="AFM-VUMAT-CACHE"):
        build(tmp_path)


def test_generated_source_collision_is_rejected(tmp_path):
    path = tmp_path / "agentfem-wrapper.f90"
    path.write_text(SOURCE)
    with pytest.raises(ValueError, match="AFM-VUMAT-SOURCE"):
        c.compile_vumat(path, state_schema=c.MaterialStateSchema("elastic"),
            properties=[4., 3.], cache_directory=tmp_path/"cache", trusted=True)


@pytest.mark.parametrize("utility", ["XIT", "XPLB_EXIT", "UNKNOWN_HELPER"])
def test_no_argument_vendor_or_external_calls_are_rejected(tmp_path, utility):
    source = SOURCE.replace("  do k=1,nblock", f"  call {utility}\n  do k=1,nblock")
    with pytest.raises(ValueError, match="AFM-VUMAT-SOURCE"):
        build(tmp_path, source)


def test_failed_output_does_not_mutate_old_state(tmp_path):
    source = SOURCE.replace("stressnew(k,j)=stressold(k,j)+2*props(2)*straininc(k,j)", "stressold(k,j)=999")
    material = build(tmp_path, source)
    r = request()
    with pytest.raises(ValueError, match="stress_new"):
        material.update(r)
    np.testing.assert_array_equal(r.stress_old, 0)


def test_unsupported_modern_signature_rejected_before_compilation(tmp_path):
    with pytest.raises(ValueError, match="AFM-VUMAT-ABI"):
        build(tmp_path, SOURCE.replace("lanneal", "jInfoArray").replace("dt,cmname", "dtArray,cmname"))


def test_trust_is_required_before_reading_or_building(tmp_path):
    with pytest.raises(ValueError, match="AFM-VUMAT-TRUST"):
        c.compile_vumat(tmp_path/"absent.for", state_schema=None, properties=[], cache_directory=tmp_path)


@pytest.mark.parametrize("change", [dict(density=np.zeros(4)),
    dict(strain_increment=np.zeros((4,3))), dict(time_increment=0),
    dict(field_new=np.zeros((4,2))), dict(step_time=2),
    dict(stress_old=np.full((4,6),np.nan)), dict(characteristic_length=-np.ones(4)),
    dict(strain_increment=np.ma.array(np.zeros((4,6))))])
def test_invalid_inputs_rejected(change):
    with pytest.raises(ValueError):
        replace(request(), **change)


def test_schema_mismatch(tmp_path):
    with pytest.raises(ValueError, match="schema"):
        build(tmp_path).update(request(ns=3))


def test_empty_immutable_transport_slice():
    from agentfem.constitutive.material_array_batch import _array
    original = _array(np.empty((4, 0)), (4, 0), "fields")
    for start in range(4):
        copied = _array(original[start:start+1], (1, 0), "fields")
        assert copied.shape == (1, 0)
        assert not copied.flags.writeable


def test_inspection_and_cli_admit_vumat_without_execution(tmp_path, capsys):
    from agentfem import cli
    path = tmp_path / "law.f90"
    path.write_text(SOURCE)
    assert cli.main(["inspect-user-material", str(path), "--kind", "VUMAT", "--json"]) == 0
    record = json.loads(capsys.readouterr().out)
    assert record["interface"] == "VUMAT"
    assert record["recommended_route"] == "restricted_vumat_batch_adapter"
    assert record["executable"] is False


def test_second_law_relaxation_history_and_stateless_restart(tmp_path):
    source = SOURCE.replace("stressold(k,j)+2*props(2)*straininc(k,j)",
                            "stressold(k,j)*exp(-dt/props(3))+2*props(2)*straininc(k,j)")
    material = build(tmp_path, source, properties=[4.,3.,.7], ns=3)
    r = replace(request(ns=3), strain_increment=np.zeros((4,6)), stress_old=np.ones((4,6)))
    for i in range(5):
        out = material.update(r)
        r = replace(r, step_time=(i+1)*.1, total_time=(i+1)*.1, stress_old=out.stress_new,
                    state_old=out.state_new, internal_energy_old=out.internal_energy_new,
                    inelastic_energy_old=out.inelastic_energy_new)
    np.testing.assert_allclose(r.stress_old, np.exp(-.5/.7), rtol=2e-15)
    restored = build(tmp_path, source, properties=[4.,3.,.7], ns=3)
    np.testing.assert_array_equal(restored.update(r).stress_new, material.update(r).stress_new)
    np.testing.assert_array_equal(restored.update(r).state_new, material.update(r).state_new)


def test_native_boundary_rechecks_mutable_array_metadata(tmp_path):
    material = build(tmp_path)
    r = request()
    r.strain_increment.shape = (2,12)
    with pytest.raises(ValueError, match="strain_increment"):
        material.update(r)
    props = material.properties
    props.shape = (1,2)
    assert material.properties.shape == (2,)


def test_fixed_format_source_and_empty_state(tmp_path):
    # Convert this simple fixture's free-format continuations to fixed form.
    lines = []
    continuation = False
    for line in SOURCE.splitlines():
        lines.append(("     &" if continuation else "      ") + line.strip().rstrip("&"))
        continuation = line.rstrip().endswith("&")
    source = tmp_path / "legacy.for"
    source.write_text("\n".join(lines))
    if shutil.which("gfortran") is None:
        pytest.skip("GNU Fortran required")
    material = c.compile_vumat(source, state_schema=c.MaterialStateSchema("elastic"),
        properties=[4.,3.], cache_directory=tmp_path/"cache", trusted=True)
    assert material.update(request(ns=0)).state_new.shape == (4,0)
