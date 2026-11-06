"""Independent 3D invariant checks for planar thermoelastic output."""

import numpy as np
import pytest
from mpi4py import MPI

from agentfem import constitutive, eigenstrains, fields, mesh, results, studies


@pytest.mark.parametrize("assumption", ["plane_strain", "plane_stress"])
@pytest.mark.parametrize("rise", [0., 50.])
def test_planar_thermal_mises_matches_full_three_dimensional_invariant(assumption, rise):
    domain = mesh.rectangle((0., 0.), (1., 1.), (2, 2),
                            cell_type="quadrilateral", comm=MPI.COMM_SELF)
    u = fields.displacement(domain)
    u.value.interpolate(lambda x: np.vstack((.002 * x[0] + .0015 * x[1], -.0005 * x[1])))
    temperature = fields.temperature(domain, value=300. + rise)
    material = constitutive.thermoelastic(
        young=200000., poisson=.3, density=1., thermal_expansion=1.e-5,
        reference_temperature=300., conductivity=1., specific_heat=1.,
    )
    projected, = results.small_strain_partition_fields(
        u, ((material, None),),
        study=studies.static_solid(dimension=2, assumption=assumption),
        variables=("MISES",), eigenstrains=(eigenstrains.thermal(temperature),),
    )
    mu = 200000. / 2.6
    lame = 200000. * .3 / (1.3 * .4)
    strain = np.array([[.002, .00075, 0.], [.00075, -.0005, 0.], [0., 0., 0.]])
    mechanical = strain - 1.e-5 * rise * np.eye(3)
    if assumption == "plane_stress":
        mechanical[2, 2] = -lame * (mechanical[0, 0] + mechanical[1, 1]) / (lame + 2 * mu)
    stress = lame * np.trace(mechanical) * np.eye(3) + 2 * mu * mechanical
    principal = np.linalg.eigvalsh(stress)
    expected = np.sqrt(sum((principal[i] - principal[j])**2
                           for i, j in ((0, 1), (1, 2), (2, 0))) / 2.)
    np.testing.assert_allclose(projected.x.array, expected, rtol=1.e-12)


def test_hot_wall_uses_standard_invariant_and_versioned_golden(tmp_path, monkeypatch):
    import json
    import runpy
    from pathlib import Path

    monkeypatch.setenv("AGENTFEM_RELEASE_SMOKE", "1")
    example = Path(__file__).resolve().parents[1] / "examples/creep_hot_wall_assessment.py"
    runpy.run_path(str(example))["main"](output=tmp_path)
    record = json.loads((tmp_path / "result.json").read_text())
    assert record["metadata"]["governing_stress_sampling"].startswith("maximum DG0")
    # Golden verification is required by main(), not disabled by the test.
    import xml.etree.ElementTree as ET
    root = ET.parse(tmp_path / "thermoelastic_state.xdmf")
    names = {attribute.attrib["Name"] for attribute in root.findall(".//Attribute")}
    assert {"S", "MISES", results.field_variable("TEMP").name} <= names

