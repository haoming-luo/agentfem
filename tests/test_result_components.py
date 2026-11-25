# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
import xml.etree.ElementTree as ET
import numpy as np
import pytest
from agentfem.results._components import attach_xdmf_components, declared_components


def test_declared_order_not_field_name():
    assert declared_components('Tensor', 9) == ('XX','XY','XZ','YX','YY','YZ','ZX','ZY','ZZ')
    assert declared_components('Tensor', 4) == ('XX','XY','YX','YY')
    assert declared_components('Tensor6', 6) == ()
    assert declared_components('Matrix', 9) == ()


def test_custom_state_and_tensor_labels_share_viewer_contract():
    pv = pytest.importorskip('pyvista')
    from agentfem.viewer import component_label
    grid = pv.ImageData(dimensions=(2,2,2)).cast_to_unstructured_grid()
    grid.cell_data['arbitrary_tensor'] = np.arange(9.).reshape(1,9)
    grid.cell_data['state'] = np.array([[.1,.2]])
    root = ET.fromstring('''<Xdmf><Domain><Grid>
      <Attribute Name="arbitrary_tensor" Center="Cell" AttributeType="Tensor"/>
      <Attribute Name="state" Center="Cell" AttributeType="Matrix">
        <Information Name="component_names" Value='["damage", "hardening"]'/>
      </Attribute></Grid></Domain></Xdmf>''')
    attach_xdmf_components(grid,root)
    assert component_label(grid, ('cell','arbitrary_tensor',3)) == 'YX'
    assert component_label(grid, ('cell','state',1)) == 'hardening'
    np.testing.assert_array_equal(grid.cell_data['arbitrary_tensor'], np.arange(9.).reshape(1,9))


def test_array_writer_declares_layout_for_reader():
    from agentfem.results.output import _attribute
    grid = ET.Element('Grid')
    field = _attribute(grid, 'strain', 'Cell', np.zeros((2,9)), 'data.h5:/strain')
    assert field.get('AttributeType') == 'Tensor'
    assert [child.tag for child in field] == ['DataItem']
    assert declared_components(field.get('AttributeType'), 9)[3] == 'YX'
    compact = _attribute(grid, 'custom', 'Cell', np.zeros((2,6)), 'data.h5:/custom')
    assert compact.find('Information') is None
