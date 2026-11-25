# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Component labels from declared field layout, never from field names."""
import json


def declared_components(kind, count):
    if kind.lower() == 'vector' and count in (2, 3):
        return tuple('XYZ'[:count])
    if kind.lower() == 'tensor' and count in (4, 9):
        axes = 'XY' if count == 4 else 'XYZ'
        return tuple(a+b for a in axes for b in axes)
    return ()  # Tensor6 / arbitrary state arrays require explicit ordering.


def attach_xdmf_components(grid, root):
    declarations = {}
    for attr in root.findall('.//Attribute'):
        association = attr.get('Center', 'Node')
        name = attr.get('Name')
        data = grid.cell_data if association == 'Cell' else grid.point_data
        if association not in ('Cell', 'Node') or name not in data:
            continue
        array = data.GetArray(name)
        count = array.GetNumberOfComponents()
        names = declared_components(attr.get('AttributeType', ''), count)
        for info in attr.findall('Information'):
            if info.get('Name') == 'component_names':
                try:
                    parsed = json.loads(info.get('Value', ''))
                    if isinstance(parsed, list) and len(parsed) == count and all(isinstance(s, str) and 0 < len(s) <= 80 for s in parsed):
                        names = tuple(parsed)
                except ValueError:
                    pass
        key = (association, name)
        if key in declarations and declarations[key] != names:
            declarations[key] = ()  # Conflicting time-series declarations: do not guess.
        else:
            declarations[key] = names
    for (association, name), names in declarations.items():
        array = (grid.cell_data if association == 'Cell' else grid.point_data).GetArray(name)
        for i, label in enumerate(names):
            array.SetComponentName(i, label)
