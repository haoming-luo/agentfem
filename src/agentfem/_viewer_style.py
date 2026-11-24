# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Shared scene layout for interactive viewing and exported figures."""


def axis_marker():
    """Geometry-based XYZ labels survive both VTK and vtk.js rendering."""
    import numpy as np
    import pyvista as pv
    pieces = []
    for i, (label, color) in enumerate(zip('XYZ', ['#b74c60', '#458b76', '#4d74ae'])):
        direction = np.eye(3)[i]
        for part in (pv.Arrow(direction=direction),
                     pv.Text3D(label, depth=.015, height=.25).translate(direction*1.15)):
            part.cell_data['axis_rgb'] = np.tile(pv.Color(color).int_rgb, (part.n_cells, 1)).astype(np.uint8)
            pieces.append(part)
    mesh = pieces[0].merge(pieces[1:], merge_points=False)
    mapper = pv.DataSetMapper(dataset=mesh)
    mapper.set_scalars(mesh.cell_data['axis_rgb'], 'axis_rgb', preference='cell', rgb=True)
    actor = pv.Actor(mapper=mapper)
    actor.prop.lighting = False
    return actor


def legend_panel(plotter):
    from vtkmodules.vtkRenderingCore import vtkRenderer
    # Non-overlapping viewports: the model cannot cover the legend.
    plotter.renderer.SetViewport(0., .16, 1., 1.)
    panel = vtkRenderer()
    panel.SetViewport(0., 0., 1., .16)
    panel.SetInteractive(False)
    plotter.render_window.AddRenderer(panel)
    return panel


def scalar_bar(plotter, panel, label, mapper, foreground):
    bar = plotter.add_scalar_bar(
        label, mapper=mapper, vertical=False, position_x=.12, position_y=.18,
        width=.76, height=.72, title_font_size=24, label_font_size=20,
        n_labels=5, fmt="%.3g", font_family="arial", bold=False, shadow=False,
        color=foreground)
    bar.SetObjectName('agentfem-legend')
    plotter.renderer.RemoveViewProp(bar)
    panel.AddViewProp(bar)
    return bar


def install_browser_legend():
    """Preserve our legend layout through Trame's VTK-to-vtk.js adapter.

    Its default adapter forces automatic layout and serif fonts, independently
    of the VTK actor. Limit this override to our named legend actors.
    """
    from trame_vtk.modules.vtk.serializers.actors import scalar_bar_actor_serializer
    from trame_vtk.modules.vtk.serializers.registry import register_instance_serializer, SERIALIZERS
    if getattr(SERIALIZERS.get('vtkScalarBarActor'), '_agentfem', False):
        return

    def serialize(parent, actor, actor_id, context, depth):
        result = scalar_bar_actor_serializer(parent, actor, actor_id, context, depth)
        if result and actor.GetObjectName() == 'agentfem-legend':
            props = dict(result['properties'])
            # Browser uses an accessible HTML legend in the reserved panel.
            # Native exports retain this VTK actor. vtk.js rescales text inside
            # inset renderers, so forwarding font sizes does not preserve UX.
            props['visibility'] = False
            result['properties'] = props
        return result

    serialize._agentfem = True
    register_instance_serializer('vtkScalarBarActor', serialize)


def background(plotter, panel, preset):
    plotter.set_background(preset['background'], top=preset.get('top'))
    import pyvista as pv
    panel.SetBackground(pv.Color(preset['background']).float_rgb)
