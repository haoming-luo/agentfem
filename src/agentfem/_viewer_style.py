# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Shared scene layout for interactive viewing and exported figures."""


def draw_axes(plotter, dark=False):
    """Screen-facing labels with camera-projected axes, also during orbit export."""
    import numpy as np
    import pyvista as pv
    from vtkmodules.vtkRenderingCore import vtkActor2D, vtkPolyDataMapper2D, vtkCoordinate
    camera = plotter.camera
    z = np.asarray(camera.position)-camera.focal_point
    z /= np.linalg.norm(z)
    x = np.cross(camera.up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    width, height = plotter.window_size
    scale = height/1080
    center = np.array([85., 85.])*scale
    length = 32*scale
    colors = ['#eda6b9', '#9ed4c4', '#a8bee9'] if dark else ['#983f60', '#267762', '#3d5c9b']
    points, lines, rgb = [], [], []
    for i in np.argsort(z):
        delta = np.array([x[i], y[i]])*length
        norm = np.linalg.norm(delta)
        direction = delta/norm if norm > 1 else np.array([0., 1.])
        end = center+delta
        normal = np.array([-direction[1], direction[0]])
        segments = [(center, end)]
        if norm > 5:
            segments += [(end, end-direction*6*scale+normal*2.5*scale), (end, end-direction*6*scale-normal*2.5*scale)]
        for start, finish in segments:
            lines.extend([2, len(points), len(points)+1])
            points.extend([[start[0]/width, start[1]/height, 0], [finish[0]/width, finish[1]/height, 0]])
            rgb.append(pv.Color(colors[i]).int_rgb)
        pos = end+direction*12*scale
        plotter.add_text('XYZ'[i], position=tuple(pos.astype(int)), font_size=max(8, round(13*scale)),
                         color=colors[i], font='arial', name=f'axis-label-{i}')
    mesh = pv.PolyData(np.array(points), lines=np.array(lines))
    mesh.cell_data['rgb'] = np.asarray(rgb, dtype=np.uint8)
    mapper = vtkPolyDataMapper2D()
    mapper.SetInputData(mesh)
    mapper.SetScalarModeToUseCellData()
    mapper.SetColorModeToDirectScalars()
    coordinate = vtkCoordinate()
    coordinate.SetCoordinateSystemToNormalizedViewport()
    mapper.SetTransformCoordinate(coordinate)
    actor = vtkActor2D()
    actor.SetMapper(mapper)
    actor.GetProperty().SetLineWidth(2.5*scale)
    plotter.add_actor(actor, name='orientation-axes', reset_camera=False)


def legend_panel(plotter):
    from vtkmodules.vtkRenderingCore import vtkRenderer
    # Transparent annotation layer over the same full model viewport.
    plotter.renderer.SetViewport(0., 0., 1., 1.)
    panel = vtkRenderer()
    panel.SetViewport(0., 0., 1., 1.)
    plotter.render_window.SetNumberOfLayers(2)
    panel.SetLayer(1)
    panel.SetPreserveColorBuffer(True)
    panel.SetInteractive(False)
    plotter.render_window.AddRenderer(panel)
    return panel


def scalar_bar(plotter, panel, label, mapper, foreground):
    from agentfem._viewer_camera import text_size
    bar = plotter.add_scalar_bar(
        label, mapper=mapper, vertical=False, position_x=.15, position_y=.025,
        width=.80, height=.07, title_font_size=text_size(24, plotter.window_size), label_font_size=text_size(20, plotter.window_size),
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
