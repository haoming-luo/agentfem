import numpy as np
import pytest

from agentfem._viewer_motion import automatic_scale, displacement_metrics, interpolate_displacement, timeline_sample


def test_current_coordinates_do_not_double_deform():
    pv = pytest.importorskip('pyvista')
    from agentfem._viewer_motion import reference_frame, has_current_coordinates
    from agentfem.results import deform_visualization_frame
    a = pv.ImageData(dimensions=(3, 3, 3)).cast_to_unstructured_grid()
    b = a.copy(deep=True)
    reference = a.points.copy()
    a.point_data['U'] = reference * .1
    b.point_data['U'] = reference * .2
    a.points = reference + a.point_data['U']
    b.points = reference + b.point_data['U']
    assert has_current_coordinates(a, b, 'U')
    left, right = reference_frame(a, 'U', True), reference_frame(b, 'U', True)
    middle = interpolate_displacement(left, right, 'U', .5)
    visible = deform_visualization_frame(middle, 'U', reference_geometry=True)
    np.testing.assert_allclose(visible.points, reference*1.15)
    np.testing.assert_allclose(a.points, reference*1.1)
    b.points[0] += .01
    assert not has_current_coordinates(a, b, 'U')


def test_scale_bounds_and_large_motion():
    assert automatic_scale([(100., 10., 4.)]) == 1.
    assert automatic_scale([(100., 0., 0.)]) == 1.
    assert automatic_scale([(100., .01, .005)]) == 2000.
    assert automatic_scale([(100., 1., .001)]) == 20.  # isolated peak guard
    assert automatic_scale([(100., .01, .005), (100., .1, .05)]) == 200.
    with pytest.raises(ValueError):
        automatic_scale([(0., 0., 0.)])


def test_physical_timeline_uses_nonuniform_coordinates():
    assert timeline_sample([0., 1., 10.], .5) == (1, 4/9, 5.)
    assert timeline_sample([0., 1., 10.], 1.) == (1, 1., 10.)
    for times in ([0., 0.], [1., 0.], [0., float("nan")], []):
        with pytest.raises(ValueError):
            timeline_sample(times, .5)


def test_displacement_interpolation_preserves_sources_and_rejects_other_fields():
    pv = pytest.importorskip("pyvista")
    a = pv.ImageData(dimensions=(3, 3, 3)).cast_to_unstructured_grid()
    a.point_data["U"] = np.zeros((a.n_points, 3))
    a.cell_data["damage"] = np.zeros(a.n_cells)
    b = a.copy(deep=True)
    b.point_data["U"][:] = [1., 2., 3.]
    b.cell_data["damage"][:] = 1.
    c = interpolate_displacement(a, b, "U", .25, ("point", "U", "magnitude"))
    np.testing.assert_allclose(c.point_data["U"], np.tile([.25, .5, .75], (a.n_points, 1)))
    assert not np.any(a.point_data["U"])
    assert not np.any(c.cell_data["damage"])
    assert displacement_metrics(b, "U")[2] == 0.  # rigid translation
    with pytest.raises(ValueError, match="not damage"):
        interpolate_displacement(a, b, "U", .5, ("cell", "damage", None))
    b.points[0] += .1
    with pytest.raises(ValueError, match="identical reference"):
        interpolate_displacement(a, b, "U", .5)


def test_installed_controls_auto_scale_smooth_seek_and_field_change(tmp_path, monkeypatch):
    import asyncio
    pv = pytest.importorskip("pyvista")
    pytest.importorskip("trame")
    from agentfem.viewer import build_viewer
    plotters = []
    original = pv.Plotter
    def capture(*args, **kwargs):
        plotter = original(*args, **kwargs)
        plotters.append(plotter)
        return plotter
    monkeypatch.setattr(pv, "Plotter", capture)
    a = pv.ImageData(dimensions=(3, 3, 3)).cast_to_unstructured_grid()
    for i in range(2):
        a.point_data["U"] = a.points * (.001*i)
        a.cell_data["damage"] = np.zeros(a.n_cells)
        a.save(tmp_path / f"{i}.vtu")
    path = tmp_path / "series.pvd"
    path.write_text('<VTKFile type="Collection"><Collection><DataSet timestep="0" file="0.vtu"/>'
                    '<DataSet timestep="1" file="1.vtu"/></Collection></VTKFile>')
    server = build_viewer(path, reference_geometry=True, displacement="U")
    state, ctrl = server.state, server.controller
    state.ready()
    try:
        ctrl.on_server_ready()
        asyncio.run(ctrl.fit_deformation())
        assert state.scale > 1. and "2 saved frames" in state.scale_status
        assert not state.error
        async def exercise():
            with state:
                state.smooth = True
                state.frame = 0
                state.playback_duration = 1.
            ctrl.toggle_playback()
            previous_mtime = plotters[0].ren_win.GetMTime()
            await asyncio.sleep(.15)
            ctrl.stop_playback()
            assert plotters[0].ren_win.GetMTime() > previous_mtime
            assert 0 < state.blend < 1 and "Interpolated" in state.coordinate
            assert not state.error
            ctrl.seek_frame()
            assert state.blend == 0. and "Saved frame" in state.coordinate
            with state:
                state.field = 0  # cell damage
            assert not state.smooth_available and not state.smooth
            # Switching away during playback must stop the displacement task,
            # not leave it injecting blend weights into a damage field.
            with state:
                state.field = 1  # displacement magnitude
                state.frame = 0
                state.smooth = True
            ctrl.toggle_playback()
            await asyncio.sleep(.1)
            with state:
                state.field = 0
            await asyncio.sleep(.1)
            assert not state.playing and not state.smooth and state.blend == 0.
            assert not state.error
        asyncio.run(exercise())
    finally:
        ctrl.on_server_exited()
