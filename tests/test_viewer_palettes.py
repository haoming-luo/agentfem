"""Display-color screening, not a perceptual-uniformity certification."""

import numpy as np
import pytest

from agentfem._viewer_export import DIVERGING_STOPS, color_map, palette_items, validate_view


# Machado et al., IEEE TVCG 2009, doi:10.1109/TVCG.2009.113.
# Severity-1 linear-RGB coefficients verified against Colour's reference tables:
# https://github.com/colour-science/colour/blob/develop/colour/blindness/datasets/machado2010.py
CVD = {
    "normal": np.eye(3),
    "protan": np.array([[.152286, 1.052583, -.204868], [.114503, .786281, .099216], [-.003882, -.048116, 1.051998]]),
    "deutan": np.array([[.367322, .860646, -.227968], [.280085, .672501, .047413], [-.011820, .042940, .968881]]),
    "tritan": np.array([[1.255528, -.076749, -.178779], [-.078411, .930809, .147602], [.004733, .691367, .303900]]),
}


def palette_screening(name, matrix):
    rgb = color_map(name)(np.linspace(0, 1, 257))[:, :3]
    linear = np.where(rgb <= .04045, rgb/12.92, ((rgb+.055)/1.055)**2.4)
    simulated = np.clip(linear @ matrix.T, 0, 1)
    luminance = simulated @ [.2126, .7152, .0722]
    # CIE Lab, sRGB D65. Endpoint separation is only a coarse screening metric;
    # no claim is made about every adjacent sample or every observer.
    xyz = simulated @ np.array([[.4124564, .3575761, .1804375],
                                 [.2126729, .7151522, .0721750],
                                 [.0193339, .1191920, .9503041]]).T
    xyz /= [.95047, 1., 1.08883]
    f = np.where(xyz > (6/29)**3, np.cbrt(xyz), xyz/(3*(6/29)**2)+4/29)
    lab = np.column_stack((116*f[:, 1]-16, 500*(f[:, 0]-f[:, 1]), 200*(f[:, 1]-f[:, 2])))
    return luminance, float(np.linalg.norm(lab[0]-lab[-1]))


@pytest.mark.parametrize("name", DIVERGING_STOPS)
@pytest.mark.parametrize("vision", CVD)
def test_diverging_palettes_have_ordered_halves_and_distinct_ends(name, vision):
    y, difference = palette_screening(name, CVD[vision])
    assert np.all(np.diff(y[:129]) > 0)
    assert np.all(np.diff(y[128:]) < 0)
    assert y[128] > .85
    assert difference > 10  # engineering screening threshold, not a standard
    if vision == "normal":
        assert abs(y[0]-y[-1]) < .03


def test_palette_menu_groups_without_removing_saved_choices():
    assert {x['value'] for x in palette_items('zero')} == {'coolwarm', 'azure-rose', 'lagoon-rose'}
    assert {'ocean', 'viridis', 'cividis', 'azure-rose', 'lagoon-rose', 'coolwarm'} == {x['value'] for x in palette_items('sequential')}
    from test_viewer_export import view_record
    for name in DIVERGING_STOPS:
        record = dict(view_record(), palette=name, color_mode='zero', range=[-2., 2.])
        assert validate_view(record) == record
