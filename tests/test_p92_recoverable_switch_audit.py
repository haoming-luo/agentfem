from copy import deepcopy

import pytest

from tools.p92_measured_controls import measured_path
from tools.p92_recoverable_switch_audit import audit, common_support
from test_p92_measured_controls import data


@pytest.mark.parametrize("protocol", ["RF", "SLR", "SLC"])
def test_bracket_paths_keep_observation_identity(protocol):
    paths = [measured_path(data(protocol), cycles=1, switch_bound=side)
             for side in ("lower", "upper")]
    common_support(paths)
    bad = deepcopy(paths)
    bad[1]["segments"][0]["source_indices"][0] += 1
    with pytest.raises(ValueError, match="observation support"):
        common_support(bad)
    bad = deepcopy(paths)
    bad[1]["segments"][0]["target"] += .001
    with pytest.raises(ValueError, match="control classes"):
        common_support(bad)


def test_unverified_card_is_rejected_before_source_or_material_use():
    with pytest.raises(ValueError, match="numerically verified"):
        audit({"protocols": {}}, None)
    with pytest.raises(ValueError, match="numerically verified"):
        audit({"protocols": {p: {"numerical": {"passed": p != "SLC"}}
                             for p in ("RF", "SLR", "SLC")}}, None)
