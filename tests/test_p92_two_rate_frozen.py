from pathlib import Path

import pytest

from tools.p92_two_rate_frozen import evaluate


def test_failed_or_non_early_fit_cannot_enter_frozen_replay():
    with pytest.raises(ValueError, match="early-only"):
        evaluate(dict(training_cycles=[1, 2, 3, 4]), Path("unused"))
    with pytest.raises(ValueError, match="early-only"):
        evaluate(dict(training_cycles=[1, 2, 3], conditional_fit=dict(success=False)), Path("unused"))
