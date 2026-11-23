# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Output metadata records completed writes, never attempted solver steps."""
import pytest

from agentfem.results._series_observation import ObservedSeries


def test_observed_series_records_only_successful_writes():
    class Writer:
        def __enter__(self):
            return self

        def write_fields(self, time, *fields):
            if time == 2.0:
                raise OSError("write failed")

        def __exit__(self, *exception):
            self.exception = exception
            return False

    writer = Writer()
    series = ObservedSeries(writer)
    with pytest.raises(OSError, match="write failed"):
        with series as output:
            output.write_fields(0.0, object())
            output.write_fields(1.0, object())
            output.write_fields(2.0, object())
    assert series.times == [0.0, 1.0]
    assert writer.exception[0] is OSError
