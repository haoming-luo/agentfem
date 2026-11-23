# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Record successful field writes without owning the solver's clock."""


class ObservedSeries:
    def __init__(self, writer):
        self.writer = writer
        self.times = []

    def __enter__(self):
        self.active = self.writer.__enter__()
        return self

    def write_fields(self, time, *fields):
        self.active.write_fields(time, *fields)
        self.times.append(float(time))

    def __exit__(self, *exception):
        return self.writer.__exit__(*exception)
