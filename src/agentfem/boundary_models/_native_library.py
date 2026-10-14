# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Load only the compiled module shipped in the installed AgentFEM package."""
import ctypes
import importlib


def load_library(*, required=True):
    try:
        module = importlib.import_module('agentfem._q1_native')
        return ctypes.CDLL(module.__file__)
    except (ImportError, OSError) as error:
        if required:
            raise RuntimeError('Packaged native Q1 geometry is unavailable; use the reference backend') from error
        return None
