# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Optional, stateless arithmetic accelerator; pure Python remains supported."""
import os
from setuptools import Extension, setup

extensions = []
if os.environ.get('AGENTFEM_BUILD_NATIVE') == '1':
    extensions.append(Extension(
        'agentfem._hex_native',
        ['src/agentfem/_hex_native.c'],
        extra_compile_args=['/O2','/fp:strict'] if os.name == 'nt' else ['-O3','-ffp-contract=off'],
        libraries=[] if os.name == 'nt' else ['m'], optional=True,
    ))
setup(ext_modules=extensions)
