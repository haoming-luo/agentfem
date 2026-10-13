# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Optional, stateless arithmetic accelerator; pure Python remains supported."""
import os
import sys
from pathlib import Path
from setuptools import Extension, setup

extensions = []
if os.environ.get('AGENTFEM_BUILD_NATIVE', '1') != '0':
    extensions.append(Extension(
        'agentfem._hex_native',
        ['src/agentfem/_hex_native.c'],
        extra_compile_args=['/O2','/fp:strict'] if os.name == 'nt' else ['-O3','-ffp-contract=off'],
        libraries=[] if os.name == 'nt' else ['m'], optional=True,
    ))
    if os.name != 'nt':
        extensions.append(Extension('agentfem._body_pipeline',
            ['src/agentfem/_body_pipeline.c'], extra_compile_args=['-O3', '-ffp-contract=off'],
            libraries=['m', 'pthread'], optional=True))
    # GMP is optional. No library download or dependency installation occurs
    # during build. Missing GMP retains the exact Python geometry route.
    prefix = Path(os.environ.get('AGENTFEM_GMP_PREFIX', sys.prefix))
    if os.name != 'nt' and (prefix / 'include/gmp.h').is_file():
        extensions.append(Extension(
            'agentfem._q1_native',
            [str(path) for path in sorted(Path('src/agentfem/native').glob('*.c'))],
            include_dirs=[str(prefix / 'include')],
            library_dirs=[str(prefix / 'lib')], libraries=['gmp', 'm'],
            define_macros=[('Q1_BERNSTEIN_BOUND', '1'), ('Q1_ADAPTIVE_MARGIN', '1'),
                           ('Q1_INTERVAL_SUBDIVISION', '1'),
                           ('Q1_REFINE_DEPTH', '30'), ('Q1_REFINE_BOXES', '4096')],
            extra_compile_args=['-O3', '-ffp-contract=off'], optional=True,
        ))
setup(ext_modules=extensions)
