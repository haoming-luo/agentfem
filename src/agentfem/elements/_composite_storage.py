# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""P2+bubble nodal STORAGE for ten boundary nodes and an independent centre.

This polynomial supplies entity DOF ownership only. Composite quadrature,
interpolation, forces and results never use its polynomial shape functions.
"""

from functools import lru_cache
import numpy as np


@lru_cache(maxsize=1)
def scalar_element():
    import basix

    cell = basix.CellType.tetrahedron
    vertices = basix.geometry(cell)
    topology = basix.topology(cell)
    x = []
    M = []
    for dim, entities in enumerate(topology):
        x.append(
            [
                vertices[e].mean(axis=0, keepdims=True)
                if dim != 2
                else np.empty((0, 3))
                for e in entities
            ]
        )
        M.append(
            [
                np.ones((1, 1, 1, 1)) if dim != 2 else np.empty((0, 1, 0, 1))
                for e in entities
            ]
        )
    q, w = basix.make_quadrature(cell, 8)
    a, b, c = q.T
    generators = np.array(
        [
            np.ones(len(q)),
            a,
            b,
            c,
            a * a,
            a * b,
            a * c,
            b * b,
            b * c,
            c * c,
            (1 - a - b - c) * a * b * c,
        ]
    )
    poly = basix.polynomials.tabulate_polynomials(
        basix.PolynomialType.legendre, cell, 4, q
    )
    coefficients = (generators * w) @ poly.T
    return basix.create_custom_element(
        cell,
        [],
        coefficients,
        x,
        M,
        0,
        basix.MapType.identity,
        basix.SobolevSpace.H1,
        False,
        2,
        4,
        basix.PolysetType.standard,
    )


def vector_element():
    import basix.ufl

    return basix.ufl.blocked_element(
        basix.ufl.wrap_element(scalar_element()), shape=(3,)
    )
