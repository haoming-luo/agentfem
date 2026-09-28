# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Reusable weak boundary models."""

from . import absorbing
from . import mechanical
from . import thermal
from .mechanical import (
    ElasticFoundation,
    RigidObstaclePenaltyContact,
    elastic_foundation,
    rigid_obstacle_contact,
)
from .thermal import ConvectionBoundary, convection

__all__ = [
    "ConvectionBoundary",
    "ElasticFoundation",
    "RigidObstaclePenaltyContact",
    "absorbing",
    "convection",
    "elastic_foundation",
    "mechanical",
    "rigid_obstacle_contact",
    "thermal",
]
