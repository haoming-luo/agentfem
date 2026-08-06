"""Reusable weak boundary models."""

from . import absorbing
from . import mechanical
from . import thermal
from .mechanical import ElasticFoundation, elastic_foundation
from .thermal import ConvectionBoundary, convection

__all__ = [
    "ConvectionBoundary", "ElasticFoundation", "absorbing", "convection",
    "elastic_foundation", "mechanical", "thermal",
]
