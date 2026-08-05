"""Reusable weak boundary models."""

from . import absorbing
from . import thermal
from .thermal import ConvectionBoundary, convection

__all__ = ["ConvectionBoundary", "absorbing", "convection", "thermal"]
