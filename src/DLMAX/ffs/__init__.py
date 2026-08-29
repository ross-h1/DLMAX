"""
DLMAX.ffs — internal package for the AutoFFS API and supporting helpers.

The model set is expressed as :class:`~DLMAX.ffs.block.Block` objects combined
into a :class:`~DLMAX.ffs.universe.Universe`, with each block exposing its
forecasts as a :class:`~DLMAX.ffs.predictive.Predictive` on the observation
scale so that blocks of different kinds combine through one seam.

Import the public API from ``DLMAX`` directly. ``DLMAX.ffs_core`` also works
as a re-export layer. Direct imports from ``DLMAX.ffs.<submodule>`` are not
part of the supported public API and may change without notice.
"""

from DLMAX.ffs.block import Block
from DLMAX.ffs.universe import Universe
from DLMAX.ffs.predictive import (
    Predictive,
    GaussianPredictive,
    StudentTPredictive,
    LogNormalPredictive,
    combine,
)

__all__ = [
    "Block",
    "Universe",
    "Predictive",
    "GaussianPredictive",
    "StudentTPredictive",
    "LogNormalPredictive",
    "combine",
]
