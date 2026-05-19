"""
and_gate
--------
Three-axis consolidation threshold model for addiction prevention.

Axes
----
da_salience  : Dopamine salience signal (VTA burst / tonic ratio)
nmda_plasticity : NMDA-dependent synaptic plasticity probability
reward_contrast : Signed reward prediction error (RPE) magnitude

AND-gate
--------
Addiction consolidation requires ALL three axes to exceed their
respective thresholds simultaneously. Blocking any single axis
provides partial protection; blocking multiple axes provides
super-additive (non-linear) protection.
"""

from .axes import SalienceAxis, NMDAPlasticityAxis, RewardContrastAxis
from .gate import ANDGate, ConsolidationResult

__all__ = [
    "SalienceAxis",
    "NMDAPlasticityAxis",
    "RewardContrastAxis",
    "ANDGate",
    "ConsolidationResult",
]
