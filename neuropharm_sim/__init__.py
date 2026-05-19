"""
neuropharm-sim
==============
Circuit-level dopaminergic pharmacology simulator with AND-Gate
Consolidation Threshold Model for addiction prevention research.

Modules
-------
receptor_dynamics : D1/D2 Hill-equation binding kinetics
circuit_model     : ODE firing-rate model (VTA, NAc, PFC)
drug_library      : Published PK/PD parameters for key compounds
and_gate          : Three-axis consolidation threshold and AND-gate logic
simulation        : High-level runner and result containers
visualization     : Occupancy curves, circuit dynamics, phase diagrams
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("neuropharm-sim")
except PackageNotFoundError:
    __version__ = "0.1.0-dev"

__all__ = [
    "receptor_dynamics",
    "circuit_model",
    "drug_library",
    "and_gate",
    "simulation",
    "visualization",
]
