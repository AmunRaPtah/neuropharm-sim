"""
receptor_dynamics.py
--------------------
D1 and D2 dopamine receptor binding kinetics via the Hill equation.

Published receptor parameters
------------------------------
D1 receptor Kd  = 1 500 nM   (Sibley & Monsma 1992; Seeman 1980)
D2 receptor Kd  =    20 nM   (Seeman 1987; Farde et al. 1992)
Hill coefficient n is receptor-subtype specific (see defaults below).

Competitive antagonism is handled via the Cheng-Prusoff relation:
    Kd_eff = Kd * (1 + [antagonist] / Ki_antagonist)

References
----------
Seeman P (1987) Dopamine receptors and the dopamine hypothesis of
    schizophrenia. Synapse 1:133-152.
Farde L et al. (1992) Positron emission tomography analysis of central
    D1 and D2 dopamine receptor occupancy in patients treated with
    classical neuroleptics and clozapine. Arch Gen Psychiatry 49:538-544.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

# ── Published receptor constants ──────────────────────────────────────────────
KD_D1_NM: float = 1500.0  # nM equilibrium dissociation constant, D1
KD_D2_NM: float = 20.0    # nM equilibrium dissociation constant, D2
HILL_D1: float = 0.8       # Hill coefficient, D1 (slightly sub-unity, consistent with cooperative binding data)
HILL_D2: float = 0.8       # Hill coefficient, D2


def hill_occupancy(
    concentration: ArrayLike,
    kd: float,
    n: float = 1.0,
) -> NDArray[np.float64]:
    """Fractional receptor occupancy via the Hill equation.

    Parameters
    ----------
    concentration : array-like
        Free ligand concentration in nM. Scalar or array.
    kd : float
        Equilibrium dissociation constant in nM (Kd or effective Kd).
    n : float, optional
        Hill coefficient. Default 1.0 (Langmuir / non-cooperative).

    Returns
    -------
    occupancy : ndarray
        Fractional occupancy in [0, 1].

    Notes
    -----
    Hill equation: θ = [L]^n / (Kd^n + [L]^n)
    """
    c = np.asarray(concentration, dtype=float)
    c = np.clip(c, 0.0, None)
    cn = np.power(c, n)
    kdn = np.power(kd, n)
    return cn / (kdn + cn)


def d1_occupancy(
    dopamine_nm: ArrayLike,
    antagonist_nm: float = 0.0,
    ki_antagonist_nm: float | None = None,
) -> NDArray[np.float64]:
    """D1 receptor occupancy by dopamine under optional competitive antagonism.

    Parameters
    ----------
    dopamine_nm : array-like
        Free dopamine concentration in nM.
    antagonist_nm : float, optional
        Competing antagonist concentration in nM. Default 0.
    ki_antagonist_nm : float or None, optional
        Antagonist Ki at D1 in nM. Required when antagonist_nm > 0.

    Returns
    -------
    occupancy : ndarray
        Fractional D1 occupancy in [0, 1].
    """
    kd_eff = _effective_kd(KD_D1_NM, antagonist_nm, ki_antagonist_nm)
    return hill_occupancy(dopamine_nm, kd_eff, HILL_D1)


def d2_occupancy(
    dopamine_nm: ArrayLike,
    antagonist_nm: float = 0.0,
    ki_antagonist_nm: float | None = None,
) -> NDArray[np.float64]:
    """D2 receptor occupancy by dopamine under optional competitive antagonism.

    Parameters
    ----------
    dopamine_nm : array-like
        Free dopamine concentration in nM.
    antagonist_nm : float, optional
        Competing antagonist concentration in nM. Default 0.
    ki_antagonist_nm : float or None, optional
        Antagonist Ki at D2 in nM. Required when antagonist_nm > 0.

    Returns
    -------
    occupancy : ndarray
        Fractional D2 occupancy in [0, 1].
    """
    kd_eff = _effective_kd(KD_D2_NM, antagonist_nm, ki_antagonist_nm)
    return hill_occupancy(dopamine_nm, kd_eff, HILL_D2)


def antagonist_d2_occupancy(
    antagonist_nm: ArrayLike,
    ki_nm: float,
    n: float = 1.0,
) -> NDArray[np.float64]:
    """D2 occupancy by an antagonist alone (no competing agonist).

    Uses the Hill equation with Kd = Ki (valid when [agonist] → 0
    or when measuring radioligand displacement in PET studies).

    Parameters
    ----------
    antagonist_nm : array-like
        Free antagonist concentration in nM.
    ki_nm : float
        Antagonist Ki at D2 in nM.
    n : float, optional
        Hill coefficient. Default 1.0.

    Returns
    -------
    occupancy : ndarray
        Fractional D2 receptor occupancy in [0, 1].
    """
    return hill_occupancy(antagonist_nm, ki_nm, n)


def dose_to_brain_concentration(
    dose_mg: ArrayLike,
    bioavailability: float,
    volume_dist_l: float,
    molecular_weight: float,
    brain_plasma_ratio: float = 1.0,
    free_fraction: float = 1.0,
) -> NDArray[np.float64]:
    """Convert oral dose to estimated free brain concentration at steady state.

    Assumes single-compartment PK at steady state with once-daily dosing.
    Css_plasma = F * dose / (Vd * ke * tau), ke = ln2 / t_half.
    For simplicity, Css is approximated as the average steady-state level.

    Parameters
    ----------
    dose_mg : array-like
        Oral dose in mg.
    bioavailability : float
        Oral bioavailability fraction (0–1).
    volume_dist_l : float
        Apparent volume of distribution in litres.
    molecular_weight : float
        Drug molecular weight in g/mol (for mg→nM conversion).
    brain_plasma_ratio : float, optional
        Brain-to-plasma concentration ratio. Default 1.0.
    free_fraction : float, optional
        Unbound fraction in plasma. Default 1.0.

    Returns
    -------
    c_brain_nm : ndarray
        Estimated free brain concentration in nM.
    """
    dose_mg = np.asarray(dose_mg, dtype=float)
    # mg → nmol
    dose_nmol = dose_mg * 1e6 / molecular_weight
    # distribute into volume (litres → nM = nmol/L)
    c_plasma_nm = bioavailability * dose_nmol / volume_dist_l
    c_free_plasma_nm = c_plasma_nm * free_fraction
    return c_free_plasma_nm * brain_plasma_ratio


def _effective_kd(
    kd: float,
    antagonist_nm: float,
    ki_nm: float | None,
) -> float:
    """Cheng-Prusoff apparent Kd under competitive antagonism."""
    if antagonist_nm == 0.0 or ki_nm is None:
        return kd
    if ki_nm <= 0:
        raise ValueError("ki_nm must be positive")
    return kd * (1.0 + antagonist_nm / ki_nm)
