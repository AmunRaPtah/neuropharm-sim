"""
gate.py
-------
AND-Gate Consolidation Threshold Model.

The AND-gate integrates all three consolidation axes. A single session
contributes to addiction consolidation only when all three axes exceed
their thresholds simultaneously. This creates a non-linear protective
effect: modulating any single axis provides partial protection, but
multi-axis modulation yields super-additive (multiplicative) protection
because the gate probability is the product of independent axis activations.

Mathematical formulation
------------------------
Let a₁, a₂, a₃ ∈ [0,1] be the activations of the three axes.

    P_consolidation = a₁ × a₂ × a₃

Under single-axis intervention (axis i reduced by factor f):
    P_single = f × a₁ × a₂ × a₃  =  f × P_baseline

Under triple-axis intervention (each axis reduced by factor f):
    P_triple = f³ × P_baseline

Super-additivity ratio: P_triple / P_single = f²
For f = 0.5: P_single = 0.5 × P_baseline vs P_triple = 0.125 × P_baseline
    → triple-axis is 4× more protective than single-axis

References
----------
Hyman SE, Malenka RC, Nestler EJ (2006) Neural mechanisms of addiction:
    the role of reward-related learning and memory.
    Annu Rev Neurosci 29:565-598.

Kalivas PW, Volkow ND (2005) The neural basis of addiction: a pathology
    of motivation and choice. Am J Psychiatry 162:1403-1413.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .axes import NMDAPlasticityAxis, RewardContrastAxis, SalienceAxis


@dataclass
class ConsolidationResult:
    """Output of a single AND-gate evaluation.

    Parameters
    ----------
    salience_activation : ndarray
        DA salience axis activation (0–1).
    nmda_activation : ndarray
        NMDA plasticity axis activation (0–1).
    contrast_activation : ndarray
        Reward contrast axis activation (0–1).
    gate_probability : ndarray
        AND-gate output = product of all three activations (0–1).
    above_threshold : ndarray of bool
        True where gate_probability exceeds consolidation_threshold.
    consolidation_threshold : float
        Gate probability threshold used.
    """

    salience_activation: NDArray[np.float64]
    nmda_activation: NDArray[np.float64]
    contrast_activation: NDArray[np.float64]
    gate_probability: NDArray[np.float64]
    above_threshold: NDArray[np.bool_]
    consolidation_threshold: float

    @property
    def mean_gate_probability(self) -> float:
        """Mean AND-gate probability over the evaluated input space."""
        return float(np.mean(self.gate_probability))

    @property
    def consolidation_fraction(self) -> float:
        """Fraction of input points where all axes clear threshold."""
        return float(np.mean(self.above_threshold))


class ANDGate:
    """AND-Gate Consolidation Threshold Model.

    Evaluates addiction consolidation probability given the three
    physiological axis signals. Supports comparison of single-axis
    vs multi-axis intervention strategies.

    Parameters
    ----------
    salience_axis : SalienceAxis, optional
        DA salience axis. Defaults to published parameters.
    nmda_axis : NMDAPlasticityAxis, optional
        NMDA plasticity axis.
    contrast_axis : RewardContrastAxis, optional
        Reward contrast axis.
    consolidation_threshold : float, optional
        Minimum gate_probability to count as a consolidation event.
        Default 0.1 (10% of maximum).

    Examples
    --------
    >>> gate = ANDGate()
    >>> result = gate.evaluate(
    ...     burst_to_tonic=3.0,
    ...     synaptic_strength=0.8,
    ...     actual_reward=0.9,
    ...     predicted_reward=0.2,
    ... )
    >>> print(f"Consolidation probability: {result.gate_probability:.3f}")
    """

    def __init__(
        self,
        salience_axis: SalienceAxis | None = None,
        nmda_axis: NMDAPlasticityAxis | None = None,
        contrast_axis: RewardContrastAxis | None = None,
        consolidation_threshold: float = 0.1,
    ) -> None:
        self.salience_axis = salience_axis or SalienceAxis()
        self.nmda_axis = nmda_axis or NMDAPlasticityAxis()
        self.contrast_axis = contrast_axis or RewardContrastAxis()
        self.consolidation_threshold = consolidation_threshold

    def evaluate(
        self,
        burst_to_tonic: ArrayLike,
        synaptic_strength: ArrayLike,
        actual_reward: ArrayLike,
        predicted_reward: ArrayLike,
    ) -> ConsolidationResult:
        """Evaluate the AND-gate for given physiological inputs.

        Parameters
        ----------
        burst_to_tonic : array-like
            Phasic burst / tonic DA ratio for the salience axis.
        synaptic_strength : array-like
            Corticostriatal synaptic input strength for the NMDA axis.
        actual_reward : array-like
            Received reward magnitude (0–1).
        predicted_reward : array-like
            Predicted reward magnitude (0–1).

        Returns
        -------
        ConsolidationResult
        """
        a_sal = self.salience_axis.activation(burst_to_tonic)
        a_nmda = self.nmda_axis.activation(synaptic_strength)
        a_con = self.contrast_axis.activation(actual_reward, predicted_reward)

        gate = a_sal * a_nmda * a_con
        above = gate >= self.consolidation_threshold

        return ConsolidationResult(
            salience_activation=a_sal,
            nmda_activation=a_nmda,
            contrast_activation=a_con,
            gate_probability=gate,
            above_threshold=above,
            consolidation_threshold=self.consolidation_threshold,
        )

    def protection_analysis(
        self,
        burst_to_tonic: float = 3.0,
        synaptic_strength: float = 0.8,
        actual_reward: float = 0.9,
        predicted_reward: float = 0.2,
        modulation_levels: NDArray | None = None,
    ) -> dict[str, NDArray]:
        """Compare single-axis vs dual-axis vs triple-axis protection.

        Sweeps a modulation factor (0=complete block, 1=no intervention)
        across all three axes simultaneously (triple) or one at a time
        (single), and computes the gate probability at each level.

        Parameters
        ----------
        burst_to_tonic : float, optional
            Baseline phasic DA ratio. Default 3.0.
        synaptic_strength : float, optional
            Baseline synaptic strength. Default 0.8.
        actual_reward : float, optional
            Baseline received reward. Default 0.9.
        predicted_reward : float, optional
            Baseline predicted reward. Default 0.2.
        modulation_levels : ndarray, optional
            Modulation factors to sweep (0–1). Default linspace(0, 1, 50).

        Returns
        -------
        dict with keys
            'modulation'          : (n,) modulation levels
            'baseline'            : (n,) gate probability, no intervention (flat = constant)
            'salience_only'       : (n,) gate P when only salience axis is modulated
            'nmda_only'           : (n,) gate P when only NMDA axis is modulated
            'contrast_only'       : (n,) gate P when only contrast axis is modulated
            'dual_salience_nmda'  : (n,) both salience + NMDA modulated
            'dual_salience_contrast': (n,)
            'dual_nmda_contrast'  : (n,)
            'triple'              : (n,) all three axes modulated
        """
        if modulation_levels is None:
            modulation_levels = np.linspace(0.0, 1.0, 50)

        m = np.asarray(modulation_levels, dtype=float)
        results: dict[str, NDArray] = {"modulation": m}

        # Baseline (no intervention)
        baseline_gate = self._gate_scalar(
            burst_to_tonic, synaptic_strength, actual_reward, predicted_reward,
            sal_mod=1.0, nmda_mod=1.0, con_prec=self.contrast_axis.expectation_precision,
        )
        results["baseline"] = np.full_like(m, baseline_gate)

        # Sweep modulation
        results["salience_only"] = np.array([
            self._gate_scalar(burst_to_tonic, synaptic_strength, actual_reward, predicted_reward,
                              sal_mod=mi, nmda_mod=1.0, con_prec=self.contrast_axis.expectation_precision)
            for mi in m
        ])
        results["nmda_only"] = np.array([
            self._gate_scalar(burst_to_tonic, synaptic_strength, actual_reward, predicted_reward,
                              sal_mod=1.0, nmda_mod=mi, con_prec=self.contrast_axis.expectation_precision)
            for mi in m
        ])
        results["contrast_only"] = np.array([
            self._gate_scalar(burst_to_tonic, synaptic_strength, actual_reward, predicted_reward,
                              sal_mod=1.0, nmda_mod=1.0,
                              con_prec=self.contrast_axis.expectation_precision * mi)
            for mi in m
        ])
        results["dual_salience_nmda"] = np.array([
            self._gate_scalar(burst_to_tonic, synaptic_strength, actual_reward, predicted_reward,
                              sal_mod=mi, nmda_mod=mi, con_prec=self.contrast_axis.expectation_precision)
            for mi in m
        ])
        results["dual_salience_contrast"] = np.array([
            self._gate_scalar(burst_to_tonic, synaptic_strength, actual_reward, predicted_reward,
                              sal_mod=mi, nmda_mod=1.0,
                              con_prec=self.contrast_axis.expectation_precision * mi)
            for mi in m
        ])
        results["dual_nmda_contrast"] = np.array([
            self._gate_scalar(burst_to_tonic, synaptic_strength, actual_reward, predicted_reward,
                              sal_mod=1.0, nmda_mod=mi,
                              con_prec=self.contrast_axis.expectation_precision * mi)
            for mi in m
        ])
        results["triple"] = np.array([
            self._gate_scalar(burst_to_tonic, synaptic_strength, actual_reward, predicted_reward,
                              sal_mod=mi, nmda_mod=mi,
                              con_prec=self.contrast_axis.expectation_precision * mi)
            for mi in m
        ])

        return results

    def _gate_scalar(
        self,
        burst_to_tonic: float,
        synaptic_strength: float,
        actual_reward: float,
        predicted_reward: float,
        sal_mod: float,
        nmda_mod: float,
        con_prec: float,
    ) -> float:
        """Compute gate probability at a single modulation point."""
        sal_ax = SalienceAxis(
            threshold=self.salience_axis.threshold,
            slope=self.salience_axis.slope,
            modulation=sal_mod,
        )
        nmda_ax = NMDAPlasticityAxis(
            threshold=self.nmda_axis.threshold,
            slope=self.nmda_axis.slope,
            nmda_modulation=nmda_mod,
        )
        con_ax = RewardContrastAxis(
            threshold=self.contrast_axis.threshold,
            slope=self.contrast_axis.slope,
            expectation_precision=con_prec,
        )
        a_sal = float(sal_ax.activation(burst_to_tonic))
        a_nmda = float(nmda_ax.activation(synaptic_strength))
        a_con = float(con_ax.activation(actual_reward, predicted_reward))
        return a_sal * a_nmda * a_con

    def phase_diagram(
        self,
        axis1_values: NDArray,
        axis2_values: NDArray,
        fixed_contrast_activation: float = 0.8,
    ) -> NDArray:
        """2-D phase diagram of gate probability in (salience × NMDA) space.

        Parameters
        ----------
        axis1_values : ndarray
            Burst-to-tonic values for the salience axis.
        axis2_values : ndarray
            Synaptic strength values for the NMDA axis.
        fixed_contrast_activation : float, optional
            Fixed reward contrast activation. Default 0.8.

        Returns
        -------
        gate_grid : ndarray of shape (len(axis1_values), len(axis2_values))
            Gate probability at each (salience, NMDA) combination.
        """
        a_sal = self.salience_axis.activation(axis1_values)   # (n1,)
        a_nmda = self.nmda_axis.activation(axis2_values)       # (n2,)
        gate_grid = np.outer(a_sal, a_nmda) * fixed_contrast_activation
        return gate_grid
