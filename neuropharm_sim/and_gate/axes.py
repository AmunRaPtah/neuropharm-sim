"""
axes.py
-------
Three consolidation axes for the AND-Gate model.

Each axis converts a physiological input signal into a normalised
activation value in [0, 1] via a sigmoid threshold function.
Activation > threshold contributes to the AND-gate trigger.

References
----------
Berridge KC, Robinson TE (1998) What is the role of dopamine in reward:
    hedonic impact, reward learning, or incentive salience?
    Brain Res Rev 28:309-369.

Malenka RC, Bear MF (2004) LTP and LTD: an embarrassment of riches.
    Neuron 44:5-21.

Schultz W, Dayan P, Montague PR (1997) A neural substrate of prediction
    and reward. Science 275:1593-1599.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _sigmoid_activation(
    x: ArrayLike,
    threshold: float,
    slope: float = 10.0,
) -> NDArray[np.float64]:
    """Sigmoid centred at threshold with given slope.

    Parameters
    ----------
    x : array-like
        Input values.
    threshold : float
        Half-maximal activation point.
    slope : float, optional
        Steepness of transition. Default 10.

    Returns
    -------
    activation : ndarray
        Values in [0, 1].
    """
    x = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-slope * (x - threshold)))


@dataclass
class SalienceAxis:
    """DA salience axis: incentive salience encoded by phasic DA burst magnitude.

    The signal is the ratio of phasic VTA burst firing to tonic baseline,
    reflecting the degree to which a stimulus is tagged as motivationally
    relevant (Berridge & Robinson 1998).

    Parameters
    ----------
    threshold : float, optional
        Burst/tonic ratio at half-maximal activation. Default 2.0
        (2x above baseline = salient event).
    slope : float, optional
        Sigmoid steepness. Default 8.0.
    modulation : float, optional
        Multiplicative suppressor on the raw signal (0–1).
        1.0 = no modulation; 0.0 = complete suppression.
        Models pharmacological attenuation (e.g. D2 antagonist dampening phasic DA).
    """

    threshold: float = 2.0
    slope: float = 8.0
    modulation: float = 1.0

    def activation(self, burst_to_tonic_ratio: ArrayLike) -> NDArray[np.float64]:
        """Compute axis activation.

        Parameters
        ----------
        burst_to_tonic_ratio : array-like
            Phasic burst magnitude normalised to tonic baseline.
            1.0 = tonic only; >1 = phasic burst; <1 = dip.

        Returns
        -------
        activation : ndarray
            Salience axis activation in [0, 1].
        """
        signal = np.asarray(burst_to_tonic_ratio, dtype=float) * self.modulation
        return _sigmoid_activation(signal, self.threshold, self.slope)


@dataclass
class NMDAPlasticityAxis:
    """NMDA plasticity axis: probability of LTP induction.

    NMDA receptors act as coincidence detectors for pre- and post-synaptic
    activity. LTP probability is a sigmoid function of effective synaptic
    input strength, gated by Mg²⁺ relief (depolarisation-dependent).
    Modulators (e.g. NMDA antagonists like memantine) shift the threshold.

    Parameters
    ----------
    threshold : float, optional
        Synaptic input strength at half-maximal LTP probability. Default 0.6.
    slope : float, optional
        Sigmoid slope. Default 12.0.
    nmda_modulation : float, optional
        Fractional NMDA conductance available (0–1). 1.0 = intact;
        <1 = antagonist-mediated reduction. Shifts threshold right.
    """

    threshold: float = 0.6
    slope: float = 12.0
    nmda_modulation: float = 1.0

    def activation(self, synaptic_strength: ArrayLike) -> NDArray[np.float64]:
        """Compute NMDA plasticity axis activation.

        Parameters
        ----------
        synaptic_strength : array-like
            Normalised corticostriatal synaptic input strength (0–1).

        Returns
        -------
        activation : ndarray
            LTP probability in [0, 1].
        """
        # Reduced NMDA conductance raises the effective threshold
        effective_threshold = self.threshold / max(self.nmda_modulation, 1e-6)
        signal = np.asarray(synaptic_strength, dtype=float)
        return _sigmoid_activation(signal, effective_threshold, self.slope)


@dataclass
class RewardContrastAxis:
    """Reward contrast axis: signed reward prediction error (RPE) magnitude.

    Encodes the Schultz et al. (1997) temporal difference RPE:
        RPE = actual_reward - predicted_reward
    Positive RPE (unexpected reward) drives addiction consolidation.
    The axis activation is a sigmoid on the positive RPE magnitude,
    modulated by the precision of reward expectation.

    Parameters
    ----------
    threshold : float, optional
        RPE magnitude at half-maximal activation. Default 0.5
        (normalised reward units).
    slope : float, optional
        Sigmoid slope. Default 10.0.
    expectation_precision : float, optional
        How sharply reward is predicted (0–1). High precision means
        even a moderate RPE produces high contrast. Low precision
        (uncertain environment) blunts the axis.
    """

    threshold: float = 0.5
    slope: float = 10.0
    expectation_precision: float = 0.5

    def activation(
        self,
        actual_reward: ArrayLike,
        predicted_reward: ArrayLike,
    ) -> NDArray[np.float64]:
        """Compute reward contrast axis activation.

        Parameters
        ----------
        actual_reward : array-like
            Received reward magnitude (normalised 0–1).
        predicted_reward : array-like
            Predicted reward magnitude (0–1).

        Returns
        -------
        activation : ndarray
            Reward contrast axis activation in [0, 1].
        """
        r = np.asarray(actual_reward, dtype=float)
        p = np.asarray(predicted_reward, dtype=float)
        rpe = r - p
        # Only positive RPE drives consolidation; negative RPE is extinction
        positive_rpe = np.clip(rpe, 0.0, None)
        # Scale by expectation precision
        signal = positive_rpe * (1.0 + self.expectation_precision)
        return _sigmoid_activation(signal, self.threshold, self.slope)
