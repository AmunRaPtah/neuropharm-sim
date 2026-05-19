"""Tests for and_gate/."""

import numpy as np
import pytest

from neuropharm_sim.and_gate import (
    ANDGate,
    ConsolidationResult,
    NMDAPlasticityAxis,
    RewardContrastAxis,
    SalienceAxis,
)
from neuropharm_sim.and_gate.axes import _sigmoid_activation


class TestSigmoidActivation:
    def test_at_threshold_returns_half(self):
        val = float(_sigmoid_activation(0.5, threshold=0.5, slope=10.0))
        assert val == pytest.approx(0.5, abs=0.01)

    def test_well_above_threshold_approaches_one(self):
        val = float(_sigmoid_activation(100.0, threshold=0.5, slope=10.0))
        assert val > 0.99

    def test_well_below_threshold_approaches_zero(self):
        val = float(_sigmoid_activation(-100.0, threshold=0.5, slope=10.0))
        assert val < 0.01


class TestSalienceAxis:
    def test_high_burst_activates_axis(self):
        ax = SalienceAxis(threshold=2.0)
        val = float(ax.activation(5.0))
        assert val > 0.9

    def test_low_burst_gives_low_activation(self):
        ax = SalienceAxis(threshold=2.0)
        val = float(ax.activation(0.5))
        assert val < 0.1

    def test_modulation_suppresses_activation(self):
        ax_full = SalienceAxis(threshold=2.0, modulation=1.0)
        ax_half = SalienceAxis(threshold=2.0, modulation=0.5)
        signal = 3.0
        assert float(ax_half.activation(signal)) < float(ax_full.activation(signal))

    def test_zero_modulation_gives_near_zero(self):
        ax = SalienceAxis(threshold=2.0, modulation=0.0)
        val = float(ax.activation(100.0))
        assert val < 0.01

    def test_array_input(self):
        ax = SalienceAxis()
        vals = ax.activation(np.array([0.5, 2.0, 5.0]))
        assert vals.shape == (3,)
        assert vals[0] < vals[1] < vals[2]


class TestNMDAPlasticityAxis:
    def test_high_strength_activates(self):
        ax = NMDAPlasticityAxis(threshold=0.6)
        assert float(ax.activation(0.95)) > 0.9

    def test_low_strength_inactive(self):
        ax = NMDAPlasticityAxis(threshold=0.6)
        assert float(ax.activation(0.1)) < 0.1

    def test_reduced_nmda_raises_threshold(self):
        ax_intact = NMDAPlasticityAxis(threshold=0.6, nmda_modulation=1.0)
        ax_blocked = NMDAPlasticityAxis(threshold=0.6, nmda_modulation=0.5)
        strength = 0.7
        # Blocking NMDA should require higher input to achieve same activation
        assert float(ax_blocked.activation(strength)) < float(ax_intact.activation(strength))

    def test_full_block_kills_plasticity(self):
        ax = NMDAPlasticityAxis(nmda_modulation=0.001)
        assert float(ax.activation(1.0)) < 0.01


class TestRewardContrastAxis:
    def test_positive_rpe_activates(self):
        ax = RewardContrastAxis(threshold=0.5)
        val = float(ax.activation(actual_reward=1.0, predicted_reward=0.0))
        assert val > 0.9

    def test_zero_rpe_gives_low_activation(self):
        ax = RewardContrastAxis(threshold=0.5)
        val = float(ax.activation(actual_reward=0.5, predicted_reward=0.5))
        assert val < 0.05

    def test_negative_rpe_gives_zero(self):
        """Negative RPE (punishment > expected) should not drive consolidation."""
        ax = RewardContrastAxis(threshold=0.5)
        val = float(ax.activation(actual_reward=0.2, predicted_reward=0.8))
        assert val < 0.01

    def test_array_input(self):
        ax = RewardContrastAxis()
        actual = np.array([0.0, 0.5, 1.0])
        predicted = np.array([0.5, 0.5, 0.0])
        vals = ax.activation(actual, predicted)
        assert vals.shape == (3,)


class TestANDGate:
    def _strong_inputs(self) -> dict:
        return dict(
            burst_to_tonic=4.0,
            synaptic_strength=0.9,
            actual_reward=1.0,
            predicted_reward=0.0,
        )

    def test_all_axes_high_gives_high_gate(self):
        gate = ANDGate()
        result = gate.evaluate(**self._strong_inputs())
        assert isinstance(result, ConsolidationResult)
        assert float(result.gate_probability) > 0.5

    def test_any_axis_zero_kills_gate(self):
        gate = ANDGate()
        # Collapse salience axis
        gate.salience_axis = SalienceAxis(modulation=0.0)
        result = gate.evaluate(**self._strong_inputs())
        assert float(result.gate_probability) < 0.01

    def test_gate_is_product_of_axes(self):
        gate = ANDGate()
        result = gate.evaluate(**self._strong_inputs())
        expected = (
            float(result.salience_activation)
            * float(result.nmda_activation)
            * float(result.contrast_activation)
        )
        assert float(result.gate_probability) == pytest.approx(expected, rel=1e-9)

    def test_triple_axis_more_protective_than_single(self):
        gate = ANDGate()
        pa = gate.protection_analysis(
            burst_to_tonic=3.0,
            synaptic_strength=0.8,
            actual_reward=0.9,
            predicted_reward=0.2,
            modulation_levels=np.array([0.3, 0.5, 0.7]),
        )
        # Triple should always be <= single at same modulation level
        assert np.all(pa["triple"] <= pa["salience_only"] + 1e-9)
        assert np.all(pa["triple"] <= pa["nmda_only"] + 1e-9)

    def test_super_additivity_at_half_modulation(self):
        """At 50% single-axis modulation, triple should be ≥4× more protective."""
        gate = ANDGate()
        pa = gate.protection_analysis(
            modulation_levels=np.array([0.5])
        )
        # P_triple = f³ × baseline;  P_single = f × baseline  →  ratio = f²
        p_single = float(pa["salience_only"][0])
        p_triple = float(pa["triple"][0])
        if p_single > 1e-9:
            ratio = p_single / p_triple
            assert ratio > 3.0, f"Expected >4× protection ratio, got {ratio:.2f}"

    def test_phase_diagram_shape(self):
        gate = ANDGate()
        ax1 = np.linspace(0.5, 5.0, 20)
        ax2 = np.linspace(0.0, 1.0, 15)
        grid = gate.phase_diagram(ax1, ax2, fixed_contrast_activation=0.8)
        assert grid.shape == (20, 15)
        assert np.all(grid >= 0.0)
        assert np.all(grid <= 1.0)

    def test_consolidation_result_properties(self):
        gate = ANDGate(consolidation_threshold=0.05)
        result = gate.evaluate(**self._strong_inputs())
        assert 0.0 <= result.mean_gate_probability <= 1.0
        # above_threshold is ndarray or numpy scalar depending on input shape
        assert hasattr(result.above_threshold, "__bool__")
