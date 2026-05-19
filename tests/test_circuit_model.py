"""Tests for circuit_model.py."""

import numpy as np
import pytest

from neuropharm_sim.circuit_model import (
    CircuitParams,
    CircuitState,
    MesocorticolimbicCircuit,
)


class TestMesocorticolimbicCircuit:
    def test_baseline_simulation_runs(self):
        circuit = MesocorticolimbicCircuit()
        result = circuit.simulate(t_span_ms=(0, 200), n_points=100)
        assert "r_vta" in result
        assert result["r_vta"].shape == (100,)

    def test_firing_rates_bounded(self):
        circuit = MesocorticolimbicCircuit()
        result = circuit.simulate(t_span_ms=(0, 500), n_points=200)
        for key in ("r_vta", "r_nacd1", "r_nacd2", "r_pfc"):
            assert np.all(result[key] >= 0.0), f"{key} went negative"
            assert np.all(result[key] <= 1.0), f"{key} exceeded 1.0"

    def test_da_concentration_positive(self):
        circuit = MesocorticolimbicCircuit()
        result = circuit.simulate(t_span_ms=(0, 300), n_points=100)
        assert np.all(result["da_nm"] >= 0.0)

    def test_d2_antagonist_reduces_d2_occupancy(self):
        """Haloperidol should reduce steady-state D2 occupancy vs baseline."""
        baseline = MesocorticolimbicCircuit()
        haldol = MesocorticolimbicCircuit(
            drug_d2_antagonist_nm=3.6, drug_ki_d2_nm=1.2
        )
        r_base = baseline.simulate(t_span_ms=(0, 1000), n_points=100)
        r_drug = haldol.simulate(t_span_ms=(0, 1000), n_points=100)
        # D2 occupancy by dopamine should be lower with antagonist present
        assert r_drug["d2_occ"][-1] < r_base["d2_occ"][-1]

    def test_dat_block_increases_da(self):
        """Cocaine-like DAT blockade should elevate DA."""
        baseline = MesocorticolimbicCircuit()
        cocaine_circ = MesocorticolimbicCircuit(drug_da_multiplier=1.5)
        r_base = baseline.simulate(t_span_ms=(0, 1000), n_points=100)
        r_drug = cocaine_circ.simulate(t_span_ms=(0, 1000), n_points=100)
        assert r_drug["da_nm"][-1] > r_base["da_nm"][-1]

    def test_steady_state_returns_circuit_state(self):
        circuit = MesocorticolimbicCircuit()
        ss = circuit.steady_state(t_settle_ms=500.0)
        assert isinstance(ss, CircuitState)
        assert 0.0 <= ss.r_vta <= 1.0
        assert 0.0 <= ss.r_nacd1 <= 1.0
        assert 0.0 <= ss.r_nacd2 <= 1.0
        assert 0.0 <= ss.r_pfc <= 1.0
        assert ss.da_nm > 0

    def test_occupancy_arrays_same_length_as_time(self):
        circuit = MesocorticolimbicCircuit()
        result = circuit.simulate(t_span_ms=(0, 100), n_points=50)
        n = len(result["t_ms"])
        assert len(result["d1_occ"]) == n
        assert len(result["d2_occ"]) == n
        assert len(result["da_nm"]) == n

    def test_custom_params(self):
        params = CircuitParams(tau_ms=10.0, baseline_da_nm=50.0)
        circuit = MesocorticolimbicCircuit(params=params)
        result = circuit.simulate(t_span_ms=(0, 200), n_points=50)
        assert result["r_vta"].shape == (50,)
