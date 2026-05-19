"""Tests for receptor_dynamics.py."""

import numpy as np
import pytest

from neuropharm_sim.receptor_dynamics import (
    KD_D1_NM,
    KD_D2_NM,
    HILL_D1,
    HILL_D2,
    _effective_kd,
    antagonist_d2_occupancy,
    d1_occupancy,
    d2_occupancy,
    dose_to_brain_concentration,
    hill_occupancy,
)


class TestHillOccupancy:
    def test_at_kd_returns_half_for_n1(self):
        """At [L] = Kd with n=1, occupancy should be exactly 0.5."""
        occ = hill_occupancy(100.0, kd=100.0, n=1.0)
        assert float(occ) == pytest.approx(0.5, abs=1e-9)

    def test_zero_concentration_gives_zero(self):
        assert float(hill_occupancy(0.0, kd=50.0)) == pytest.approx(0.0)

    def test_very_high_concentration_approaches_one(self):
        occ = hill_occupancy(1e10, kd=1.0, n=1.0)
        assert float(occ) > 0.9999

    def test_negative_concentration_clipped_to_zero(self):
        occ = hill_occupancy(-5.0, kd=100.0, n=1.0)
        assert float(occ) == pytest.approx(0.0)

    def test_array_input(self):
        c = np.array([0.0, 100.0, 1e9])
        occ = hill_occupancy(c, kd=100.0, n=1.0)
        assert occ.shape == (3,)
        assert occ[0] == pytest.approx(0.0)
        assert occ[1] == pytest.approx(0.5, abs=1e-9)
        assert occ[2] > 0.9999

    def test_hill_coefficient_steepens_curve(self):
        """Higher n gives steeper transition."""
        c = np.linspace(0, 200, 100)
        occ_n1 = hill_occupancy(c, kd=100.0, n=1.0)
        occ_n2 = hill_occupancy(c, kd=100.0, n=2.0)
        # Below Kd, n=2 should be lower (steeper at Kd, flatter below)
        below_kd = c < 100
        assert np.all(occ_n2[below_kd] <= occ_n1[below_kd] + 1e-6)


class TestD1D2Occupancy:
    def test_d2_has_lower_kd_than_d1(self):
        """D2 should be more sensitive (higher occ) at same low DA."""
        da = 50.0  # below both Kds
        occ_d1 = float(d1_occupancy(da))
        occ_d2 = float(d2_occupancy(da))
        assert occ_d2 > occ_d1, "D2 (Kd=20 nM) should show higher occupancy than D1 (Kd=1500 nM) at 50 nM DA"

    def test_at_published_kd_d2(self):
        occ = float(d2_occupancy(KD_D2_NM))
        assert occ == pytest.approx(0.5, abs=0.05)

    def test_at_published_kd_d1(self):
        occ = float(d1_occupancy(KD_D1_NM))
        assert occ == pytest.approx(0.5, abs=0.05)

    def test_d2_competitive_antagonism_reduces_occupancy(self):
        """A D2 antagonist at Ki should shift D2 occupancy down at fixed DA."""
        da = 200.0
        occ_no_drug = float(d2_occupancy(da))
        occ_with_drug = float(d2_occupancy(da, antagonist_nm=10.0, ki_antagonist_nm=1.2))
        assert occ_with_drug < occ_no_drug

    def test_antagonist_occupancy_farde_benchmark(self):
        """At haloperidol Ki_D2=1.2 nM and ~3.6 nM brain conc, D2 occ ≈ 75%."""
        occ = float(antagonist_d2_occupancy(3.6, ki_nm=1.2))
        assert 0.70 < occ < 0.80, f"Expected ~75% D2 occupancy, got {occ*100:.1f}%"


class TestEffectiveKd:
    def test_no_antagonist_returns_kd(self):
        assert _effective_kd(100.0, 0.0, None) == 100.0

    def test_antagonist_at_ki_doubles_kd(self):
        """[antagonist] = Ki should double the apparent Kd."""
        kd_eff = _effective_kd(100.0, antagonist_nm=10.0, ki_nm=10.0)
        assert kd_eff == pytest.approx(200.0)

    def test_invalid_ki_raises(self):
        with pytest.raises(ValueError):
            _effective_kd(100.0, antagonist_nm=10.0, ki_nm=-1.0)


class TestDoseToBrainConcentration:
    def test_output_is_positive(self):
        c = dose_to_brain_concentration(
            5.0, bioavailability=0.44, volume_dist_l=1800.0,
            molecular_weight=375.86, brain_plasma_ratio=12.0, free_fraction=0.08,
        )
        assert float(np.squeeze(c)) > 0

    def test_proportional_to_dose(self):
        kwargs = dict(bioavailability=0.5, volume_dist_l=100.0,
                      molecular_weight=200.0, brain_plasma_ratio=1.0, free_fraction=1.0)
        c1 = float(dose_to_brain_concentration(1.0, **kwargs))
        c2 = float(dose_to_brain_concentration(2.0, **kwargs))
        assert c2 == pytest.approx(2 * c1, rel=1e-6)
