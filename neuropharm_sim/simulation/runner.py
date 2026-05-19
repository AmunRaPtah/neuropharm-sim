"""
runner.py
---------
High-level SimulationRunner: combine drug PK, receptor dynamics,
circuit ODE, and AND-gate into a single callable API.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from ..drug_library import DrugProfile, get_drug
from ..receptor_dynamics import (
    antagonist_d2_occupancy,
    dose_to_brain_concentration,
)
from ..circuit_model import CircuitParams, MesocorticolimbicCircuit
from ..and_gate import ANDGate, ConsolidationResult


@dataclass
class SimulationResult:
    """Container for a complete simulation run.

    Parameters
    ----------
    drug_name : str
        Name of the simulated drug.
    dose_mg : float
        Administered dose in mg.
    brain_concentration_nm : float
        Estimated free brain concentration in nM.
    d2_occupancy : float
        D2 receptor occupancy fraction (0–1) at this dose.
    d1_occupancy : float
        D1 receptor occupancy fraction (0–1).
    circuit_trajectory : dict
        Time-series from MesocorticolimbicCircuit.simulate().
    steady_state : dict
        Final (steady-state) values of all circuit variables.
    consolidation : ConsolidationResult or None
        AND-gate output, if evaluated.
    notes : str
        Free-text notes (drug mechanism, clinical context).
    """

    drug_name: str
    dose_mg: float
    brain_concentration_nm: float
    d2_occupancy: float
    d1_occupancy: float
    circuit_trajectory: dict[str, NDArray]
    steady_state: dict[str, float]
    consolidation: Optional[ConsolidationResult] = None
    notes: str = ""

    def summary(self) -> str:
        """Return a human-readable summary string."""
        lines = [
            f"Drug         : {self.drug_name}",
            f"Dose         : {self.dose_mg:.2f} mg",
            f"Brain [drug] : {self.brain_concentration_nm:.3f} nM",
            f"D2 occupancy : {self.d2_occupancy * 100:.1f}%",
            f"D1 occupancy : {self.d1_occupancy * 100:.1f}%",
            "Steady-state firing rates:",
            f"  VTA   = {self.steady_state['r_vta']:.3f}",
            f"  NAcD1 = {self.steady_state['r_nacd1']:.3f}",
            f"  NAcD2 = {self.steady_state['r_nacd2']:.3f}",
            f"  PFC   = {self.steady_state['r_pfc']:.3f}",
        ]
        if self.consolidation is not None:
            lines.append(
                f"AND-gate P   : {self.consolidation.mean_gate_probability:.4f}"
            )
        if self.notes:
            lines.append(f"Notes        : {self.notes}")
        return "\n".join(lines)


class SimulationRunner:
    """High-level interface for neuropharmacological circuit simulations.

    Chains: drug PK → brain concentration → receptor occupancy →
    circuit ODE → (optional) AND-gate consolidation.

    Parameters
    ----------
    circuit_params : CircuitParams, optional
        ODE parameters for the mesocorticolimbic circuit.
    and_gate : ANDGate, optional
        AND-gate consolidation model.
    t_simulate_ms : float, optional
        ODE integration duration in ms. Default 1000 ms.
    n_points : int, optional
        Number of ODE output time points. Default 500.

    Examples
    --------
    >>> runner = SimulationRunner()
    >>> result = runner.run_drug("haloperidol", dose_mg=5.0)
    >>> print(result.summary())
    """

    def __init__(
        self,
        circuit_params: CircuitParams | None = None,
        and_gate: ANDGate | None = None,
        t_simulate_ms: float = 1000.0,
        n_points: int = 500,
    ) -> None:
        self.circuit_params = circuit_params or CircuitParams()
        self.and_gate = and_gate or ANDGate()
        self.t_simulate_ms = t_simulate_ms
        self.n_points = n_points

    def run_drug(
        self,
        drug: str | DrugProfile,
        dose_mg: float,
        evaluate_and_gate: bool = True,
        burst_to_tonic: float = 3.0,
        synaptic_strength: float = 0.8,
        actual_reward: float = 0.9,
        predicted_reward: float = 0.2,
    ) -> SimulationResult:
        """Run a full simulation for a drug at a given dose.

        Parameters
        ----------
        drug : str or DrugProfile
            Drug name (looked up in library) or DrugProfile instance.
        dose_mg : float
            Oral dose in mg.
        evaluate_and_gate : bool, optional
            Whether to run AND-gate evaluation. Default True.
        burst_to_tonic : float, optional
            Phasic DA burst ratio for AND-gate input. Default 3.0.
        synaptic_strength : float, optional
            Corticostriatal strength for AND-gate. Default 0.8.
        actual_reward : float, optional
            Actual reward magnitude. Default 0.9.
        predicted_reward : float, optional
            Predicted reward magnitude. Default 0.2.

        Returns
        -------
        SimulationResult
        """
        if isinstance(drug, str):
            profile = get_drug(drug)
        else:
            profile = drug

        # ── Step 1: PK → brain concentration ──────────────────────────────────
        c_brain_nm = float(dose_to_brain_concentration(
            dose_mg,
            bioavailability=profile.bioavailability,
            volume_dist_l=profile.volume_dist_l,
            molecular_weight=profile.molecular_weight,
            brain_plasma_ratio=profile.brain_plasma_ratio,
            free_fraction=profile.free_fraction_plasma,
        ))

        # ── Step 2: Receptor occupancies ──────────────────────────────────────
        d2_occ = float(antagonist_d2_occupancy(
            c_brain_nm,
            ki_nm=profile.ki_d2_nm if profile.ki_d2_nm is not None else 1e9,
        )) if profile.ki_d2_nm is not None else 0.0

        from ..receptor_dynamics import hill_occupancy, KD_D1_NM, HILL_D1, _effective_kd
        ki_d1 = profile.ki_d1_nm if profile.ki_d1_nm is not None else 1e9
        d1_occ = float(hill_occupancy(
            c_brain_nm,
            kd=_effective_kd(KD_D1_NM, c_brain_nm, ki_d1),
            n=HILL_D1,
        )) if profile.ki_d1_nm is not None else 0.0

        # ── Step 3: Circuit simulation ─────────────────────────────────────────
        circuit = MesocorticolimbicCircuit(
            params=self.circuit_params,
            drug_da_multiplier=profile.dat_release_fold * (1.0 - profile.dat_block_fraction) + profile.dat_block_fraction,
            drug_d2_antagonist_nm=c_brain_nm if profile.ki_d2_nm is not None else 0.0,
            drug_ki_d2_nm=profile.ki_d2_nm,
            drug_d1_antagonist_nm=c_brain_nm if profile.ki_d1_nm is not None else 0.0,
            drug_ki_d1_nm=profile.ki_d1_nm,
        )
        traj = circuit.simulate(
            t_span_ms=(0.0, self.t_simulate_ms),
            n_points=self.n_points,
        )
        ss = {
            "r_vta": float(traj["r_vta"][-1]),
            "r_nacd1": float(traj["r_nacd1"][-1]),
            "r_nacd2": float(traj["r_nacd2"][-1]),
            "r_pfc": float(traj["r_pfc"][-1]),
            "da_nm": float(traj["da_nm"][-1]),
        }

        # ── Step 4: AND-gate ───────────────────────────────────────────────────
        consolidation: Optional[ConsolidationResult] = None
        if evaluate_and_gate:
            # D2 antagonists reduce phasic DA salience
            effective_burst = burst_to_tonic * (1.0 - d2_occ * 0.5)
            # DA releasers amplify phasic response
            effective_burst *= profile.dat_release_fold
            consolidation = self.and_gate.evaluate(
                burst_to_tonic=effective_burst,
                synaptic_strength=synaptic_strength,
                actual_reward=actual_reward,
                predicted_reward=predicted_reward,
            )

        return SimulationResult(
            drug_name=profile.name,
            dose_mg=dose_mg,
            brain_concentration_nm=c_brain_nm,
            d2_occupancy=d2_occ,
            d1_occupancy=d1_occ,
            circuit_trajectory=traj,
            steady_state=ss,
            consolidation=consolidation,
            notes=profile.mechanism,
        )

    def dose_response(
        self,
        drug: str | DrugProfile,
        doses_mg: NDArray,
        evaluate_and_gate: bool = True,
    ) -> list[SimulationResult]:
        """Run simulations across a dose range.

        Parameters
        ----------
        drug : str or DrugProfile
            Drug to simulate.
        doses_mg : ndarray
            Array of doses in mg.
        evaluate_and_gate : bool, optional
            Whether to run AND-gate at each dose. Default True.

        Returns
        -------
        list of SimulationResult, one per dose.
        """
        return [
            self.run_drug(drug, float(d), evaluate_and_gate=evaluate_and_gate)
            for d in doses_mg
        ]

    def compare_drugs(
        self,
        drug_names: list[str],
        dose_mg: float = 5.0,
        evaluate_and_gate: bool = True,
    ) -> dict[str, SimulationResult]:
        """Compare multiple drugs at a common dose.

        Parameters
        ----------
        drug_names : list of str
            Drug names to compare.
        dose_mg : float, optional
            Dose in mg for each drug. Default 5.0.
        evaluate_and_gate : bool, optional
            Whether to run AND-gate. Default True.

        Returns
        -------
        dict mapping drug_name → SimulationResult.
        """
        return {
            name: self.run_drug(name, dose_mg, evaluate_and_gate=evaluate_and_gate)
            for name in drug_names
        }
