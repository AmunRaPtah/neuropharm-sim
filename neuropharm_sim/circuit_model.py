"""
circuit_model.py
----------------
ODE firing-rate model of the mesocorticolimbic dopamine circuit.

Nodes
-----
VTA  : Ventral tegmental area (dopamine source)
NAcD1 : Nucleus accumbens D1-expressing MSNs (direct pathway, reward)
NAcD2 : Nucleus accumbens D2-expressing MSNs (indirect pathway, aversion)
PFC  : Prefrontal cortex (cognitive control, top-down)

Circuit topology (key projections)
-----------------------------------
VTA  →(DA)→ NAcD1, NAcD2, PFC
NAcD1 →(GABA, indirect)→ VTA  [negative feedback via striatonigral loop]
NAcD2 →(GABA, direct)→  VTA  [stronger negative feedback via indirect path]
PFC  →(Glu)→ VTA              [top-down excitatory drive]

Dopamine modulation
-------------------
D1 receptor activation (high Kd, requires high DA): excites NAcD1 MSNs.
D2 receptor activation (low Kd, active at baseline DA): inhibits NAcD2 MSNs
    and also inhibits VTA via autoreceptors.

Each firing rate follows:
    tau * dr/dt = -r + F(I_total)
where F is a rectified sigmoid transfer function and I_total integrates
all afferent drive.

References
----------
Humphries MD, Prescott TJ (2010) The ventral basal ganglia, a selection
    mechanism at the crossroads of space, strategy, and reward.
    Prog Neurobiol 90:385-417.
Frank MJ (2005) Dynamic dopamine modulation in the basal ganglia: a
    neurocomputational account of cognitive deficits in medicated and
    nonmedicated Parkinsonism. J Cogn Neurosci 17:51-72.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

from .receptor_dynamics import d1_occupancy, d2_occupancy


# ── Transfer function ─────────────────────────────────────────────────────────

def _sigmoid(x: float | NDArray, gain: float = 1.0) -> float | NDArray:
    """Rectified sigmoid: F(x) = 1 / (1 + exp(-gain*x)), clipped to [0,1]."""
    return 1.0 / (1.0 + np.exp(-gain * x))


# ── Circuit parameters ────────────────────────────────────────────────────────

@dataclass
class CircuitParams:
    """Biophysical and synaptic parameters for the mesocorticolimbic model.

    Parameters
    ----------
    tau_ms : float
        Membrane/population time constant in ms. Default 20 ms.
    baseline_da_nm : float
        Tonic extracellular dopamine in the NAc (nM). ~100 nM at rest
        (Schultz 2007; Floresco et al. 2003).
    gain : float
        Sigmoid transfer function slope. Default 2.0.
    w_vta_to_d1 : float
        VTA→NAcD1 synaptic weight (dopaminergic drive to D1 MSNs).
    w_vta_to_d2 : float
        VTA→NAcD2 synaptic weight.
    w_vta_to_pfc : float
        VTA→PFC mesocortical dopamine weight.
    w_d1_to_vta : float
        NAcD1→VTA feedback (striatonigral, indirect; negative = inhibitory).
    w_d2_to_vta : float
        NAcD2→VTA feedback (striatopallidal, direct; negative = inhibitory).
    w_pfc_to_vta : float
        PFC→VTA top-down glutamatergic excitation.
    i_ext_vta : float
        External (thalamo-brainstem) drive to VTA.
    i_ext_pfc : float
        External sensory/cognitive input to PFC.
    d1_modulation_scale : float
        Scaling of D1 occupancy effect on NAcD1 excitability.
    d2_modulation_scale : float
        Scaling of D2 occupancy effect on NAcD2 inhibition.
    autoreceptor_scale : float
        Scaling of D2 autoreceptor inhibition on VTA firing.
    """

    tau_ms: float = 20.0
    baseline_da_nm: float = 100.0      # ~100 nM tonic NAc DA (Schultz 2007)
    gain: float = 2.0

    # Synaptic weights
    w_vta_to_d1: float = 0.8
    w_vta_to_d2: float = 0.6
    w_vta_to_pfc: float = 0.4
    w_d1_to_vta: float = -0.2          # striatonigral feedback
    w_d2_to_vta: float = -0.4          # stronger indirect-path inhibition
    w_pfc_to_vta: float = 0.5

    # External inputs (tonic baseline drive)
    i_ext_vta: float = 0.5
    i_ext_pfc: float = 0.3

    # Dopamine modulation scaling
    d1_modulation_scale: float = 1.5
    d2_modulation_scale: float = 1.5
    autoreceptor_scale: float = 0.8


@dataclass
class CircuitState:
    """Snapshot of circuit firing rates.

    Parameters
    ----------
    t_ms : float
        Time in milliseconds.
    r_vta : float
        VTA firing rate (normalised, 0–1).
    r_nacd1 : float
        NAcD1 MSN population rate (0–1).
    r_nacd2 : float
        NAcD2 MSN population rate (0–1).
    r_pfc : float
        PFC firing rate (0–1).
    da_nm : float
        Effective extracellular DA concentration in nM.
    d1_occ : float
        D1 receptor occupancy (0–1).
    d2_occ : float
        D2 receptor occupancy (0–1).
    """

    t_ms: float
    r_vta: float
    r_nacd1: float
    r_nacd2: float
    r_pfc: float
    da_nm: float
    d1_occ: float
    d2_occ: float


class MesocorticolimbicCircuit:
    """ODE firing-rate model of the mesocorticolimbic dopamine circuit.

    Parameters
    ----------
    params : CircuitParams, optional
        Biophysical parameters. Defaults to published values.
    drug_da_multiplier : float, optional
        Multiplicative factor on tonic DA (e.g., >1 for DAT blockers/releasers,
        <1 for synthesis inhibitors). Default 1.0.
    drug_d2_antagonist_nm : float, optional
        Free concentration of a D2 antagonist (nM). Default 0.
    drug_ki_d2_nm : float or None, optional
        Ki of the antagonist at D2 (nM). Required when antagonist > 0.
    drug_d1_antagonist_nm : float, optional
        Free D1 antagonist concentration (nM). Default 0.
    drug_ki_d1_nm : float or None, optional
        Ki of the D1 antagonist (nM).

    Examples
    --------
    Simulate haloperidol-treated circuit for 500 ms:

    >>> params = CircuitParams()
    >>> circuit = MesocorticolimbicCircuit(
    ...     params=params,
    ...     drug_d2_antagonist_nm=3.6,
    ...     drug_ki_d2_nm=1.2,
    ... )
    >>> result = circuit.simulate(t_span_ms=(0, 500), n_points=1000)
    """

    def __init__(
        self,
        params: CircuitParams | None = None,
        drug_da_multiplier: float = 1.0,
        drug_d2_antagonist_nm: float = 0.0,
        drug_ki_d2_nm: float | None = None,
        drug_d1_antagonist_nm: float = 0.0,
        drug_ki_d1_nm: float | None = None,
    ) -> None:
        self.params = params or CircuitParams()
        self.drug_da_multiplier = drug_da_multiplier
        self.drug_d2_antagonist_nm = drug_d2_antagonist_nm
        self.drug_ki_d2_nm = drug_ki_d2_nm
        self.drug_d1_antagonist_nm = drug_d1_antagonist_nm
        self.drug_ki_d1_nm = drug_ki_d1_nm

    def _effective_da(self, r_vta: float) -> float:
        """Effective extracellular DA as function of VTA firing + drug."""
        p = self.params
        # DA scales with VTA firing rate around tonic baseline
        da = p.baseline_da_nm * (0.5 + r_vta) * self.drug_da_multiplier
        return max(da, 0.0)

    def _ode(self, t: float, y: NDArray) -> NDArray:
        """Right-hand side of the ODE system (units: ms⁻¹)."""
        r_vta, r_d1, r_d2, r_pfc = y
        p = self.params
        tau = p.tau_ms

        da = self._effective_da(r_vta)

        # Receptor occupancies
        occ_d1 = float(d1_occupancy(
            da,
            antagonist_nm=self.drug_d1_antagonist_nm,
            ki_antagonist_nm=self.drug_ki_d1_nm,
        ))
        occ_d2 = float(d2_occupancy(
            da,
            antagonist_nm=self.drug_d2_antagonist_nm,
            ki_antagonist_nm=self.drug_ki_d2_nm,
        ))

        # D2 autoreceptor feedback on VTA (D2 activation reduces firing)
        autoreceptor_inhibition = p.autoreceptor_scale * occ_d2

        # VTA total input
        i_vta = (
            p.i_ext_vta
            + p.w_d1_to_vta * r_d1
            + p.w_d2_to_vta * r_d2
            + p.w_pfc_to_vta * r_pfc
            - autoreceptor_inhibition
        )

        # NAcD1: excited by D1 occupancy (direct pathway)
        d1_excitation = p.d1_modulation_scale * occ_d1
        i_d1 = p.w_vta_to_d1 * r_vta + d1_excitation - 0.5

        # NAcD2: inhibited by D2 occupancy (indirect pathway)
        # When D2 antagonist blocks receptors, D2-MSNs lose DA-mediated inhibition
        d2_inhibition = p.d2_modulation_scale * occ_d2
        i_d2 = p.w_vta_to_d2 * r_vta - d2_inhibition + 0.5

        # PFC: top-down cognition, receives mesocortical DA
        mesocortical_da = p.w_vta_to_pfc * r_vta * (0.3 + 0.7 * occ_d1)
        i_pfc = p.i_ext_pfc + mesocortical_da - 0.3 * r_d2

        dr_vta = (-r_vta + _sigmoid(i_vta, p.gain)) / tau
        dr_d1 = (-r_d1 + _sigmoid(i_d1, p.gain)) / tau
        dr_d2 = (-r_d2 + _sigmoid(i_d2, p.gain)) / tau
        dr_pfc = (-r_pfc + _sigmoid(i_pfc, p.gain)) / tau

        return np.array([dr_vta, dr_d1, dr_d2, dr_pfc])

    def simulate(
        self,
        t_span_ms: tuple[float, float] = (0.0, 500.0),
        y0: NDArray | None = None,
        n_points: int = 500,
        rtol: float = 1e-6,
        atol: float = 1e-8,
    ) -> dict[str, NDArray]:
        """Integrate the circuit ODE and return trajectories.

        Parameters
        ----------
        t_span_ms : tuple of float
            (t_start, t_end) in milliseconds.
        y0 : ndarray of shape (4,), optional
            Initial firing rates [VTA, NAcD1, NAcD2, PFC]. Default 0.5 each.
        n_points : int, optional
            Number of output time points. Default 500.
        rtol, atol : float, optional
            ODE solver tolerances.

        Returns
        -------
        dict with keys
            't_ms'   : (n_points,) time array in ms
            'r_vta'  : (n_points,) VTA firing rate
            'r_nacd1': (n_points,) NAcD1 rate
            'r_nacd2': (n_points,) NAcD2 rate
            'r_pfc'  : (n_points,) PFC rate
            'da_nm'  : (n_points,) effective DA (nM)
            'd1_occ' : (n_points,) D1 occupancy
            'd2_occ' : (n_points,) D2 occupancy
        """
        if y0 is None:
            y0 = np.array([0.5, 0.5, 0.5, 0.5])

        t_eval = np.linspace(t_span_ms[0], t_span_ms[1], n_points)
        sol = solve_ivp(
            self._ode,
            t_span_ms,
            y0,
            t_eval=t_eval,
            method="RK45",
            rtol=rtol,
            atol=atol,
        )
        if not sol.success:
            raise RuntimeError(f"ODE solver failed: {sol.message}")

        r_vta, r_d1, r_d2, r_pfc = sol.y

        # Compute derived quantities along trajectory
        da_nm = np.array([self._effective_da(r) for r in r_vta])
        d1_occ = np.array([
            float(d1_occupancy(da, self.drug_d1_antagonist_nm, self.drug_ki_d1_nm))
            for da in da_nm
        ])
        d2_occ = np.array([
            float(d2_occupancy(da, self.drug_d2_antagonist_nm, self.drug_ki_d2_nm))
            for da in da_nm
        ])

        return {
            "t_ms": sol.t,
            "r_vta": r_vta,
            "r_nacd1": r_d1,
            "r_nacd2": r_d2,
            "r_pfc": r_pfc,
            "da_nm": da_nm,
            "d1_occ": d1_occ,
            "d2_occ": d2_occ,
        }

    def steady_state(
        self,
        y0: NDArray | None = None,
        t_settle_ms: float = 2000.0,
    ) -> CircuitState:
        """Run until approximate steady state and return the final circuit state.

        Parameters
        ----------
        y0 : ndarray of shape (4,), optional
            Initial conditions.
        t_settle_ms : float, optional
            Integration time (ms) to reach steady state. Default 2000 ms.

        Returns
        -------
        CircuitState
        """
        traj = self.simulate((0.0, t_settle_ms), y0=y0, n_points=200)
        r_vta = traj["r_vta"][-1]
        r_d1 = traj["r_nacd1"][-1]
        r_d2 = traj["r_nacd2"][-1]
        r_pfc = traj["r_pfc"][-1]
        da = self._effective_da(r_vta)
        return CircuitState(
            t_ms=t_settle_ms,
            r_vta=r_vta,
            r_nacd1=r_d1,
            r_nacd2=r_d2,
            r_pfc=r_pfc,
            da_nm=da,
            d1_occ=float(d1_occupancy(da, self.drug_d1_antagonist_nm, self.drug_ki_d1_nm)),
            d2_occ=float(d2_occupancy(da, self.drug_d2_antagonist_nm, self.drug_ki_d2_nm)),
        )
