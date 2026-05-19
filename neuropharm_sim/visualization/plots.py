"""
plots.py
--------
Publication-quality visualisation for neuropharm-sim.

All functions return (fig, axes) tuples so callers can save or further
customise the output.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray

# ── Global style ──────────────────────────────────────────────────────────────
_STYLE: dict[str, Any] = {
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 1.2,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.frameon": False,
}

PALETTE = {
    "haloperidol": "#2166ac",
    "cocaine":     "#d6604d",
    "ldopa":       "#4dac26",
    "amphetamine": "#f4a582",
    "baseline":    "#aaaaaa",
    "triple":      "#1a9641",
    "dual":        "#fdae61",
    "single":      "#d7191c",
}


def _apply_style() -> None:
    matplotlib.rcParams.update(_STYLE)


# ── Occupancy curves ──────────────────────────────────────────────────────────

def plot_occupancy_curves(
    concentration_range_nm: NDArray | None = None,
    figsize: tuple[float, float] = (8, 4.5),
) -> tuple[Figure, Sequence[Axes]]:
    """D1 and D2 occupancy vs free ligand concentration.

    Plots the Hill-equation binding curves for both receptor subtypes,
    annotated with clinical reference concentrations.

    Parameters
    ----------
    concentration_range_nm : ndarray, optional
        Concentration axis in nM. Default log-space 0.1–10 000 nM.
    figsize : tuple, optional
        Figure size in inches.

    Returns
    -------
    fig, axes : Figure and (ax_d1, ax_d2) tuple.
    """
    from ..receptor_dynamics import (
        hill_occupancy, KD_D1_NM, KD_D2_NM, HILL_D1, HILL_D2
    )

    _apply_style()

    if concentration_range_nm is None:
        concentration_range_nm = np.logspace(-1, 4, 400)

    c = concentration_range_nm
    occ_d1 = hill_occupancy(c, KD_D1_NM, HILL_D1)
    occ_d2 = hill_occupancy(c, KD_D2_NM, HILL_D2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, sharey=True)

    for ax, occ, kd, label, color in [
        (ax1, occ_d1, KD_D1_NM, "D1 receptor", "#e08214"),
        (ax2, occ_d2, KD_D2_NM, "D2 receptor", "#2166ac"),
    ]:
        ax.semilogx(c, occ * 100, lw=2, color=color, label=label)
        ax.axvline(kd, ls="--", lw=1, color="gray", alpha=0.7)
        ax.axhline(50, ls=":", lw=1, color="gray", alpha=0.5)
        ax.text(kd * 1.15, 5, f"Kd = {kd:.0f} nM", color="gray", fontsize=9)
        ax.set_xlabel("Free ligand (nM)")
        ax.set_ylabel("Receptor occupancy (%)")
        ax.set_title(label)
        ax.set_xlim(c[0], c[-1])
        ax.set_ylim(0, 105)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%d%%"))

    # Annotate tonic DA level
    tonic_da = 100.0
    for ax in (ax1, ax2):
        ax.axvline(tonic_da, ls="-.", lw=1.2, color="#666", alpha=0.6)
        ax.text(tonic_da * 1.1, 88, "tonic DA\n~100 nM", color="#555",
                fontsize=8.5, ha="left", va="top")

    fig.suptitle("Dopamine Receptor Hill Binding Curves", fontweight="bold")
    fig.tight_layout()
    return fig, (ax1, ax2)


# ── Circuit dynamics ──────────────────────────────────────────────────────────

def plot_circuit_dynamics(
    trajectory: dict[str, NDArray],
    title: str = "Circuit Dynamics",
    figsize: tuple[float, float] = (10, 6),
) -> tuple[Figure, Sequence[Axes]]:
    """Plot time-series of all circuit node firing rates and DA.

    Parameters
    ----------
    trajectory : dict
        Output of MesocorticolimbicCircuit.simulate().
    title : str, optional
        Figure title.
    figsize : tuple, optional
        Figure size.

    Returns
    -------
    fig, axes
    """
    _apply_style()
    t = trajectory["t_ms"]

    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)

    # Top: firing rates
    ax0 = axes[0]
    ax0.plot(t, trajectory["r_vta"],   lw=1.8, color="#b35806", label="VTA")
    ax0.plot(t, trajectory["r_nacd1"], lw=1.8, color="#2166ac", label="NAcD1")
    ax0.plot(t, trajectory["r_nacd2"], lw=1.8, color="#d6604d", label="NAcD2")
    ax0.plot(t, trajectory["r_pfc"],   lw=1.8, color="#4dac26", label="PFC")
    ax0.set_ylabel("Firing rate (norm.)")
    ax0.set_ylim(0, 1)
    ax0.legend(ncol=4, fontsize=9, loc="upper right")
    ax0.set_title(title, fontweight="bold")

    # Middle: DA concentration
    ax1 = axes[1]
    ax1.plot(t, trajectory["da_nm"], lw=1.8, color="#762a83")
    ax1.set_ylabel("Extracellular DA (nM)")
    ax1.axhline(100, ls="--", lw=1, color="gray", alpha=0.6, label="baseline 100 nM")
    ax1.legend(fontsize=9)

    # Bottom: receptor occupancies
    ax2 = axes[2]
    ax2.plot(t, trajectory["d1_occ"] * 100, lw=1.8, color="#e08214", label="D1")
    ax2.plot(t, trajectory["d2_occ"] * 100, lw=1.8, color="#2166ac", label="D2")
    ax2.set_ylabel("Occupancy (%)")
    ax2.set_xlabel("Time (ms)")
    ax2.set_ylim(0, 105)
    ax2.legend(fontsize=9)
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%d%%"))

    fig.tight_layout()
    return fig, axes


# ── AND-gate protection ───────────────────────────────────────────────────────

def plot_and_gate_protection(
    protection_data: dict[str, NDArray],
    figsize: tuple[float, float] = (9, 5.5),
) -> tuple[Figure, Axes]:
    """Plot AND-gate consolidation probability vs modulation level.

    Demonstrates super-additive protection with multi-axis intervention.

    Parameters
    ----------
    protection_data : dict
        Output of ANDGate.protection_analysis().
    figsize : tuple, optional
        Figure size.

    Returns
    -------
    fig, ax
    """
    _apply_style()
    m = protection_data["modulation"]

    fig, ax = plt.subplots(figsize=figsize)

    # Baseline
    ax.hlines(
        protection_data["baseline"][0], m[0], m[-1],
        lw=1.5, ls="--", color=PALETTE["baseline"], label="No intervention (baseline)",
    )

    # Single-axis interventions
    for key, label, ls in [
        ("salience_only", "DA salience only",     "--"),
        ("nmda_only",     "NMDA plasticity only", "-."),
        ("contrast_only", "Reward contrast only", ":"),
    ]:
        ax.plot(m, protection_data[key], lw=1.5, ls=ls, color=PALETTE["single"], alpha=0.75, label=label)

    # Dual-axis
    for key, label in [
        ("dual_salience_nmda",    "DA + NMDA (dual)"),
        ("dual_salience_contrast","DA + Contrast (dual)"),
        ("dual_nmda_contrast",    "NMDA + Contrast (dual)"),
    ]:
        ax.plot(m, protection_data[key], lw=1.8, color=PALETTE["dual"], alpha=0.7, label=label)

    # Triple-axis (most prominent)
    ax.plot(m, protection_data["triple"], lw=2.5, color=PALETTE["triple"],
            label="All three axes (triple)", zorder=5)

    ax.set_xlabel("Axis modulation factor (1 = no block, 0 = full block)")
    ax.set_ylabel("AND-gate consolidation probability")
    ax.set_title("Super-Additive Protection: Multi-Axis vs Single-Axis Intervention",
                 fontweight="bold")
    ax.set_xlim(m[0], m[-1])
    ax.set_ylim(0, None)
    ax.legend(fontsize=8.5, loc="upper left")

    # Annotate super-additivity at 50% modulation
    m_half = 0.5
    idx = np.argmin(np.abs(m - m_half))
    p_single = protection_data["salience_only"][idx]
    p_triple = protection_data["triple"][idx]
    if p_single > 1e-6:
        ratio = p_single / p_triple if p_triple > 0 else np.inf
        ax.annotate(
            f"At 50% block:\nsingle = {p_single:.3f}\ntriple = {p_triple:.3f}\n({ratio:.1f}× more protective)",
            xy=(m_half, p_triple),
            xytext=(m_half + 0.12, p_single * 0.6),
            fontsize=8.5,
            arrowprops=dict(arrowstyle="->", color="gray", lw=0.8),
        )

    fig.tight_layout()
    return fig, ax


# ── Phase diagram ─────────────────────────────────────────────────────────────

def plot_phase_diagram(
    gate_grid: NDArray,
    axis1_values: NDArray,
    axis2_values: NDArray,
    consolidation_threshold: float = 0.1,
    xlabel: str = "DA salience (burst/tonic ratio)",
    ylabel: str = "Synaptic strength (NMDA axis)",
    title: str = "AND-Gate Phase Diagram",
    figsize: tuple[float, float] = (7, 5.5),
) -> tuple[Figure, Axes]:
    """2-D heatmap of gate consolidation probability.

    Parameters
    ----------
    gate_grid : ndarray of shape (n1, n2)
        AND-gate probability at each (axis1, axis2) point.
    axis1_values : ndarray
        Values for x-axis.
    axis2_values : ndarray
        Values for y-axis.
    consolidation_threshold : float, optional
        Threshold contour to draw.
    xlabel, ylabel, title : str, optional
        Labels.
    figsize : tuple, optional

    Returns
    -------
    fig, ax
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=figsize)

    im = ax.pcolormesh(
        axis1_values, axis2_values, gate_grid.T,
        cmap="YlOrRd", shading="auto", vmin=0, vmax=1,
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Consolidation probability", fontsize=11)

    # Threshold contour
    ax.contour(
        axis1_values, axis2_values, gate_grid.T,
        levels=[consolidation_threshold],
        colors=["#2166ac"], linewidths=[2.0], linestyles=["--"],
    )
    ax.text(
        axis1_values[-1] * 0.98, axis2_values[int(len(axis2_values) * 0.55)],
        f"threshold\n= {consolidation_threshold}",
        color="#2166ac", fontsize=9, ha="right",
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    fig.tight_layout()
    return fig, ax


# ── Benchmark validation ──────────────────────────────────────────────────────

def plot_benchmark_validation(
    doses_mg: NDArray,
    simulated_occ: NDArray,
    farde_doses: NDArray,
    farde_occ: NDArray,
    figsize: tuple[float, float] = (7, 5),
) -> tuple[Figure, Axes]:
    """Plot simulated D2 occupancy vs Farde et al. 1992 data.

    Parameters
    ----------
    doses_mg : ndarray
        Simulated dose range in mg/day.
    simulated_occ : ndarray
        Simulated D2 occupancy fraction (0–1).
    farde_doses : ndarray
        Haloperidol doses from Farde 1992 in mg/day.
    farde_occ : ndarray
        D2 occupancy fractions from Farde 1992 (0–1).
    figsize : tuple, optional

    Returns
    -------
    fig, ax
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(doses_mg, simulated_occ * 100, lw=2.2,
            color=PALETTE["haloperidol"], label="Model prediction")
    ax.scatter(farde_doses, farde_occ * 100, s=70, zorder=5,
               color="#d6604d", edgecolors="white", linewidths=1,
               label="Farde et al. 1992 (PET, [¹¹C]raclopride)")

    # Clinical threshold bands
    ax.axhspan(65, 80, alpha=0.10, color="#4dac26",
               label="Therapeutic window (65–80%)")
    ax.axhspan(80, 100, alpha=0.07, color="#d6604d",
               label="EPS risk (>80%)")
    ax.axhline(80, ls="--", lw=1, color="#d6604d", alpha=0.6)
    ax.axhline(65, ls="--", lw=1, color="#4dac26", alpha=0.6)

    ax.set_xlabel("Haloperidol dose (mg/day)")
    ax.set_ylabel("D2 receptor occupancy (%)")
    ax.set_title(
        "Benchmark: Haloperidol D2 Occupancy vs Farde et al. 1992",
        fontweight="bold",
    )
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%d%%"))
    ax.legend(fontsize=9.5, loc="lower right")
    fig.tight_layout()
    return fig, ax


# ── Drug dose-response comparison ─────────────────────────────────────────────

def plot_dose_response_comparison(
    results_by_drug: dict[str, list],
    metric: str = "d2_occupancy",
    ylabel: str = "D2 receptor occupancy (%)",
    scale: float = 100.0,
    figsize: tuple[float, float] = (9, 5),
) -> tuple[Figure, Axes]:
    """Compare a metric across drugs and doses.

    Parameters
    ----------
    results_by_drug : dict
        Maps drug name → list of SimulationResult (one per dose).
    metric : str, optional
        Attribute of SimulationResult to plot. Default 'd2_occupancy'.
    ylabel : str, optional
        Y-axis label.
    scale : float, optional
        Multiply the metric by this factor (e.g. 100 for fraction→%).
    figsize : tuple, optional

    Returns
    -------
    fig, ax
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=figsize)

    for drug_name, result_list in results_by_drug.items():
        doses = np.array([r.dose_mg for r in result_list])
        values = np.array([getattr(r, metric) for r in result_list]) * scale
        color = PALETTE.get(drug_name.lower(), None)
        ax.plot(doses, values, lw=2, marker="o", markersize=4,
                label=drug_name, color=color)

    ax.set_xlabel("Dose (mg)")
    ax.set_ylabel(ylabel)
    ax.set_title("Dose–Response Comparison", fontweight="bold")
    ax.legend(fontsize=10)
    fig.tight_layout()
    return fig, ax
