"""
drug_library.py
---------------
Published pharmacokinetic and pharmacodynamic parameters for key
dopaminergic drugs used in CNS research.

All Ki values are from radioligand competition binding assays.
PK parameters are population-mean values from clinical PK studies.

References
----------
Haloperidol
    Seeman P (1992) Dopamine receptor sequences. Neuropsychopharmacology 7:261-284.
    Kudo S, Ishizaki T (1999) Pharmacokinetics of haloperidol. Clin Pharmacokinet 37:435-456.

Cocaine
    Ritz MC et al. (1987) Cocaine receptors on dopamine transporters are related to
        self-administration of cocaine. Science 237:1219-1223.
    Volkow ND et al. (1997) Relationship between subjective effects of cocaine and
        dopamine transporter occupancy. Nature 386:827-830.

L-DOPA
    Nyholm D (2006) Pharmacokinetic optimisation in the treatment of Parkinson's disease.
        Clin Pharmacokinet 45:109-136.

Amphetamine
    Richelson E, Pfenning M (1984) Blockade by antidepressants and related compounds
        of biogenic amine uptake into rat brain synaptosomes. Eur J Pharmacol 104:277-286.
    Kuczenski R, Segal DS (1997) Effects of methylphenidate on extracellular dopamine,
        serotonin, and norepinephrine. J Neurochem 68:2032-2037.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class DrugProfile:
    """Complete PK/PD profile for a dopaminergic compound.

    Parameters
    ----------
    name : str
        Drug name.
    mechanism : str
        Primary pharmacological mechanism.
    ki_d1_nm : float or None
        Ki at D1 receptor in nM (None if not applicable).
    ki_d2_nm : float or None
        Ki at D2 receptor in nM (None if not applicable).
    dat_block_fraction : float
        Dopamine transporter blockade at therapeutic exposure (0–1).
    dat_release_fold : float
        Fold-increase in synaptic DA due to reverse transport / efflux.
        1.0 = no release; >1 = amphetamine-like efflux.
    half_life_h : float
        Elimination half-life in hours.
    bioavailability : float
        Oral bioavailability fraction.
    volume_dist_l : float
        Apparent volume of distribution in litres.
    molecular_weight : float
        Molecular weight in g/mol.
    brain_plasma_ratio : float
        Brain-to-free-plasma concentration ratio.
    free_fraction_plasma : float
        Unbound fraction in plasma.
    notes : str
        Source citations and key caveats.
    """

    name: str
    mechanism: str
    ki_d1_nm: Optional[float]
    ki_d2_nm: Optional[float]
    dat_block_fraction: float          # 0–1
    dat_release_fold: float            # ≥1
    half_life_h: float
    bioavailability: float
    volume_dist_l: float
    molecular_weight: float
    brain_plasma_ratio: float
    free_fraction_plasma: float
    notes: str = ""


# ── Haloperidol ───────────────────────────────────────────────────────────────
HALOPERIDOL = DrugProfile(
    name="Haloperidol",
    mechanism="D2/D3 antagonist (butyrophenone antipsychotic)",
    ki_d1_nm=210.0,     # Seeman 1992; weak D1 affinity
    ki_d2_nm=1.2,       # Seeman 1992; high-affinity D2 block
    dat_block_fraction=0.0,
    dat_release_fold=1.0,
    half_life_h=21.0,   # Kudo & Ishizaki 1999; range 14–36 h
    bioavailability=0.44,
    volume_dist_l=1800.0,  # high lipophilicity; Kudo 1999
    molecular_weight=375.86,
    brain_plasma_ratio=12.0,   # log BB ≈ 1.1; Kudo 1999
    free_fraction_plasma=0.08,  # 92% protein-bound
    notes=(
        "Ki_D2=1.2 nM from Seeman 1992. "
        "At 5 mg/day oral: ~75% D2 occupancy (Farde et al. 1992 Arch Gen Psychiatry 49:538). "
        "EPS threshold at >80% D2 occupancy (Farde 1992; Kapur 2000)."
    ),
)

# ── Cocaine ───────────────────────────────────────────────────────────────────
COCAINE = DrugProfile(
    name="Cocaine",
    mechanism="Non-selective monoamine reuptake inhibitor (DAT/NET/SERT block)",
    ki_d1_nm=None,
    ki_d2_nm=None,
    dat_block_fraction=0.78,   # Volkow et al. 1997 Nature: ~78% DAT occupancy at euphorigenic doses
    dat_release_fold=1.0,      # reuptake block only; no active efflux (unlike amphetamine)
    half_life_h=1.0,           # plasma; cocaine t½ ≈ 0.7–1.5 h
    bioavailability=0.92,      # intranasal / smoked; oral ≈ 0.20–0.60
    volume_dist_l=200.0,
    molecular_weight=303.36,
    brain_plasma_ratio=4.0,
    free_fraction_plasma=0.10,
    notes=(
        "DAT block 78% at peak recreational dose: Volkow ND et al. (1997) Nature 386:827. "
        "Ritz MC et al. (1987) Science 237:1219 established DAT as primary abuse target. "
        "Short t½ drives compulsive re-dosing."
    ),
)

# ── L-DOPA ────────────────────────────────────────────────────────────────────
LDOPA = DrugProfile(
    name="L-DOPA",
    mechanism="Dopamine precursor (decarboxylated to DA centrally and peripherally)",
    ki_d1_nm=None,
    ki_d2_nm=None,
    dat_block_fraction=0.0,
    dat_release_fold=3.5,      # Raises synaptic DA substantially in depleted Parkinson striatum
    half_life_h=1.5,           # plasma t½; Nyholm 2006
    bioavailability=0.98,      # excellent oral bioavailability when given with AADC inhibitor
    volume_dist_l=35.0,
    molecular_weight=197.19,
    brain_plasma_ratio=0.3,    # BBB transport via LAT-1
    free_fraction_plasma=0.80,
    notes=(
        "Nyholm D (2006) Clin Pharmacokinet 45:109. "
        "dat_release_fold represents net increase in synaptic DA relative to baseline "
        "in dopamine-depleted (Parkinson) striatum; effect smaller in intact system. "
        "No direct receptor affinity; acts via conversion to dopamine."
    ),
)

# ── Amphetamine ───────────────────────────────────────────────────────────────
AMPHETAMINE = DrugProfile(
    name="Amphetamine",
    mechanism="DAT substrate / vesicular DA efflux agent (VMAT2 depletion + reverse transport)",
    ki_d1_nm=None,
    ki_d2_nm=None,
    dat_block_fraction=0.50,   # competitive reuptake block + reverse transport
    dat_release_fold=5.0,      # Kuczenski & Segal 1997; 5-fold increase in extracellular DA
    half_life_h=10.0,          # d-amphetamine; range 9–14 h
    bioavailability=0.75,
    volume_dist_l=500.0,
    molecular_weight=135.21,
    brain_plasma_ratio=6.0,
    free_fraction_plasma=0.20,
    notes=(
        "Kuczenski R & Segal DS (1997) J Neurochem 68:2032. "
        "Richelson & Pfenning (1984) Eur J Pharmacol 104:277. "
        "Efflux mechanism (VMAT2→cytosol→DAT reverse transport) is the primary DA "
        "mechanism; dat_block_fraction captures the reuptake inhibition component only."
    ),
)

# ── Registry ─────────────────────────────────────────────────────────────────
DRUG_LIBRARY: dict[str, DrugProfile] = {
    "haloperidol": HALOPERIDOL,
    "cocaine": COCAINE,
    "ldopa": LDOPA,
    "l-dopa": LDOPA,
    "amphetamine": AMPHETAMINE,
}


def get_drug(name: str) -> DrugProfile:
    """Retrieve a drug profile by name (case-insensitive).

    Parameters
    ----------
    name : str
        Drug name. Recognised keys: haloperidol, cocaine, ldopa, amphetamine.

    Returns
    -------
    DrugProfile
        Immutable dataclass with PK/PD parameters.

    Raises
    ------
    KeyError
        If name is not found in the library.
    """
    key = name.lower().strip()
    if key not in DRUG_LIBRARY:
        available = ", ".join(DRUG_LIBRARY)
        raise KeyError(f"Drug '{name}' not found. Available: {available}")
    return DRUG_LIBRARY[key]


def list_drugs() -> list[str]:
    """Return canonical drug names available in the library."""
    seen: set[int] = set()
    unique: list[str] = []
    for profile in DRUG_LIBRARY.values():
        if id(profile) not in seen:
            seen.add(id(profile))
            unique.append(profile.name)
    return unique
