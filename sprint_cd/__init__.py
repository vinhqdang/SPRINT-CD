"""SPRINT-CD: Sequential PC/FCI with aNytime-valid Tests.

Anytime-valid constraint-based causal discovery.  Conditional-independence
queries are answered by e-processes, edges are removed only against
confidence-sequence certificates, and both kinds of decision are charged
against a budget fixed over the whole potential hypothesis family -- so the
recovered graph carries time-uniform error control at *any* data-dependent
stopping time.

Main entry points
-----------------
``SprintCD`` / ``run_sprint_cd``
    Streaming, anytime-valid PC returning a CPDAG (assumes causal sufficiency).
``SprintFCI`` / ``run_sprint_fci``
    The same under latent confounding, returning a PAG.
``EICP``
    Anytime-valid Invariant Causal Prediction across streaming environments.
"""

from .e_icp import EICP, EICPConfig, icp_fixed_sample
from .eprocess import (
    DiscreteUniversalCI,
    EProcessState,
    GaussianUniversalCI,
    SafeLinearCI,
    coefficient_confidence_sequence,
    safe_linear_block_log_e,
    safe_linear_log_e,
)
from .graph import ARROW, CIRCLE, NONE, TAIL, MarkedGraph, fci_rules, meek_rules
from .multiplicity import HypothesisBudget, e_bh
from .simulate import (
    LinearGaussianSEM,
    dag_to_cpdag,
    random_dag,
    skeleton_errors,
    structural_hamming_distance,
)
from .sprint_cd import SprintCD, SprintCDConfig, run_sprint_cd
from .sprint_fci import SprintFCI, pag_from_skeleton, run_sprint_fci
from .stats import GaussianSuffStat

__version__ = "0.1.0"

__all__ = [
    "SprintCD", "SprintCDConfig", "run_sprint_cd",
    "SprintFCI", "run_sprint_fci", "pag_from_skeleton",
    "EICP", "EICPConfig", "icp_fixed_sample",
    "SafeLinearCI", "safe_linear_log_e", "safe_linear_block_log_e",
    "coefficient_confidence_sequence", "EProcessState",
    "GaussianUniversalCI", "DiscreteUniversalCI",
    "HypothesisBudget", "e_bh",
    "MarkedGraph", "meek_rules", "fci_rules", "NONE", "TAIL", "ARROW", "CIRCLE",
    "GaussianSuffStat",
    "LinearGaussianSEM", "random_dag", "dag_to_cpdag",
    "structural_hamming_distance", "skeleton_errors",
    "__version__",
]
