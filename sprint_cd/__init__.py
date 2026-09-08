"""Anytime-valid causal discovery.

The package contains two algorithms built on a shared e-process core, which
differ in how they decide that an edge is *absent*.

``CertCD`` -- certificate-only discovery (the main contribution)
    Obeys one rule: assert a feature only by REJECTING a null that is true
    whenever the feature is absent; never assert anything by failing to reject.
    Adjacency is certified by ``min_S E^{(S)}``, an e-process for the composite
    null "the pair is separable", which needs only the Markov condition;
    orientation is certified by a sequential universal-inference e-process
    against the reversed linear non-Gaussian factorisation.  The graph starts
    empty and grows, undecided pairs are reported as undecided, and no
    faithfulness assumption enters the validity argument.

``SprintCD`` / ``SprintFCI`` -- anytime-valid PC and FCI
    The conventional shape: start complete, delete edges against
    confidence-sequence certificates.  Retained as the baseline the
    certificate-only algorithm is measured against, since deleting on an
    accepted null is what forces the ``delta``-strong-faithfulness premise.

``EICP``
    Anytime-valid Invariant Causal Prediction across streaming environments.
"""

from .certcd import CertCD, CertCDConfig, run_cert_cd
from .certificates import DirectionEProcess, adjacency_log_e, admissible_sets
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
    "CertCD", "CertCDConfig", "run_cert_cd",
    "adjacency_log_e", "admissible_sets", "DirectionEProcess",
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
