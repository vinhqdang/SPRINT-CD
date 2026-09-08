import numpy as np
import pytest

from sprint_cd.graph import (ARROW, CIRCLE, NONE, TAIL, MarkedGraph, fci_rules,
                             meek_rules)
from sprint_cd.simulate import dag_to_cpdag, structural_hamming_distance


def test_marks_record_the_endpoint_they_belong_to():
    g = MarkedGraph(d=2)
    g.set_edge(0, 1, TAIL, ARROW)          # 0 -> 1
    assert g.mark_at(0, 1) == ARROW        # arrowhead sits at 1
    assert g.mark_at(1, 0) == TAIL
    assert g.is_directed(0, 1) and not g.is_directed(1, 0)
    assert "0 --> 1" == g.edge_string(0, 1)


def test_chain_and_collider_give_the_textbook_cpdags():
    chain = np.zeros((3, 3)); chain[0, 1] = chain[1, 2] = 1
    cp = dag_to_cpdag(chain)
    assert cp.is_undirected(0, 1) and cp.is_undirected(1, 2)  # unorientable

    coll = np.zeros((3, 3)); coll[0, 1] = coll[2, 1] = 1
    cp = dag_to_cpdag(coll)
    assert cp.is_directed(0, 1) and cp.is_directed(2, 1)      # v-structure


def test_meek_r1_propagates_orientation():
    # 0 -> 1, 1 - 2, 0 and 2 non-adjacent  =>  1 -> 2
    g = MarkedGraph.complete_undirected(3)
    g.remove_edge(0, 2)
    g.set_edge(0, 1, TAIL, ARROW)
    out = meek_rules(g)
    assert out.is_directed(1, 2)


def test_meek_r2_orients_a_shortcut():
    # 0 -> 1 -> 2 with 0 - 2  =>  0 -> 2 (else a cycle)
    g = MarkedGraph.complete_undirected(3)
    g.set_edge(0, 1, TAIL, ARROW)
    g.set_edge(1, 2, TAIL, ARROW)
    out = meek_rules(g)
    assert out.is_directed(0, 2)


def test_cpdag_of_a_dag_has_the_same_skeleton():
    rng = np.random.default_rng(0)
    from sprint_cd.simulate import random_dag
    for _ in range(20):
        sem = random_dag(6, 0.3, rng)
        cp = sem.true_cpdag()
        np.testing.assert_array_equal(
            cp.skeleton(), (sem.adjacency + sem.adjacency.T > 0).astype(int)
        )


def test_fci_r1_reads_the_near_endpoint_not_the_far_one():
    """Regression test.

    In ``a *-> b o-* c`` the circle belongs to the ``b`` end of edge ``b-c``.
    Testing the ``c`` end instead lets R1 fire against an arrowhead already
    placed at ``b`` by R0, producing a reversed edge.  Here ``0 *-> 1`` and
    ``2 *-> 1`` are both set, so no circle sits at ``1`` and R1 must not fire.
    """
    g = MarkedGraph.complete_circles(3)
    g.remove_edge(0, 2)
    g.set_mark(0, 1, ARROW)
    g.set_mark(2, 1, ARROW)
    out = fci_rules(g, sepsets={(0, 2): ()})
    assert out.mark_at(1, 0) == CIRCLE      # 1's end of edge 1-0 untouched
    assert out.mark_at(0, 1) == ARROW
    assert out.mark_at(2, 1) == ARROW


def test_fci_recovers_the_bidirected_edge_of_a_latent_confounder():
    """0 -> 1 <- L -> 2 <- 3 with L latent gives 0 o-> 1 <-> 2 <-o 3."""
    from sprint_cd.dsep import oracle_skeleton_and_sepsets
    from sprint_cd.sprint_fci import pag_from_skeleton

    B = np.zeros((5, 5))
    B[0, 1] = B[4, 1] = B[4, 2] = B[3, 2] = 1.0
    adj = (np.abs(B) > 0).astype(int)
    skel, seps = oracle_skeleton_and_sepsets(adj, [0, 1, 2, 3], max_order=3)
    pag = pag_from_skeleton(skel, seps)

    assert pag.mark_at(0, 1) == ARROW and pag.mark_at(1, 0) == CIRCLE
    assert pag.mark_at(1, 2) == ARROW and pag.mark_at(2, 1) == ARROW   # bi-directed
    assert pag.mark_at(3, 2) == ARROW and pag.mark_at(2, 3) == CIRCLE
    assert not pag.adjacent(0, 2) and not pag.adjacent(1, 3)


def test_shd_counts_each_edge_slot_once():
    a = MarkedGraph.complete_undirected(3)
    b = a.copy()
    assert structural_hamming_distance(a, b) == 0
    b.remove_edge(0, 1)
    assert structural_hamming_distance(a, b) == 1
    b2 = a.copy()
    b2.set_edge(0, 1, TAIL, ARROW)
    assert structural_hamming_distance(a, b2) == 1
