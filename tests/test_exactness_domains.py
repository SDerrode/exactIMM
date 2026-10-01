"""Block-level exactness-domain residuals (prg.utils.exactness) against the
gauges of the committed experiment suite and the exact mixture filter."""

from __future__ import annotations

from prg.experiments.cns_exactness import (
    ab_model,
    cgo_memory_model,
    cross_annihilation_model,
    g2_degenerate_model,
    g2_vector_model,
    mixed_branch_model,
    off_union_model,
    slaving_A_model,
)
from prg.utils.exactness import (
    assumption_g,
    cross_annihilation_residual,
    exactness_domains,
    gpb2_domain_residual,
    imm_domain_residual,
)


def test_cgo_model_is_imm_and_gpb2_exact():
    p = cgo_memory_model()
    assert imm_domain_residual(p) == 0.0
    assert gpb2_domain_residual(p) == 0.0
    d = exactness_domains(p)
    assert d["imm"] and d["gpb2"] and not d["constant_gain"]


def test_slaving_only_model_is_gpb2_exact_not_imm():
    p = slaving_A_model(0.4)
    assert imm_domain_residual(p) > 1e-3
    assert gpb2_domain_residual(p) < 1e-12
    d = exactness_domains(p)
    assert d["gpb2"] and not d["imm"] and not d["constant_gain"]


def test_ab_model_all_three_exact_but_imm():
    d = exactness_domains(ab_model(0.4))
    assert d["gpb2"] and d["constant_gain"] and not d["imm"]


def test_mixed_branch_leaves_gpb2_domain_unless_memoryless():
    assert gpb2_domain_residual(mixed_branch_model(0.0, 0.7)) < 1e-12
    r, (j, k) = cross_annihilation_residual(mixed_branch_model(0.8, 0.7))
    assert r > 1e-3
    assert (j, k) == (0, 1)  # memory of regime 0 seen by the channel of regime 1


def test_off_union_model_is_off_domain():
    assert gpb2_domain_residual(off_union_model(0.4, 0.2)) > 1e-3


def test_matrix_cross_annihilation_beyond_the_union():
    """q=2: singular nonzero memories with C_k N_j = 0 (E10) -- inside the GPB2
    domain although outside {C==0} U {A==MC}; rotated rows leave it."""
    p_in, p_out = cross_annihilation_model(True), cross_annihilation_model(False)
    assert gpb2_domain_residual(p_in) < 1e-12
    assert imm_domain_residual(p_in) > 1e-3
    assert gpb2_domain_residual(p_out) > 1e-2


def test_gpb2_domain_matches_exact_filter():
    """The block-level test predicts machine-precision agreement with the exact
    mixture filter (and disagreement off the domain)."""
    from prg.experiments.cns_exactness import gaps

    g_in = gaps(cross_annihilation_model(True), n_seeds=5)
    g_out = gaps(cross_annihilation_model(False), n_seeds=5)
    assert g_in["gpb2"][0].hi < 1e-12
    assert g_out["gpb2"][0].med > 1e-5


def test_assumption_g_holds_on_the_suite_models():
    for p in (
        cgo_memory_model(),
        slaving_A_model(0.4),
        ab_model(0.4),
        off_union_model(0.4, 0.2),
        mixed_branch_model(0.8, 0.7),
        cross_annihilation_model(True),
    ):
        g = assumption_g(p)
        assert g["g1"] and g["g2"] and g["g"], g
        assert exactness_domains(p)["assumption_g"]


def test_assumption_g2_fails_on_the_e8_family_and_is_restored_by_distinct_rows():
    # E8: regime-free observation row and i.i.d. regime -> (G2) fails
    g = assumption_g(g2_degenerate_model(0.5, distinct_rows=False))
    assert g["g1"] and not g["g2"] and not g["g"]
    assert g["g2_rows"] < 1e-12 and g["g2_channel"] < 1e-12
    # distinct transition rows alone restore (G2)
    g = assumption_g(g2_degenerate_model(0.5, distinct_rows=True))
    assert g["g2"] and g["g2_rows"] > 0.1


def test_assumption_g2_reads_the_state_informed_part_not_the_whole_row():
    # E8 vector case: the observation row is regime-dependent through an
    # autonomous component (D_22), yet its state-informed part is regime-free
    g = assumption_g(g2_vector_model(couple=False))
    assert not g["g2"], g
    # coupling y2 into y1 through a regime-dependent D_12 makes C^T SV^-1 D
    # regime-dependent and restores (G2)
    g = assumption_g(g2_vector_model(couple=True))
    assert g["g2"] and g["g2_channel"] > 1e-3, g
