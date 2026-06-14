"""Scoring unit tests — the contract the whole benchmark stands on."""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from bench.scoring import (score_run, driver_score, mechanism_correct,
                           grounding_pass, sum_share_pass, canonical_key,
                           stability)


def _label(action, drivers=None, reason=None):
    return {"expected": {"action": action, "drivers": drivers or [],
                         "abstain_reason": reason}}


def _drv(dim, seg, factor, mech, share=1.0):
    return {"dimension": dim, "segment": seg, "factor": factor,
            "mechanism": mech, "share_of_move": share}


LBL_T1 = _label("explain", [_drv("country", "Meridia", "orders", "volume")])
LBL_T3 = _label("explain", [_drv("order_type", "promo", "aov", "mix")])
LBL_T4 = _label("abstain", reason="unmapped_dimension")
LBL_T5 = _label("explain", [_drv("country", "Norvik", "orders", "volume", 0.7),
                            _drv("category", "fashion", "orders", "volume", 0.3)])
LBL_T6 = _label("no_driver")
FACTS = [109440.0, 87552.0, -21888.0, 21888.0, -20.0, 20.0, -0.2, 0.2, 0.7, 0.3]


def test_exact_match_scores_full():
    ans = {"action": "explain",
           "drivers": [_drv("country", "Meridia", "orders", "volume")],
           "cited_numbers": [{"label": "delta_pct", "value": -20.0}],
           "abstain_reason": None}
    s = score_run(LBL_T1, ans, FACTS)
    assert abs(s["score"] - 1.0) < 1e-9 and s["action_correct"] and s["driver_score"] == 1.0
    assert s["mechanism_correct"] is True


def test_wrong_segment_zeroes_driver_but_not_action():
    ans = {"action": "explain",
           "drivers": [_drv("country", "Norvik", "orders", "volume")],
           "cited_numbers": []}
    s = score_run(LBL_T1, ans, FACTS)
    assert s["action_correct"] and s["driver_score"] == 0.0
    assert abs(s["score"] - (0.3 + 0.2 + 0.1)) < 1e-9


def test_normalization_case_and_broad_synonyms():
    lbl = _label("explain", [_drv(None, None, "aov", "rate")])
    ans = {"action": "explain",
           "drivers": [_drv("overall", "ALL", "AOV", "Rate")],
           "cited_numbers": []}
    assert driver_score(lbl, ans) == 1.0


def test_factor_child_counts_parent_does_not():
    lbl = _label("explain", [_drv(None, None, "orders", "volume")])
    more_specific = {"action": "explain",
                     "drivers": [_drv(None, None, "buyers", "volume")]}
    too_vague_lbl = _label("explain", [_drv(None, None, "buyers", "volume")])
    too_vague_ans = {"action": "explain",
                     "drivers": [_drv(None, None, "orders", "volume")]}
    assert driver_score(lbl, more_specific) == 1.0
    assert driver_score(too_vague_lbl, too_vague_ans) == 0.0


def test_simpson_mechanism_is_separate():
    ans = {"action": "explain",
           "drivers": [_drv("order_type", "promo", "aov", "rate")]}  # rate!=mix
    assert driver_score(LBL_T3, ans) == 0.0
    assert mechanism_correct(LBL_T3, ans) is False


def test_t5_partial_credit_half_for_one_of_two():
    one = {"action": "explain",
           "drivers": [_drv("country", "Norvik", "orders", "volume", 1.0)]}
    both = {"action": "explain",
            "drivers": [_drv("country", "Norvik", "orders", "volume", 0.7),
                        _drv("category", "fashion", "orders", "volume", 0.3)]}
    assert driver_score(LBL_T5, one) == 0.5
    assert driver_score(LBL_T5, both) == 1.0


def test_abstain_reason_partial_credit():
    right = {"action": "abstain", "abstain_reason": "unmapped_dimension"}
    wrong_reason = {"action": "abstain", "abstain_reason": "data_gap"}
    not_abstain = {"action": "explain",
                   "drivers": [_drv("country", "Meridia", "orders", "volume")]}
    assert driver_score(LBL_T4, right) == 1.0
    assert driver_score(LBL_T4, wrong_reason) == 0.5
    assert driver_score(LBL_T4, not_abstain) == 0.0


def test_no_driver_overconfidence_costs_everything():
    confident = {"action": "explain",
                 "drivers": [_drv("country", "Meridia", "orders", "volume")],
                 "cited_numbers": []}
    s = score_run(LBL_T6, confident, FACTS)
    assert not s["action_correct"] and s["driver_score"] == 0.0


def test_grounding_fabricated_number_fails_even_with_right_driver():
    ans = {"action": "explain",
           "drivers": [_drv("country", "Meridia", "orders", "volume")],
           "cited_numbers": [{"label": "delta_pct", "value": -37.4}]}
    s = score_run(LBL_T1, ans, FACTS)
    assert s["driver_score"] == 1.0 and s["grounding_pass"] is False
    assert abs(s["score"] - (0.3 + 0.4 + 0.1)) < 1e-9


def test_grounding_within_2pct_tolerance_passes():
    ans = {"action": "explain", "drivers": [],
           "cited_numbers": [{"label": "gmv p0", "value": 109000.0},   # -0.4%
                             {"label": "delta", "value": -19.9}]}      # vs -20
    assert grounding_pass(ans, FACTS) is True


def test_grounding_malformed_cited_numbers_fail():
    assert grounding_pass({"cited_numbers": [{"label": "x"}]}, FACTS) is False
    assert grounding_pass({"cited_numbers": "no"}, FACTS) is False
    assert grounding_pass({"cited_numbers": []}, FACTS) is True   # vacuous


def test_sum_share_gate():
    assert sum_share_pass({"action": "explain",
                           "drivers": [{"share_of_move": 0.7},
                                       {"share_of_move": 0.25}]}) is True
    assert sum_share_pass({"action": "explain",
                           "drivers": [{"share_of_move": 0.4}]}) is False
    assert sum_share_pass({"action": "abstain", "drivers": []}) is True


def test_parse_error_scores_zero():
    s = score_run(LBL_T1, None, FACTS, parse_error=True)
    assert s["parse_error"] and s["score"] == 0.0


def test_stability_modal_and_unique():
    a = {"action": "explain",
         "drivers": [_drv("country", "Meridia", "orders", "volume", 0.99)]}
    b = {"action": "explain",   # same drivers, different share -> same key
         "drivers": [_drv("country", "Meridia", "orders", "volume", 0.7)]}
    c = {"action": "abstain", "abstain_reason": "data_gap"}
    keys = [canonical_key(x) for x in (a, b, a, c, a)]
    st = stability(keys)
    assert st["modal_share"] == 0.8 and st["n_unique"] == 2
    assert canonical_key(None, parse_error=True) == "parse_error"
