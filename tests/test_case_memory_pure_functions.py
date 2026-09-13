from __future__ import annotations

from sriti.core.cascade.case_memory import compute_case_bias, compute_outcome_reward


def test_outcome_reward_perfect_outcome_near_one():
    reward = compute_outcome_reward(quality_score=1.0, quality_passed=True, cost_usd=0.0)
    assert reward == 1.0


def test_outcome_reward_worst_outcome_near_zero():
    reward = compute_outcome_reward(quality_score=0.0, quality_passed=False, cost_usd=0.01)
    assert reward == 0.0


def test_outcome_reward_expensive_call_penalized_even_if_high_quality():
    cheap = compute_outcome_reward(quality_score=1.0, quality_passed=True, cost_usd=0.0)
    expensive = compute_outcome_reward(quality_score=1.0, quality_passed=True, cost_usd=0.01)
    assert expensive < cheap


def test_case_bias_empty_cases_returns_empty_dict():
    assert compute_case_bias([]) == {}


def test_case_bias_rewards_above_threshold_are_positive():
    cases = [
        {"model_used": "m1", "outcome_reward": 0.9},
        {"model_used": "m1", "outcome_reward": 0.8},
    ]
    bias = compute_case_bias(cases)
    assert bias["m1"] > 0


def test_case_bias_rewards_below_threshold_are_negative():
    cases = [
        {"model_used": "m2", "outcome_reward": 0.1},
    ]
    bias = compute_case_bias(cases)
    assert bias["m2"] < 0
