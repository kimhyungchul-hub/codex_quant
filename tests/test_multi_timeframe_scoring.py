"""
Unit Tests for Multi-Timeframe Scoring System
==============================================

Tests for the 4 core functions in core/multi_timeframe_scoring.py
"""

import pytest
import numpy as np
from core.multi_timeframe_scoring import (
    calculate_consensus_score,
    calculate_advanced_metrics,
    check_position_switching,
    check_exit_condition,
    get_best_entry_tag,
)


class TestCalculateConsensusScore:
    """Tests for calculate_consensus_score function"""
    
    def test_normal_case_all_timeframes(self):
        """Test with all timeframes present"""
        scores = {'5m': 10.0, '10m': 8.0, '30m': 6.0, '1h': 12.0}
        result = calculate_consensus_score(scores)
        # Expected: 0.1*10 + 0.2*8 + 0.3*6 + 0.4*12 = 1.0 + 1.6 + 1.8 + 4.8 = 9.2
        assert abs(result - 9.2) < 0.001
    
    def test_partial_timeframes(self):
        """Test with some timeframes missing"""
        scores = {'10m': 5.0, '1h': 10.0}
        result = calculate_consensus_score(scores)
        # Expected: (0.2*5 + 0.4*10) / (0.2 + 0.4) = 5.0 / 0.6 = 8.333...
        assert abs(result - 8.333) < 0.01
    
    def test_empty_scores(self):
        """Test with empty scores dict"""
        result = calculate_consensus_score({})
        assert result == 0.0
    
    def test_all_negative(self):
        """Test with all negative scores"""
        scores = {'5m': -5.0, '10m': -3.0, '30m': -8.0, '1h': -10.0}
        result = calculate_consensus_score(scores)
        # Expected: 0.1*(-5) + 0.2*(-3) + 0.3*(-8) + 0.4*(-10) = -0.5 - 0.6 - 2.4 - 4.0 = -7.5
        assert abs(result - (-7.5)) < 0.001
    
    def test_custom_weights(self):
        """Test with custom weights"""
        scores = {'5m': 10.0, '10m': 10.0}
        weights = {'5m': 0.7, '10m': 0.3}
        result = calculate_consensus_score(scores, weights)
        # Expected: 0.7*10 + 0.3*10 = 10.0
        assert abs(result - 10.0) < 0.001


class TestCalculateAdvancedMetrics:
    """Tests for calculate_advanced_metrics function"""
    
    def test_normal_case_stable(self):
        """Test with stable scores (low volatility)"""
        history = [10.0] * 150  # 150 samples all at 10.0
        group_score, rank_score = calculate_advanced_metrics(history)
        # Group score should be close to 10.0 - 0 (no stddev) = 10.0
        # Rank score should be close to 10.0
        assert abs(group_score - 10.0) < 0.1
        assert abs(rank_score - 10.0) < 0.1
    
    def test_normal_case_trending(self):
        """Test with trending scores (increasing)"""
        history = list(range(150))  # 0 to 149
        group_score, rank_score = calculate_advanced_metrics(history)
        # Group score = EWMA - StdDev (should be lower due to high volatility)
        # Rank score = Fast EWMA (should be close to recent values ~140+)
        assert group_score < rank_score  # Rank score should be higher for uptrend
        assert rank_score > 100  # Recent values are high
    
    def test_cold_start_insufficient_samples(self):
        """Test with less than 5 samples (cold start)"""
        history = [10.0, 11.0, 12.0]  # Only 3 samples
        group_score, rank_score = calculate_advanced_metrics(history)
        assert group_score == 0.0
        assert rank_score == 0.0
    
    def test_minimum_samples(self):
        """Test with exactly 5 samples (boundary)"""
        history = [10.0, 11.0, 12.0, 13.0, 14.0]
        group_score, rank_score = calculate_advanced_metrics(history)
        # Should not return 0 (passed threshold)
        assert group_score != 0.0 or rank_score != 0.0
    
    def test_high_volatility(self):
        """Test with high volatility scores"""
        history = [10.0, -5.0, 15.0, -10.0, 20.0] * 30  # 150 alternating values
        group_score, rank_score = calculate_advanced_metrics(history)
        # Group score should be significantly penalized due to high StdDev
        assert group_score < rank_score


class TestCheckPositionSwitching:
    """Tests for check_position_switching function"""
    
    def test_should_switch_sufficient_margin(self):
        """Test switching when candidate has sufficient margin"""
        current = {'group_score': 10.0}
        candidate = {'group_score': 12.0}
        fee_rate = 0.0005
        # Switching cost = 0.0005 * 4.0 * 4.0 = 0.008
        # Candidate advantage = 12.0 - 10.0 = 2.0 > 0.008
        result = check_position_switching(current, candidate, fee_rate)
        assert result is True
    
    def test_should_not_switch_insufficient_margin(self):
        """Test not switching when margin is insufficient"""
        current = {'group_score': 10.0}
        candidate = {'group_score': 10.005}
        fee_rate = 0.0005
        # Switching cost = 0.0005 * 4.0 * 4.0 = 0.008
        # Candidate advantage = 10.005 - 10.0 = 0.005 < 0.008
        result = check_position_switching(current, candidate, fee_rate)
        assert result is False
    
    def test_custom_scaling_factor(self):
        """Test with custom scaling factor"""
        current = {'group_score': 10.0}
        candidate = {'group_score': 10.02}
        fee_rate = 0.0005
        scaling_factor = 2.0  # Lower threshold
        # Switching cost = 0.0005 * 4.0 * 2.0 = 0.004
        # Candidate advantage = 10.02 - 10.0 = 0.02 > 0.004
        result = check_position_switching(current, candidate, fee_rate, scaling_factor)
        assert result is True
    
    def test_zero_fee(self):
        """Test with zero fee (always switch if candidate is better)"""
        current = {'group_score': 10.0}
        candidate = {'group_score': 10.001}
        fee_rate = 0.0
        # Switching cost = 0.0
        # Any positive advantage should trigger switch
        result = check_position_switching(current, candidate, fee_rate)
        assert result is True
    
    def test_missing_group_score(self):
        """Test error handling when group_score is missing"""
        current = {'other_field': 10.0}
        candidate = {'group_score': 12.0}
        fee_rate = 0.0005
        with pytest.raises(ValueError):
            check_position_switching(current, candidate, fee_rate)


class TestCheckExitCondition:
    """Tests for check_exit_condition function"""
    
    def test_timeframe_logic_broken(self):
        """Test exit when entry_tag timeframe score drops"""
        position = {'entry_tag': '1h', 'sym': 'BTC/USDT'}
        scores = {'5m': 5.0, '10m': 3.0, '30m': 2.0, '1h': -5.0}
        consensus = 0.0
        should_exit, reason = check_exit_condition(position, scores, consensus)
        assert should_exit is True
        assert 'Timeframe Logic Broken' in reason
        assert '1h' in reason
    
    def test_hard_stop_triggered(self):
        """Test hard stop when consensus is severely negative"""
        position = {'entry_tag': '1h'}
        scores = {'5m': -60.0, '10m': -55.0, '30m': -58.0, '1h': -62.0}
        consensus = -59.0
        should_exit, reason = check_exit_condition(position, scores, consensus, exit_threshold=-50.0)
        assert should_exit is True
        assert 'Hard Stop' in reason
    
    def test_no_exit_all_positive(self):
        """Test no exit when all timeframes are positive"""
        position = {'entry_tag': '1h'}
        scores = {'5m': 5.0, '10m': 8.0, '30m': 10.0, '1h': 12.0}
        consensus = 9.0
        should_exit, reason = check_exit_condition(position, scores, consensus)
        assert should_exit is False
        assert 'No exit condition met' in reason
    
    def test_cold_start_no_entry_tag(self):
        """Test with missing entry_tag (legacy position)"""
        position = {'sym': 'BTC/USDT'}  # No entry_tag
        scores = {'5m': 5.0, '10m': 3.0}
        consensus = 4.0
        should_exit, reason = check_exit_condition(position, scores, consensus)
        assert should_exit is False  # Should not exit without entry_tag
    
    def test_entry_tag_not_in_scores(self):
        """Test when entry_tag timeframe is missing from current scores"""
        position = {'entry_tag': '2h'}  # Not in scores
        scores = {'5m': 5.0, '10m': 3.0, '30m': 2.0, '1h': 8.0}
        consensus = 5.0
        should_exit, reason = check_exit_condition(position, scores, consensus)
        assert should_exit is False
        assert 'Warning' in reason


class TestGetBestEntryTag:
    """Tests for get_best_entry_tag function"""
    
    def test_highest_contribution(self):
        """Test that it returns timeframe with highest weighted contribution"""
        scores = {'5m': 100.0, '10m': 50.0, '30m': 30.0, '1h': 40.0}
        # Contributions: 5m=100*0.1=10, 10m=50*0.2=10, 30m=30*0.3=9, 1h=40*0.4=16
        result = get_best_entry_tag(scores)
        assert result == '1h'
    
    def test_empty_scores(self):
        """Test with empty scores (should return default '1h')"""
        result = get_best_entry_tag({})
        assert result == '1h'
    
    def test_negative_scores(self):
        """Test with negative scores (highest contribution still wins)"""
        scores = {'5m': -10.0, '10m': -5.0, '30m': -3.0, '1h': -1.0}
        # Contributions: 5m=-1, 10m=-1, 30m=-0.9, 1h=-0.4 (highest)
        result = get_best_entry_tag(scores)
        assert result == '1h'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
