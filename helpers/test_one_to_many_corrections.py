"""
Comprehensive tests for 1:N relationship correction algorithm.

Tests cover:
- Basic correction functionality
- Confidence filtering
- Edge cases and corner cases
- Different probability distributions
- Boundary conditions
- Error handling
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from helpers.one_to_many_corrections import (
    prepare_pair_df,
    calculate_corrections_for_pair,
    is_correction_confident,
    filter_high_confidence_corrections,
    _calculate_dynamic_confidence_threshold
)


# ============================================================================
# Test Helper Functions
# ============================================================================

def assert_correction_plan_structure(plan, parent_col, child_col):
    """Assert that correction plan has correct structure."""
    assert plan is not None, "Correction plan should not be None"
    assert 'column_to_correct' in plan, "Plan should have 'column_to_correct'"
    assert 'reference_column' in plan, "Plan should have 'reference_column'"
    assert 'corrections' in plan, "Plan should have 'corrections'"
    assert 'total_changes' in plan, "Plan should have 'total_changes'"
    assert plan['column_to_correct'] == parent_col, f"Expected column_to_correct={parent_col}, got {plan['column_to_correct']}"
    assert plan['reference_column'] == child_col, f"Expected reference_column={child_col}, got {plan['reference_column']}"


def assert_correction_structure(correction):
    """Assert that correction dict has correct structure."""
    assert 'violating_value' in correction, "Correction should have 'violating_value'"
    assert 'current_values' in correction, "Correction should have 'current_values'"
    assert 'correct_value' in correction, "Correction should have 'correct_value'"
    assert 'num_changes' in correction, "Correction should have 'num_changes'"
    assert 'rows_to_correct' in correction, "Correction should have 'rows_to_correct'"


# ============================================================================
# Basic Functionality Tests
# ============================================================================

def test_basic_correction_simple_case():
    """Test basic correction with simple violation."""
    print("Test 1: Basic correction - simple violation")
    df = pd.DataFrame({
        'parent': ['A', 'B', 'A'],  # child=1 maps to both A and B
        'child': [1, 1, 2]
    })
    
    plan = calculate_corrections_for_pair(df, 'parent', 'child', drop_na=False)
    
    assert plan is not None, "Should return correction plan"
    assert_correction_plan_structure(plan, 'parent', 'child')
    assert len(plan['corrections']) == 1, f"Expected 1 correction, got {len(plan['corrections'])}"
    assert plan['total_changes'] == 1, f"Expected 1 change, got {plan['total_changes']}"
    
    correction = plan['corrections'][0]
    assert_correction_structure(correction)
    assert correction['violating_value'] == 1, f"Expected violating_value=1, got {correction['violating_value']}"
    assert correction['correct_value'] == 'A', f"Expected correct_value='A', got {correction['correct_value']}"
    assert correction['num_changes'] == 1, f"Expected num_changes=1, got {correction['num_changes']}"
    print("✓ PASSED\n")


def test_basic_correction_multiple_violations():
    """Test correction with multiple violations."""
    print("Test 2: Basic correction - multiple violations")
    df = pd.DataFrame({
        'parent': ['A', 'B', 'A', 'C', 'B', 'A'],
        'child': [1, 1, 2, 2, 3, 3]  # child=1: A,B; child=2: A,C; child=3: B,A
    })
    
    plan = calculate_corrections_for_pair(df, 'parent', 'child', drop_na=False)
    
    assert plan is not None, "Should return correction plan"
    assert len(plan['corrections']) == 3, f"Expected 3 corrections, got {len(plan['corrections'])}"
    assert plan['total_changes'] == 3, f"Expected 3 changes, got {plan['total_changes']}"
    
    # Check that all violations are addressed
    violating_children = {c['violating_value'] for c in plan['corrections']}
    assert violating_children == {1, 2, 3}, f"Expected violations for children {1, 2, 3}, got {violating_children}"
    print("✓ PASSED\n")


def test_basic_correction_no_violations():
    """Test when there are no violations."""
    print("Test 3: Basic correction - no violations")
    df = pd.DataFrame({
        'parent': ['A', 'A', 'B', 'B'],
        'child': [1, 2, 3, 4]  # Each child maps to exactly one parent
    })
    
    plan = calculate_corrections_for_pair(df, 'parent', 'child', drop_na=False)
    
    assert plan is None, "Should return None when no violations exist"
    print("✓ PASSED\n")


def test_basic_correction_perfect_1n():
    """Test perfect 1:N relationship (no corrections needed)."""
    print("Test 4: Basic correction - perfect 1:N")
    df = pd.DataFrame({
        'parent': ['A', 'A', 'A', 'B', 'B'],
        'child': [1, 2, 3, 4, 5]  # Each child maps to exactly one parent
    })
    
    plan = calculate_corrections_for_pair(df, 'parent', 'child', drop_na=False)
    
    assert plan is None, "Should return None for perfect 1:N"
    print("✓ PASSED\n")


# ============================================================================
# Edge Cases and Corner Cases
# ============================================================================

def test_empty_dataframe():
    """Test with empty dataframe."""
    print("Test 5: Edge case - empty dataframe")
    df = pd.DataFrame({'parent': [], 'child': []})
    
    plan = calculate_corrections_for_pair(df, 'parent', 'child', drop_na=False)
    
    assert plan is None, "Should return None for empty dataframe"
    print("✓ PASSED\n")


def test_single_row():
    """Test with single row."""
    print("Test 6: Edge case - single row")
    df = pd.DataFrame({'parent': ['A'], 'child': [1]})
    
    plan = calculate_corrections_for_pair(df, 'parent', 'child', drop_na=False)
    
    assert plan is None, "Should return None for single row (no violations possible)"
    print("✓ PASSED\n")


def test_all_rows_same():
    """Test with all rows having same values."""
    print("Test 7: Edge case - all rows same")
    df = pd.DataFrame({
        'parent': ['A', 'A', 'A'],
        'child': [1, 1, 1]
    })
    
    plan = calculate_corrections_for_pair(df, 'parent', 'child', drop_na=False)
    
    assert plan is None, "Should return None when all rows are identical (no violations)"
    print("✓ PASSED\n")


def test_single_violation_all_rows():
    """Test where all rows violate but only one violation type."""
    print("Test 8: Edge case - single violation affecting all rows")
    df = pd.DataFrame({
        'parent': ['A', 'B', 'A', 'B'],
        'child': [1, 1, 1, 1]  # All rows have child=1, which maps to both A and B
    })
    
    plan = calculate_corrections_for_pair(df, 'parent', 'child', drop_na=False)
    
    assert plan is not None, "Should return correction plan"
    assert len(plan['corrections']) == 1, f"Expected 1 correction, got {len(plan['corrections'])}"
    assert plan['total_changes'] == 2, f"Expected 2 changes, got {plan['total_changes']}"
    print("✓ PASSED\n")


def test_na_values_with_drop_na():
    """Test with NA values when drop_na=True."""
    print("Test 9: Edge case - NA values with drop_na=True")
    df = pd.DataFrame({
        'parent': ['A', None, 'B', 'A', None],
        'child': [1, 1, 1, 2, 2]
    })
    
    plan = calculate_corrections_for_pair(df, 'parent', 'child', drop_na=True)
    
    # After dropping NA, we have:
    # - child=1: A (row 0), B (row 2) - violation (maps to both A and B)
    # - child=2: A (row 3) - no violation
    # So there should be a correction for child=1
    assert plan is not None, "Should return correction plan (child=1 maps to both A and B)"
    assert len(plan['corrections']) == 1, f"Expected 1 correction, got {len(plan['corrections'])}"
    assert plan['corrections'][0]['violating_value'] == 1, "Should correct child=1 violation"
    print("✓ PASSED\n")


def test_na_values_without_drop_na():
    """Test with NA values when drop_na=False."""
    print("Test 10: Edge case - NA values with drop_na=False")
    df = pd.DataFrame({
        'parent': ['A', None, 'B', 'A'],
        'child': [1, 1, 1, 2]
    })
    
    plan = calculate_corrections_for_pair(df, 'parent', 'child', drop_na=False)
    
    # child=1 maps to A, None, B (3 unique values) - violation
    assert plan is not None, "Should return correction plan"
    assert len(plan['corrections']) >= 1, "Should have at least 1 correction"
    print("✓ PASSED\n")


def test_missing_columns():
    """Test with missing columns."""
    print("Test 11: Edge case - missing columns")
    df = pd.DataFrame({'col1': [1, 2, 3]})
    
    try:
        plan = calculate_corrections_for_pair(df, 'parent', 'child', drop_na=False)
        assert False, "Should raise KeyError for missing columns"
    except KeyError:
        print("✓ PASSED (correctly raised KeyError)\n")


def test_very_large_dataset():
    """Test with very large dataset."""
    print("Test 12: Edge case - very large dataset")
    # Create 10,000 rows with violations
    n = 10000
    parent_vals = ['A'] * (n // 2) + ['B'] * (n // 2)
    child_vals = list(range(n))
    
    # Make child=0 map to both A and B
    parent_vals[0] = 'A'
    parent_vals[n // 2] = 'B'
    child_vals[0] = 0
    child_vals[n // 2] = 0
    
    df = pd.DataFrame({
        'parent': parent_vals,
        'child': child_vals
    })
    
    plan = calculate_corrections_for_pair(df, 'parent', 'child', drop_na=False)
    
    assert plan is not None, "Should return correction plan"
    assert plan['total_changes'] == 1, f"Expected 1 change, got {plan['total_changes']}"
    print("✓ PASSED\n")


def test_single_parent_multiple_children():
    """Test one parent with many children (should be valid)."""
    print("Test 13: Edge case - single parent, multiple children")
    df = pd.DataFrame({
        'parent': ['A'] * 100,
        'child': list(range(100))
    })
    
    plan = calculate_corrections_for_pair(df, 'parent', 'child', drop_na=False)
    
    assert plan is None, "Should return None (valid 1:N relationship)"
    print("✓ PASSED\n")


def test_single_child_multiple_parents():
    """Test one child with multiple parents (violation)."""
    print("Test 14: Edge case - single child, multiple parents")
    df = pd.DataFrame({
        'parent': ['A', 'B', 'C', 'D', 'E'],
        'child': [1, 1, 1, 1, 1]
    })
    
    plan = calculate_corrections_for_pair(df, 'parent', 'child', drop_na=False)
    
    assert plan is not None, "Should return correction plan"
    assert len(plan['corrections']) == 1, f"Expected 1 correction, got {len(plan['corrections'])}"
    # Most frequent parent should be chosen (all equal, so first one)
    assert plan['total_changes'] == 4, f"Expected 4 changes, got {plan['total_changes']}"
    print("✓ PASSED\n")


# ============================================================================
# Probability Distribution Tests
# ============================================================================

def test_uniform_distribution():
    """Test with uniform distribution (equal frequencies)."""
    print("Test 15: Probability distribution - uniform (equal frequencies)")
    # child=1 maps to A, B, C equally (10 each)
    df = pd.DataFrame({
        'parent': ['A'] * 10 + ['B'] * 10 + ['C'] * 10,
        'child': [1] * 30
    })
    
    plan = calculate_corrections_for_pair(df, 'parent', 'child', drop_na=False)
    
    assert plan is not None, "Should return correction plan"
    correction = plan['corrections'][0]
    # Should choose first value (A) as correct
    assert correction['correct_value'] == 'A', f"Expected correct_value='A', got {correction['correct_value']}"
    assert correction['num_changes'] == 20, f"Expected 20 changes, got {correction['num_changes']}"
    print("✓ PASSED\n")


def test_highly_skewed_distribution():
    """Test with highly skewed distribution (90-10 split)."""
    print("Test 16: Probability distribution - highly skewed (90-10)")
    # child=1 maps to A (90 times) and B (10 times)
    df = pd.DataFrame({
        'parent': ['A'] * 90 + ['B'] * 10,
        'child': [1] * 100
    })
    
    plan = calculate_corrections_for_pair(df, 'parent', 'child', drop_na=False)
    
    assert plan is not None, "Should return correction plan"
    correction = plan['corrections'][0]
    assert correction['correct_value'] == 'A', f"Expected correct_value='A', got {correction['correct_value']}"
    assert correction['num_changes'] == 10, f"Expected 10 changes, got {correction['num_changes']}"
    print("✓ PASSED\n")


def test_moderately_skewed_distribution():
    """Test with moderately skewed distribution (70-30 split)."""
    print("Test 17: Probability distribution - moderately skewed (70-30)")
    # child=1 maps to A (70 times) and B (30 times)
    df = pd.DataFrame({
        'parent': ['A'] * 70 + ['B'] * 30,
        'child': [1] * 100
    })
    
    plan = calculate_corrections_for_pair(df, 'parent', 'child', drop_na=False)
    
    assert plan is not None, "Should return correction plan"
    correction = plan['corrections'][0]
    assert correction['correct_value'] == 'A', f"Expected correct_value='A', got {correction['correct_value']}"
    assert correction['num_changes'] == 30, f"Expected 30 changes, got {correction['num_changes']}"
    print("✓ PASSED\n")


def test_slightly_skewed_distribution():
    """Test with slightly skewed distribution (55-45 split)."""
    print("Test 18: Probability distribution - slightly skewed (55-45)")
    # child=1 maps to A (55 times) and B (45 times)
    df = pd.DataFrame({
        'parent': ['A'] * 55 + ['B'] * 45,
        'child': [1] * 100
    })
    
    plan = calculate_corrections_for_pair(df, 'parent', 'child', drop_na=False)
    
    assert plan is not None, "Should return correction plan"
    correction = plan['corrections'][0]
    assert correction['correct_value'] == 'A', f"Expected correct_value='A', got {correction['correct_value']}"
    assert correction['num_changes'] == 45, f"Expected 45 changes, got {correction['num_changes']}"
    print("✓ PASSED\n")


def test_three_way_split():
    """Test with three-way split."""
    print("Test 19: Probability distribution - three-way split")
    # child=1 maps to A (50), B (30), C (20)
    df = pd.DataFrame({
        'parent': ['A'] * 50 + ['B'] * 30 + ['C'] * 20,
        'child': [1] * 100
    })
    
    plan = calculate_corrections_for_pair(df, 'parent', 'child', drop_na=False)
    
    assert plan is not None, "Should return correction plan"
    correction = plan['corrections'][0]
    assert correction['correct_value'] == 'A', f"Expected correct_value='A', got {correction['correct_value']}"
    assert correction['num_changes'] == 50, f"Expected 50 changes, got {correction['num_changes']}"
    print("✓ PASSED\n")


def test_four_way_split():
    """Test with four-way split."""
    print("Test 20: Probability distribution - four-way split")
    # child=1 maps to A (40), B (30), C (20), D (10)
    df = pd.DataFrame({
        'parent': ['A'] * 40 + ['B'] * 30 + ['C'] * 20 + ['D'] * 10,
        'child': [1] * 100
    })
    
    plan = calculate_corrections_for_pair(df, 'parent', 'child', drop_na=False)
    
    assert plan is not None, "Should return correction plan"
    correction = plan['corrections'][0]
    assert correction['correct_value'] == 'A', f"Expected correct_value='A', got {correction['correct_value']}"
    assert correction['num_changes'] == 60, f"Expected 60 changes, got {correction['num_changes']}"
    print("✓ PASSED\n")


def test_power_law_distribution():
    """Test with power law distribution."""
    print("Test 21: Probability distribution - power law")
    # child=1 maps to values with power law: A (64), B (16), C (4), D (1)
    df = pd.DataFrame({
        'parent': ['A'] * 64 + ['B'] * 16 + ['C'] * 4 + ['D'] * 1,
        'child': [1] * 85
    })
    
    plan = calculate_corrections_for_pair(df, 'parent', 'child', drop_na=False)
    
    assert plan is not None, "Should return correction plan"
    correction = plan['corrections'][0]
    assert correction['correct_value'] == 'A', f"Expected correct_value='A', got {correction['correct_value']}"
    assert correction['num_changes'] == 21, f"Expected 21 changes, got {correction['num_changes']}"
    print("✓ PASSED\n")


def test_binomial_distribution():
    """Test with binomial-like distribution."""
    print("Test 22: Probability distribution - binomial-like")
    # child=1 maps to A (80) and B (20) - like binomial
    df = pd.DataFrame({
        'parent': ['A'] * 80 + ['B'] * 20,
        'child': [1] * 100
    })
    
    plan = calculate_corrections_for_pair(df, 'parent', 'child', drop_na=False)
    
    assert plan is not None, "Should return correction plan"
    correction = plan['corrections'][0]
    assert correction['correct_value'] == 'A', f"Expected correct_value='A', got {correction['correct_value']}"
    assert correction['num_changes'] == 20, f"Expected 20 changes, got {correction['num_changes']}"
    print("✓ PASSED\n")


def test_exponential_distribution():
    """Test with exponential-like distribution."""
    print("Test 23: Probability distribution - exponential-like")
    # child=1 maps to values with exponential decay: A (50), B (25), C (12), D (6), E (3)
    df = pd.DataFrame({
        'parent': ['A'] * 50 + ['B'] * 25 + ['C'] * 12 + ['D'] * 6 + ['E'] * 3,
        'child': [1] * 96
    })
    
    plan = calculate_corrections_for_pair(df, 'parent', 'child', drop_na=False)
    
    assert plan is not None, "Should return correction plan"
    correction = plan['corrections'][0]
    assert correction['correct_value'] == 'A', f"Expected correct_value='A', got {correction['correct_value']}"
    assert correction['num_changes'] == 46, f"Expected 46 changes, got {correction['num_changes']}"
    print("✓ PASSED\n")


# ============================================================================
# Confidence Filtering Tests
# ============================================================================

def test_confidence_dynamic_threshold():
    """Test dynamic confidence threshold calculation."""
    print("Test 24: Confidence - dynamic threshold")
    
    assert _calculate_dynamic_confidence_threshold(100) == 60.0, "Sample size 100 should return 60%"
    assert _calculate_dynamic_confidence_threshold(50) == 65.0, "Sample size 50 should return 65%"
    assert _calculate_dynamic_confidence_threshold(20) == 70.0, "Sample size 20 should return 70%"
    assert _calculate_dynamic_confidence_threshold(10) == 75.0, "Sample size 10 should return 75%"
    assert _calculate_dynamic_confidence_threshold(5) == 80.0, "Sample size 5 should return 80%"
    assert _calculate_dynamic_confidence_threshold(2) == 85.0, "Sample size 2 should return 85%"
    assert _calculate_dynamic_confidence_threshold(1) == 85.0, "Sample size 1 should return 85%"
    
    print("✓ PASSED\n")


def test_confidence_min_count_threshold():
    """Test minimum count threshold filtering."""
    print("Test 25: Confidence - minimum count threshold")
    
    # Correction with total count = 3, threshold = 5 (should fail)
    correction_low = {
        'current_values': {'A': 2, 'B': 1},
        'num_changes': 1
    }
    assert not is_correction_confident(correction_low, min_count_threshold=5, ratio_threshold=3.0), \
        "Should not be confident (count < threshold)"
    
    # Correction with total count = 10, threshold = 5 (should pass count check)
    correction_high = {
        'current_values': {'A': 9, 'B': 1},
        'num_changes': 1
    }
    # Should pass: count=10 >= 5, percentage=90% >= 75% (for size 10), ratio=9 >= 3
    assert is_correction_confident(correction_high, min_count_threshold=5, ratio_threshold=3.0), \
        "Should be confident"
    
    print("✓ PASSED\n")


def test_confidence_percentage_threshold():
    """Test percentage threshold filtering."""
    print("Test 26: Confidence - percentage threshold")
    
    # Sample size 20, needs 70% threshold
    # Correction with 65% (should fail)
    correction_low_pct = {
        'current_values': {'A': 13, 'B': 7},  # 13/20 = 65%
        'num_changes': 7
    }
    assert not is_correction_confident(correction_low_pct, min_count_threshold=1, ratio_threshold=3.0), \
        "Should not be confident (65% < 70% threshold for size 20)"
    
    # Correction with 75% (should pass)
    correction_high_pct = {
        'current_values': {'A': 15, 'B': 5},  # 15/20 = 75%
        'num_changes': 5
    }
    # Should pass: count=20 >= 1, percentage=75% >= 70%, ratio=15/5=3 >= 3
    assert is_correction_confident(correction_high_pct, min_count_threshold=1, ratio_threshold=3.0), \
        "Should be confident"
    
    print("✓ PASSED\n")


def test_confidence_ratio_threshold():
    """Test ratio threshold filtering."""
    print("Test 27: Confidence - ratio threshold")
    
    # Correction with ratio 2.5 (should fail, needs 3.0)
    correction_low_ratio = {
        'current_values': {'A': 10, 'B': 4},  # ratio = 10/4 = 2.5
        'num_changes': 4
    }
    assert not is_correction_confident(correction_low_ratio, min_count_threshold=1, ratio_threshold=3.0), \
        "Should not be confident (ratio 2.5 < 3.0)"
    
    # Correction with ratio 4.0 (should pass)
    correction_high_ratio = {
        'current_values': {'A': 12, 'B': 3},  # ratio = 12/3 = 4.0
        'num_changes': 3
    }
    # Should pass: count=15 >= 1, percentage=80% >= 75% (for size 15), ratio=4 >= 3
    assert is_correction_confident(correction_high_ratio, min_count_threshold=1, ratio_threshold=3.0), \
        "Should be confident"
    
    print("✓ PASSED\n")


def test_confidence_all_criteria():
    """Test that all three criteria must be met."""
    print("Test 28: Confidence - all criteria must be met")
    
    # Fails count threshold
    correction1 = {
        'current_values': {'A': 3, 'B': 1},
        'num_changes': 1
    }
    assert not is_correction_confident(correction1, min_count_threshold=5, ratio_threshold=3.0), \
        "Should fail count threshold"
    
    # Fails percentage threshold
    correction2 = {
        'current_values': {'A': 6, 'B': 4},  # 60% < 70% for size 10
        'num_changes': 4
    }
    assert not is_correction_confident(correction2, min_count_threshold=1, ratio_threshold=3.0), \
        "Should fail percentage threshold"
    
    # Fails ratio threshold
    correction3 = {
        'current_values': {'A': 8, 'B': 4},  # ratio 2 < 3
        'num_changes': 4
    }
    assert not is_correction_confident(correction3, min_count_threshold=1, ratio_threshold=3.0), \
        "Should fail ratio threshold"
    
    # Passes all criteria
    correction4 = {
        'current_values': {'A': 15, 'B': 5},  # count=20, 75%, ratio=3
        'num_changes': 5
    }
    assert is_correction_confident(correction4, min_count_threshold=1, ratio_threshold=3.0), \
        "Should pass all criteria"
    
    print("✓ PASSED\n")


def test_filter_high_confidence_corrections():
    """Test filtering corrections by confidence."""
    print("Test 29: Confidence - filter high confidence corrections")
    
    correction_plan = {
        'column_to_correct': 'parent',
        'reference_column': 'child',
        'corrections': [
            {
                'violating_value': 1,
                'current_values': {'A': 15, 'B': 5},  # Confident: count=20, 75%, ratio=3
                'correct_value': 'A',
                'num_changes': 5,
                'rows_to_correct': [1, 2, 3, 4, 5]
            },
            {
                'violating_value': 2,
                'current_values': {'A': 3, 'B': 2},  # Not confident: count=5 < 10, ratio=1.5 < 3
                'correct_value': 'A',
                'num_changes': 2,
                'rows_to_correct': [6, 7]
            },
            {
                'violating_value': 3,
                'current_values': {'A': 10, 'B': 1},  # Confident: count=11, 91%, ratio=10
                'correct_value': 'A',
                'num_changes': 1,
                'rows_to_correct': [8]
            }
        ],
        'total_changes': 8,
        'pair_df': pd.DataFrame({'parent': ['A'] * 10, 'child': [1] * 10}),
        'direction': 'parent → child'
    }
    
    filtered = filter_high_confidence_corrections(correction_plan, min_count_threshold=5, ratio_threshold=3.0)
    
    assert len(filtered['confident_corrections']) == 2, f"Expected 2 confident corrections, got {len(filtered['confident_corrections'])}"
    assert len(filtered['non_confident_corrections']) == 1, f"Expected 1 non-confident correction, got {len(filtered['non_confident_corrections'])}"
    assert filtered['confident_total_changes'] == 6, f"Expected 6 confident changes, got {filtered['confident_total_changes']}"
    assert filtered['non_confident_total_changes'] == 2, f"Expected 2 non-confident changes, got {filtered['non_confident_total_changes']}"
    
    print("✓ PASSED\n")


def test_filter_empty_corrections():
    """Test filtering when no corrections exist."""
    print("Test 30: Confidence - filter empty corrections")
    
    correction_plan = {
        'column_to_correct': 'parent',
        'reference_column': 'child',
        'corrections': [],
        'total_changes': 0,
        'pair_df': pd.DataFrame({'parent': ['A'], 'child': [1]}),
        'direction': 'parent → child'
    }
    
    filtered = filter_high_confidence_corrections(correction_plan, min_count_threshold=5, ratio_threshold=3.0)
    
    assert len(filtered['confident_corrections']) == 0, "Should have 0 confident corrections"
    assert len(filtered['non_confident_corrections']) == 0, "Should have 0 non-confident corrections"
    assert filtered['confident_total_changes'] == 0, "Should have 0 confident changes"
    assert filtered['non_confident_total_changes'] == 0, "Should have 0 non-confident changes"
    
    print("✓ PASSED\n")


# ============================================================================
# Boundary Condition Tests
# ============================================================================

def test_boundary_min_count_threshold_1():
    """Test with min_count_threshold = 1."""
    print("Test 31: Boundary - min_count_threshold = 1")
    
    correction = {
        'current_values': {'A': 2, 'B': 1},  # count=3, ratio=2, percentage=66.67%
        'num_changes': 1
    }
    
    # For count=3, dynamic threshold is 85% (very small sample)
    # A has 66.67% which is less than 85%, so should fail percentage threshold
    assert not is_correction_confident(correction, min_count_threshold=1, ratio_threshold=3.0), \
        "Should fail percentage threshold (66.67% < 85%)"
    
    # Even with ratio threshold 2.0, should still fail percentage threshold
    assert not is_correction_confident(correction, min_count_threshold=1, ratio_threshold=2.0), \
        "Should still fail percentage threshold even with lower ratio threshold"
    
    # Test with a correction that meets percentage threshold
    correction_pass = {
        'current_values': {'A': 9, 'B': 1},  # count=10, ratio=9, percentage=90%
        'num_changes': 1
    }
    # For count=10, threshold is 75%, A has 90% > 75%, ratio=9 > 2.0, should pass
    assert is_correction_confident(correction_pass, min_count_threshold=1, ratio_threshold=2.0), \
        "Should pass with higher percentage (90% > 75%)"
    
    print("✓ PASSED\n")


def test_boundary_min_count_threshold_exact():
    """Test with count exactly at threshold."""
    print("Test 32: Boundary - count exactly at threshold")
    
    correction = {
        'current_values': {'A': 4, 'B': 1},  # count=5 (exactly at threshold)
        'num_changes': 1
    }
    
    # Should pass: count=5 >= 5, percentage=80% >= 80% (for size 5), ratio=4 >= 3
    assert is_correction_confident(correction, min_count_threshold=5, ratio_threshold=3.0), \
        "Should pass when count equals threshold"
    
    print("✓ PASSED\n")


def test_boundary_percentage_exact():
    """Test with percentage exactly at threshold."""
    print("Test 33: Boundary - percentage exactly at threshold")
    
    # Sample size 20, needs 70% threshold
    correction = {
        'current_values': {'A': 14, 'B': 6},  # 14/20 = 70% (exactly at threshold)
        'num_changes': 6
    }
    
    # Should pass: count=20 >= 1, percentage=70% >= 70%, ratio=14/6=2.33 < 3 (fails)
    assert not is_correction_confident(correction, min_count_threshold=1, ratio_threshold=3.0), \
        "Should fail ratio threshold"
    
    # With ratio threshold 2.0, should pass
    assert is_correction_confident(correction, min_count_threshold=1, ratio_threshold=2.0), \
        "Should pass with lower ratio threshold"
    
    print("✓ PASSED\n")


def test_boundary_ratio_exact():
    """Test with ratio exactly at threshold."""
    print("Test 34: Boundary - ratio exactly at threshold")
    
    correction = {
        'current_values': {'A': 15, 'B': 5},  # ratio = 15/5 = 3.0 (exactly at threshold)
        'num_changes': 5
    }
    
    # Should pass: count=20 >= 1, percentage=75% >= 70% (for size 20), ratio=3 >= 3
    assert is_correction_confident(correction, min_count_threshold=1, ratio_threshold=3.0), \
        "Should pass when ratio equals threshold"
    
    print("✓ PASSED\n")


def test_boundary_single_value():
    """Test with only one value (no violation, but test edge case)."""
    print("Test 35: Boundary - single value (no violation)")
    
    correction = {
        'current_values': {'A': 10},  # Only one value
        'num_changes': 0
    }
    
    # This shouldn't happen in practice (no violation), but test the function
    # With only one value, ratio check is skipped, but this is a valid 1:N
    # The function should handle it gracefully
    assert is_correction_confident(correction, min_count_threshold=1, ratio_threshold=3.0), \
        "Should pass (only one value, no ratio check needed)"
    
    print("✓ PASSED\n")


def test_boundary_two_values_equal():
    """Test with two values having equal frequency."""
    print("Test 36: Boundary - two values equal frequency")
    
    correction = {
        'current_values': {'A': 5, 'B': 5},  # Equal frequency, ratio = 1.0
        'num_changes': 5
    }
    
    # Should fail: ratio=1 < 3
    assert not is_correction_confident(correction, min_count_threshold=1, ratio_threshold=3.0), \
        "Should fail ratio threshold (ratio=1 < 3)"
    
    print("✓ PASSED\n")


def test_boundary_very_small_sample():
    """Test with very small sample size."""
    print("Test 37: Boundary - very small sample (size 2)")
    
    correction = {
        'current_values': {'A': 2, 'B': 0},  # Actually only A exists
        'num_changes': 0
    }
    
    # This case shouldn't occur (B has 0 count), but test edge case
    # If only A exists, it's not really a violation
    # But if we have A:2, B:0, total=2, needs 85% threshold, A=100% > 85%, ratio check skipped
    assert is_correction_confident(correction, min_count_threshold=1, ratio_threshold=3.0), \
        "Should pass (only one value)"
    
    print("✓ PASSED\n")


def test_boundary_very_large_sample():
    """Test with very large sample size."""
    print("Test 38: Boundary - very large sample (size 1000)")
    
    correction = {
        'current_values': {'A': 600, 'B': 400},  # 60% for A
        'num_changes': 400
    }
    
    # Should pass: count=1000 >= 1, percentage=60% >= 60% (for size 1000), ratio=600/400=1.5 < 3 (fails)
    assert not is_correction_confident(correction, min_count_threshold=1, ratio_threshold=3.0), \
        "Should fail ratio threshold"
    
    # With ratio threshold 1.5, should pass
    assert is_correction_confident(correction, min_count_threshold=1, ratio_threshold=1.5), \
        "Should pass with lower ratio threshold"
    
    print("✓ PASSED\n")


# ============================================================================
# Complex Scenario Tests
# ============================================================================

def test_multiple_violations_mixed_confidence():
    """Test multiple violations with mixed confidence levels."""
    print("Test 39: Complex - multiple violations, mixed confidence")
    
    df = pd.DataFrame({
        'parent': ['A'] * 15 + ['B'] * 5 + ['A'] * 3 + ['B'] * 2 + ['A'] * 10 + ['B'] * 1,
        'child': [1] * 20 + [2] * 5 + [3] * 11
    })
    
    plan = calculate_corrections_for_pair(df, 'parent', 'child', drop_na=False)
    filtered = filter_high_confidence_corrections(plan, min_count_threshold=5, ratio_threshold=3.0)
    
    assert plan is not None, "Should return correction plan"
    assert len(plan['corrections']) == 3, f"Expected 3 corrections, got {len(plan['corrections'])}"
    
    # child=1: A(15), B(5) - ratio=3, count=20, 75% - CONFIDENT
    # child=2: A(3), B(2) - ratio=1.5, count=5 - NOT CONFIDENT
    # child=3: A(10), B(1) - ratio=10, count=11, 91% - CONFIDENT
    
    assert len(filtered['confident_corrections']) == 2, f"Expected 2 confident corrections, got {len(filtered['confident_corrections'])}"
    assert len(filtered['non_confident_corrections']) == 1, f"Expected 1 non-confident correction, got {len(filtered['non_confident_corrections'])}"
    
    print("✓ PASSED\n")


def test_correction_with_ties():
    """Test correction when multiple values tie for most frequent."""
    print("Test 40: Complex - ties in frequency")
    
    # child=1 maps to A(10), B(10), C(5) - A and B tie
    df = pd.DataFrame({
        'parent': ['A'] * 10 + ['B'] * 10 + ['C'] * 5,
        'child': [1] * 25
    })
    
    plan = calculate_corrections_for_pair(df, 'parent', 'child', drop_na=False)
    
    assert plan is not None, "Should return correction plan"
    correction = plan['corrections'][0]
    # Should choose first value (A) when there's a tie
    assert correction['correct_value'] == 'A', f"Expected correct_value='A' (first in tie), got {correction['correct_value']}"
    assert correction['num_changes'] == 15, f"Expected 15 changes, got {correction['num_changes']}"
    
    print("✓ PASSED\n")


def test_correction_with_three_way_tie():
    """Test correction with three-way tie."""
    print("Test 41: Complex - three-way tie")
    
    # child=1 maps to A(5), B(5), C(5) - three-way tie
    df = pd.DataFrame({
        'parent': ['A'] * 5 + ['B'] * 5 + ['C'] * 5,
        'child': [1] * 15
    })
    
    plan = calculate_corrections_for_pair(df, 'parent', 'child', drop_na=False)
    
    assert plan is not None, "Should return correction plan"
    correction = plan['corrections'][0]
    # Should choose first value (A)
    assert correction['correct_value'] == 'A', f"Expected correct_value='A', got {correction['correct_value']}"
    assert correction['num_changes'] == 10, f"Expected 10 changes, got {correction['num_changes']}"
    
    print("✓ PASSED\n")


def test_correction_with_many_unique_values():
    """Test correction with many unique parent values."""
    print("Test 42: Complex - many unique parent values")
    
    # child=1 maps to 10 different parent values
    df = pd.DataFrame({
        'parent': [chr(65 + i) for i in range(10)] + ['A'] * 10,  # A-J once, then A 10 times
        'child': [1] * 20
    })
    
    plan = calculate_corrections_for_pair(df, 'parent', 'child', drop_na=False)
    
    assert plan is not None, "Should return correction plan"
    correction = plan['corrections'][0]
    # A should be most frequent (11 times)
    assert correction['correct_value'] == 'A', f"Expected correct_value='A', got {correction['correct_value']}"
    assert correction['num_changes'] == 9, f"Expected 9 changes, got {correction['num_changes']}"
    
    print("✓ PASSED\n")


def test_correction_preserves_index():
    """Test that correction preserves original dataframe indices."""
    print("Test 43: Complex - index preservation")
    
    # Create dataframe with non-sequential index
    df = pd.DataFrame({
        'parent': ['A', 'B', 'A', 'C'],
        'child': [1, 1, 2, 2]
    }, index=[10, 20, 30, 40])
    
    plan = calculate_corrections_for_pair(df, 'parent', 'child', drop_na=False)
    
    assert plan is not None, "Should return correction plan"
    assert len(plan['corrections']) == 2, f"Expected 2 corrections, got {len(plan['corrections'])}"
    
    # Check that row indices are preserved
    for correction in plan['corrections']:
        assert len(correction['rows_to_correct']) > 0, "Should have rows to correct"
        # Indices should be from original dataframe
        for idx in correction['rows_to_correct']:
            assert idx in df.index, f"Index {idx} should be in original dataframe index"
    
    print("✓ PASSED\n")


# ============================================================================
# Data Type Tests
# ============================================================================

def test_numeric_parent_values():
    """Test with numeric parent values."""
    print("Test 44: Data type - numeric parent values")
    
    df = pd.DataFrame({
        'parent': [10, 20, 10, 30],
        'child': [1, 1, 2, 2]
    })
    
    plan = calculate_corrections_for_pair(df, 'parent', 'child', drop_na=False)
    
    assert plan is not None, "Should return correction plan"
    correction = plan['corrections'][0]
    assert correction['correct_value'] == 10, f"Expected correct_value=10, got {correction['correct_value']}"
    print("✓ PASSED\n")


def test_numeric_child_values():
    """Test with numeric child values."""
    print("Test 45: Data type - numeric child values")
    
    df = pd.DataFrame({
        'parent': ['A', 'B', 'A', 'C'],
        'child': [1.5, 1.5, 2.5, 2.5]
    })
    
    plan = calculate_corrections_for_pair(df, 'parent', 'child', drop_na=False)
    
    assert plan is not None, "Should return correction plan"
    assert len(plan['corrections']) == 2, f"Expected 2 corrections, got {len(plan['corrections'])}"
    print("✓ PASSED\n")


def test_mixed_types():
    """Test with mixed data types."""
    print("Test 46: Data type - mixed types")
    
    df = pd.DataFrame({
        'parent': ['A', 1, 'A', 2],
        'child': [1, 'X', 2, 'X']
    })
    
    plan = calculate_corrections_for_pair(df, 'parent', 'child', drop_na=False)
    
    assert plan is not None, "Should return correction plan"
    # child='X' maps to both 1 and 2
    assert len(plan['corrections']) >= 1, "Should have at least 1 correction"
    print("✓ PASSED\n")


def test_string_with_special_characters():
    """Test with strings containing special characters."""
    print("Test 47: Data type - special characters in strings")
    
    df = pd.DataFrame({
        'parent': ['A-B', 'A-B', 'C_D', 'C_D', 'E.F'],
        'child': [1, 1, 2, 2, 3]
    })
    
    plan = calculate_corrections_for_pair(df, 'parent', 'child', drop_na=False)
    
    assert plan is None, "Should return None (no violations)"
    print("✓ PASSED\n")


def test_unicode_characters():
    """Test with unicode characters."""
    print("Test 48: Data type - unicode characters")
    
    df = pd.DataFrame({
        'parent': ['北京', '上海', '北京', '广州'],
        'child': [1, 1, 2, 2]
    })
    
    plan = calculate_corrections_for_pair(df, 'parent', 'child', drop_na=False)
    
    assert plan is not None, "Should return correction plan"
    correction = plan['corrections'][0]
    assert correction['correct_value'] == '北京', f"Expected correct_value='北京', got {correction['correct_value']}"
    print("✓ PASSED\n")


# ============================================================================
# prepare_pair_df Tests
# ============================================================================

def test_prepare_pair_df_basic():
    """Test basic prepare_pair_df functionality."""
    print("Test 49: prepare_pair_df - basic")
    
    df = pd.DataFrame({
        'col1': ['A', 'B', 'C'],
        'col2': [1, 2, 3],
        'col3': [10, 20, 30]
    })
    
    result = prepare_pair_df(df, 'col1', 'col2', drop_na=False)
    
    assert result is not None, "Should return dataframe"
    assert len(result) == 3, f"Expected 3 rows, got {len(result)}"
    assert list(result.columns) == ['col1', 'col2'], f"Expected columns ['col1', 'col2'], got {list(result.columns)}"
    print("✓ PASSED\n")


def test_prepare_pair_df_drop_na():
    """Test prepare_pair_df with drop_na=True."""
    print("Test 50: prepare_pair_df - drop_na=True")
    
    df = pd.DataFrame({
        'col1': ['A', None, 'B', 'C', None],
        'col2': [1, 2, None, 3, 4]
    })
    
    result = prepare_pair_df(df, 'col1', 'col2', drop_na=True)
    
    assert result is not None, "Should return dataframe"
    # Only rows where both col1 and col2 are not NA: row 0 (A, 1) and row 3 (C, 3)
    assert len(result) == 2, f"Expected 2 rows after dropping NA, got {len(result)}"
    print("✓ PASSED\n")


def test_prepare_pair_df_empty_after_drop():
    """Test prepare_pair_df when all rows are dropped."""
    print("Test 51: prepare_pair_df - empty after drop_na")
    
    df = pd.DataFrame({
        'col1': [None, None],
        'col2': [None, None]
    })
    
    result = prepare_pair_df(df, 'col1', 'col2', drop_na=True)
    
    assert result is None, "Should return None when all rows are dropped"
    print("✓ PASSED\n")


# ============================================================================
# Integration Tests
# ============================================================================

def test_end_to_end_correction_flow():
    """Test complete correction flow from calculation to filtering."""
    print("Test 52: Integration - end-to-end correction flow")
    
    # Create dataset with violations
    df = pd.DataFrame({
        'parent': ['A'] * 20 + ['B'] * 5 + ['A'] * 10 + ['B'] * 2 + ['A'] * 15 + ['B'] * 1,
        'child': [1] * 25 + [2] * 12 + [3] * 16
    })
    
    # Step 1: Calculate corrections
    plan = calculate_corrections_for_pair(df, 'parent', 'child', drop_na=False)
    assert plan is not None, "Should calculate corrections"
    
    # Step 2: Filter by confidence
    filtered = filter_high_confidence_corrections(plan, min_count_threshold=5, ratio_threshold=3.0)
    
    # Step 3: Verify structure
    assert 'confident_corrections' in filtered, "Should have confident_corrections"
    assert 'non_confident_corrections' in filtered, "Should have non_confident_corrections"
    assert filtered['confident_total_changes'] + filtered['non_confident_total_changes'] == plan['total_changes'], \
        "Total changes should match"
    
    print("✓ PASSED\n")


def test_multiple_pairs_independent():
    """Test that corrections for different pairs are independent."""
    print("Test 53: Integration - multiple pairs independent")
    
    df = pd.DataFrame({
        'parent1': ['A', 'B', 'A'],
        'child1': [1, 1, 2],
        'parent2': ['X', 'Y', 'X'],
        'child2': [10, 10, 20]
    })
    
    plan1 = calculate_corrections_for_pair(df, 'parent1', 'child1', drop_na=False)
    plan2 = calculate_corrections_for_pair(df, 'parent2', 'child2', drop_na=False)
    
    assert plan1 is not None, "Should calculate corrections for pair 1"
    assert plan2 is not None, "Should calculate corrections for pair 2"
    assert plan1['column_to_correct'] == 'parent1', "Plan 1 should correct parent1"
    assert plan2['column_to_correct'] == 'parent2', "Plan 2 should correct parent2"
    
    print("✓ PASSED\n")


# ============================================================================
# Run All Tests
# ============================================================================

def run_all_tests():
    """Run all tests."""
    print("=" * 80)
    print("Running comprehensive tests for 1:N correction algorithm")
    print("=" * 80 + "\n")
    
    tests = [
        # Basic functionality
        test_basic_correction_simple_case,
        test_basic_correction_multiple_violations,
        test_basic_correction_no_violations,
        test_basic_correction_perfect_1n,
        
        # Edge cases
        test_empty_dataframe,
        test_single_row,
        test_all_rows_same,
        test_single_violation_all_rows,
        test_na_values_with_drop_na,
        test_na_values_without_drop_na,
        test_missing_columns,
        test_very_large_dataset,
        test_single_parent_multiple_children,
        test_single_child_multiple_parents,
        
        # Probability distributions
        test_uniform_distribution,
        test_highly_skewed_distribution,
        test_moderately_skewed_distribution,
        test_slightly_skewed_distribution,
        test_three_way_split,
        test_four_way_split,
        test_power_law_distribution,
        test_binomial_distribution,
        test_exponential_distribution,
        
        # Confidence filtering
        test_confidence_dynamic_threshold,
        test_confidence_min_count_threshold,
        test_confidence_percentage_threshold,
        test_confidence_ratio_threshold,
        test_confidence_all_criteria,
        test_filter_high_confidence_corrections,
        test_filter_empty_corrections,
        
        # Boundary conditions
        test_boundary_min_count_threshold_1,
        test_boundary_min_count_threshold_exact,
        test_boundary_percentage_exact,
        test_boundary_ratio_exact,
        test_boundary_single_value,
        test_boundary_two_values_equal,
        test_boundary_very_small_sample,
        test_boundary_very_large_sample,
        
        # Complex scenarios
        test_multiple_violations_mixed_confidence,
        test_correction_with_ties,
        test_correction_with_three_way_tie,
        test_correction_with_many_unique_values,
        test_correction_preserves_index,
        
        # Data types
        test_numeric_parent_values,
        test_numeric_child_values,
        test_mixed_types,
        test_string_with_special_characters,
        test_unicode_characters,
        
        # prepare_pair_df
        test_prepare_pair_df_basic,
        test_prepare_pair_df_drop_na,
        test_prepare_pair_df_empty_after_drop,
        
        # Integration
        test_end_to_end_correction_flow,
        test_multiple_pairs_independent,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"✗ FAILED: {test.__name__}")
            print(f"  {e}\n")
            failed += 1
        except Exception as e:
            print(f"✗ ERROR in {test.__name__}: {e}\n")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("=" * 80)
    print(f"Tests completed: {passed} passed, {failed} failed")
    print("=" * 80)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)

