"""
Tests for drop_na behavior in 1:N relationship calculation.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from helpers.one_to_many import calculate_one_to_many_violation_ratio


def test_drop_na_with_na_in_col1():
    """Test drop_na behavior when col1 has NA values."""
    print("Test: drop_na with NA in col1")
    
    # Create dataframe with NA in col1
    df = pd.DataFrame({
        'col1': ['A', 'A', None, 'B', None],
        'col2': [1, 2, 3, 4, 5]
    })
    
    # Without dropping NA: col2=3 maps to None, col2=5 maps to None (both map to same value, valid)
    # But we need to check if None is treated as a single value
    violation_ratio, percentage, valid_rows, total_rows, valid_mask = calculate_one_to_many_violation_ratio(df, 'col1', 'col2')
    
    print(f"  Without drop_na: percentage={percentage:.2f}%, valid_rows={valid_rows}/{total_rows}")
    print(f"  Violation ratio: {violation_ratio:.4f}")
    
    # With dropping NA: should only analyze rows without NA
    df_no_na = df.dropna(subset=['col1', 'col2'])
    violation_ratio_no_na, percentage_no_na, valid_rows_no_na, total_rows_no_na, valid_mask_no_na = calculate_one_to_many_violation_ratio(df_no_na, 'col1', 'col2')
    
    print(f"  With drop_na: percentage={percentage_no_na:.2f}%, valid_rows={valid_rows_no_na}/{total_rows_no_na}")
    print(f"  Violation ratio: {violation_ratio_no_na:.4f}")
    
    # After dropping NA, we should have 3 rows (col1='A' with col2=1,2 and col1='B' with col2=4)
    assert total_rows_no_na == 3, f"Expected 3 rows after dropping NA, got {total_rows_no_na}"
    assert percentage_no_na == 100.0, f"Expected 100% after dropping NA, got {percentage_no_na}"
    print("✓ PASSED\n")


def test_drop_na_with_na_in_col2():
    """Test drop_na behavior when col2 has NA values."""
    print("Test: drop_na with NA in col2")
    
    # Create dataframe with NA in col2
    df = pd.DataFrame({
        'col1': ['A', 'A', 'B', 'B', 'C'],
        'col2': [1, 2, None, None, 3]
    })
    
    # Without dropping NA: rows with NA in col2 should be invalid
    violation_ratio, percentage, valid_rows, total_rows, valid_mask = calculate_one_to_many_violation_ratio(df, 'col1', 'col2')
    
    print(f"  Without drop_na: percentage={percentage:.2f}%, valid_rows={valid_rows}/{total_rows}")
    print(f"  Violation ratio: {violation_ratio:.4f}")
    
    # Rows with NA in col2 should be invalid
    assert not valid_mask.iloc[2], "Row 2 (col2=NA) should be invalid"
    assert not valid_mask.iloc[3], "Row 3 (col2=NA) should be invalid"
    
    # With dropping NA: should only analyze rows without NA
    df_no_na = df.dropna(subset=['col1', 'col2'])
    violation_ratio_no_na, percentage_no_na, valid_rows_no_na, total_rows_no_na, valid_mask_no_na = calculate_one_to_many_violation_ratio(df_no_na, 'col1', 'col2')
    
    print(f"  With drop_na: percentage={percentage_no_na:.2f}%, valid_rows={valid_rows_no_na}/{total_rows_no_na}")
    print(f"  Violation ratio: {violation_ratio_no_na:.4f}")
    
    # After dropping NA, we should have 3 rows, all valid
    assert total_rows_no_na == 3, f"Expected 3 rows after dropping NA, got {total_rows_no_na}"
    assert percentage_no_na == 100.0, f"Expected 100% after dropping NA, got {percentage_no_na}"
    print("✓ PASSED\n")


def test_drop_na_with_na_in_both():
    """Test drop_na behavior when both columns have NA values."""
    print("Test: drop_na with NA in both columns")
    
    df = pd.DataFrame({
        'col1': ['A', None, 'B', None, 'C'],
        'col2': [1, 2, None, None, 3]
    })
    
    # Without dropping NA
    violation_ratio, percentage, valid_rows, total_rows, valid_mask = calculate_one_to_many_violation_ratio(df, 'col1', 'col2')
    
    print(f"  Without drop_na: percentage={percentage:.2f}%, valid_rows={valid_rows}/{total_rows}")
    
    # With dropping NA
    df_no_na = df.dropna(subset=['col1', 'col2'])
    violation_ratio_no_na, percentage_no_na, valid_rows_no_na, total_rows_no_na, valid_mask_no_na = calculate_one_to_many_violation_ratio(df_no_na, 'col1', 'col2')
    
    print(f"  With drop_na: percentage={percentage_no_na:.2f}%, valid_rows={valid_rows_no_na}/{total_rows_no_na}")
    
    # After dropping NA, we should have 2 rows (A-1 and C-3)
    assert total_rows_no_na == 2, f"Expected 2 rows after dropping NA, got {total_rows_no_na}"
    assert percentage_no_na == 100.0, f"Expected 100% after dropping NA, got {percentage_no_na}"
    print("✓ PASSED\n")


def test_drop_na_violation_case():
    """Test drop_na with violations."""
    print("Test: drop_na with violations")
    
    df = pd.DataFrame({
        'col1': ['A', 'B', 'A', None, 'B'],
        'col2': [1, 1, 2, 3, None]  # col2=1 violates (maps to A and B)
    })
    
    # Without dropping NA
    violation_ratio, percentage, valid_rows, total_rows, valid_mask = calculate_one_to_many_violation_ratio(df, 'col1', 'col2')
    
    print(f"  Without drop_na: percentage={percentage:.2f}%, valid_rows={valid_rows}/{total_rows}")
    
    # With dropping NA
    df_no_na = df.dropna(subset=['col1', 'col2'])
    violation_ratio_no_na, percentage_no_na, valid_rows_no_na, total_rows_no_na, valid_mask_no_na = calculate_one_to_many_violation_ratio(df_no_na, 'col1', 'col2')
    
    print(f"  With drop_na: percentage={percentage_no_na:.2f}%, valid_rows={valid_rows_no_na}/{total_rows_no_na}")
    
    # After dropping NA, we have 3 rows: (A,1), (B,1), (A,2)
    # col2=1 maps to both A and B (violation), col2=2 maps to A (valid)
    # So 1 out of 3 rows is valid = 33.33%
    expected_percentage = (1 / 3) * 100
    assert abs(percentage_no_na - expected_percentage) < 0.01, f"Expected {expected_percentage}%, got {percentage_no_na}%"
    assert valid_rows_no_na == 1, f"Expected 1 valid row, got {valid_rows_no_na}"
    print("✓ PASSED\n")


def run_all_tests():
    """Run all drop_na tests."""
    print("=" * 60)
    print("Running tests for drop_na behavior")
    print("=" * 60 + "\n")
    
    tests = [
        test_drop_na_with_na_in_col1,
        test_drop_na_with_na_in_col2,
        test_drop_na_with_na_in_both,
        test_drop_na_violation_case,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"✗ FAILED: {e}\n")
            failed += 1
        except Exception as e:
            print(f"✗ ERROR: {e}\n")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("=" * 60)
    print(f"Tests completed: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)

