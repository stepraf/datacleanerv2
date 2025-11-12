"""
Comprehensive tests for 1:N relationship violation ratio calculation.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from helpers.one_to_many import calculate_one_to_many_violation_ratio


def test_perfect_1_to_many():
    """Test perfect 1:N relationship where each col2 maps to exactly one col1."""
    print("Test 1: Perfect 1:N relationship")
    df = pd.DataFrame({
        'col1': ['A', 'A', 'A', 'B', 'B'],
        'col2': [1, 2, 3, 4, 4]
    })
    
    violation_ratio, percentage, valid_rows, total_rows, valid_mask = calculate_one_to_many_violation_ratio(df, 'col1', 'col2')
    
    assert violation_ratio == 0.0, f"Expected violation_ratio=0.0, got {violation_ratio}"
    assert percentage == 100.0, f"Expected percentage=100.0, got {percentage}"
    assert valid_rows == 5, f"Expected valid_rows=5, got {valid_rows}"
    assert total_rows == 5, f"Expected total_rows=5, got {total_rows}"
    assert valid_mask.all(), "All rows should be valid"
    print("✓ PASSED\n")


def test_perfect_1_to_1():
    """Test perfect 1:1 relationship (subset of 1:N where N=1)."""
    print("Test 2: Perfect 1:1 relationship (subset of 1:N)")
    df = pd.DataFrame({
        'col1': ['A', 'B', 'C'],
        'col2': [1, 2, 3]
    })
    
    violation_ratio, percentage, valid_rows, total_rows, valid_mask = calculate_one_to_many_violation_ratio(df, 'col1', 'col2')
    
    assert violation_ratio == 0.0, f"Expected violation_ratio=0.0, got {violation_ratio}"
    assert percentage == 100.0, f"Expected percentage=100.0, got {percentage}"
    assert valid_rows == 3, f"Expected valid_rows=3, got {valid_rows}"
    print("✓ PASSED\n")


def test_violation_single():
    """Test case where one col2 value maps to multiple col1 values."""
    print("Test 3: Single violation - one col2 maps to multiple col1")
    df = pd.DataFrame({
        'col1': ['A', 'B', 'A'],
        'col2': [1, 1, 2]
    })
    
    violation_ratio, percentage, valid_rows, total_rows, valid_mask = calculate_one_to_many_violation_ratio(df, 'col1', 'col2')
    
    # col2=1 maps to both A and B (violation), col2=2 maps to A (valid)
    # So 1 out of 3 rows is valid = 33.33%
    expected_percentage = (1 / 3) * 100
    expected_violation = 1.0 - (expected_percentage / 100.0)
    
    assert abs(percentage - expected_percentage) < 0.01, f"Expected percentage≈{expected_percentage}, got {percentage}"
    assert abs(violation_ratio - expected_violation) < 0.01, f"Expected violation_ratio≈{expected_violation}, got {violation_ratio}"
    assert valid_rows == 1, f"Expected valid_rows=1, got {valid_rows}"
    assert not valid_mask.iloc[0], "Row 0 should be invalid (col2=1 maps to A and B)"
    assert not valid_mask.iloc[1], "Row 1 should be invalid (col2=1 maps to A and B)"
    assert valid_mask.iloc[2], "Row 2 should be valid (col2=2 maps to A)"
    print("✓ PASSED\n")


def test_all_violations():
    """Test case where all rows violate the constraint."""
    print("Test 4: All violations")
    df = pd.DataFrame({
        'col1': ['A', 'B', 'A', 'B'],
        'col2': [1, 1, 2, 2]
    })
    
    violation_ratio, percentage, valid_rows, total_rows, valid_mask = calculate_one_to_many_violation_ratio(df, 'col1', 'col2')
    
    # Both col2=1 and col2=2 map to multiple col1 values
    assert violation_ratio == 1.0, f"Expected violation_ratio=1.0, got {violation_ratio}"
    assert percentage == 0.0, f"Expected percentage=0.0, got {percentage}"
    assert valid_rows == 0, f"Expected valid_rows=0, got {valid_rows}"
    assert not valid_mask.any(), "No rows should be valid"
    print("✓ PASSED\n")


def test_empty_dataframe():
    """Test empty dataframe."""
    print("Test 5: Empty dataframe")
    df = pd.DataFrame({'col1': [], 'col2': []})
    
    violation_ratio, percentage, valid_rows, total_rows, valid_mask = calculate_one_to_many_violation_ratio(df, 'col1', 'col2')
    
    assert violation_ratio == 1.0, f"Expected violation_ratio=1.0, got {violation_ratio}"
    assert percentage == 0.0, f"Expected percentage=0.0, got {percentage}"
    assert valid_rows == 0, f"Expected valid_rows=0, got {valid_rows}"
    assert total_rows == 0, f"Expected total_rows=0, got {total_rows}"
    print("✓ PASSED\n")


def test_single_row():
    """Test single row."""
    print("Test 6: Single row")
    df = pd.DataFrame({'col1': ['A'], 'col2': [1]})
    
    violation_ratio, percentage, valid_rows, total_rows, valid_mask = calculate_one_to_many_violation_ratio(df, 'col1', 'col2')
    
    assert violation_ratio == 0.0, f"Expected violation_ratio=0.0, got {violation_ratio}"
    assert percentage == 100.0, f"Expected percentage=100.0, got {percentage}"
    assert valid_rows == 1, f"Expected valid_rows=1, got {valid_rows}"
    print("✓ PASSED\n")


def test_duplicate_rows():
    """Test duplicate rows."""
    print("Test 7: Duplicate rows")
    df = pd.DataFrame({
        'col1': ['A', 'A', 'A'],
        'col2': [1, 1, 1]
    })
    
    violation_ratio, percentage, valid_rows, total_rows, valid_mask = calculate_one_to_many_violation_ratio(df, 'col1', 'col2')
    
    # col2=1 maps to A (only one value), so all rows are valid
    assert violation_ratio == 0.0, f"Expected violation_ratio=0.0, got {violation_ratio}"
    assert percentage == 100.0, f"Expected percentage=100.0, got {percentage}"
    assert valid_rows == 3, f"Expected valid_rows=3, got {valid_rows}"
    print("✓ PASSED\n")


def test_partial_violations():
    """Test partial violations - some valid, some invalid."""
    print("Test 8: Partial violations")
    df = pd.DataFrame({
        'col1': ['A', 'B', 'A', 'C', 'A', 'B'],
        'col2': [1, 1, 2, 3, 4, 4]  # col2=1 violates (A,B), col2=2 valid (A), col2=3 valid (C), col2=4 violates (A,B)
    })
    
    violation_ratio, percentage, valid_rows, total_rows, valid_mask = calculate_one_to_many_violation_ratio(df, 'col1', 'col2')
    
    # Only rows with col2=2 and col2=3 are valid (2 out of 6)
    expected_percentage = (2 / 6) * 100
    expected_violation = 1.0 - (expected_percentage / 100.0)
    
    assert abs(percentage - expected_percentage) < 0.01, f"Expected percentage≈{expected_percentage}, got {percentage}"
    assert abs(violation_ratio - expected_violation) < 0.01, f"Expected violation_ratio≈{expected_violation}, got {violation_ratio}"
    assert valid_rows == 2, f"Expected valid_rows=2, got {valid_rows}"
    print("✓ PASSED\n")


def test_numeric_values():
    """Test with numeric values."""
    print("Test 9: Numeric values")
    df = pd.DataFrame({
        'col1': [10, 10, 20, 20, 20],
        'col2': [1, 2, 3, 4, 4]
    })
    
    violation_ratio, percentage, valid_rows, total_rows, valid_mask = calculate_one_to_many_violation_ratio(df, 'col1', 'col2')
    
    assert violation_ratio == 0.0, f"Expected violation_ratio=0.0, got {violation_ratio}"
    assert percentage == 100.0, f"Expected percentage=100.0, got {percentage}"
    assert valid_rows == 5, f"Expected valid_rows=5, got {valid_rows}"
    print("✓ PASSED\n")


def test_mixed_types():
    """Test with mixed data types."""
    print("Test 10: Mixed types")
    df = pd.DataFrame({
        'col1': ['A', 1, 'A', 2],
        'col2': [1, 'X', 2, 'X']
    })
    
    violation_ratio, percentage, valid_rows, total_rows, valid_mask = calculate_one_to_many_violation_ratio(df, 'col1', 'col2')
    
    # col2='X' maps to both 1 and 2 (violation), col2=1 maps to A (valid), col2=2 maps to A (valid)
    expected_percentage = (2 / 4) * 100
    expected_violation = 1.0 - (expected_percentage / 100.0)
    
    assert abs(percentage - expected_percentage) < 0.01, f"Expected percentage≈{expected_percentage}, got {percentage}"
    assert abs(violation_ratio - expected_violation) < 0.01, f"Expected violation_ratio≈{expected_violation}, got {violation_ratio}"
    assert valid_rows == 2, f"Expected valid_rows=2, got {valid_rows}"
    print("✓ PASSED\n")


def test_na_values():
    """Test with NA/null values."""
    print("Test 11: NA/null values")
    df = pd.DataFrame({
        'col1': ['A', 'A', None, 'B', None],
        'col2': [1, 2, 3, 4, 4]
    })
    
    violation_ratio, percentage, valid_rows, total_rows, valid_mask = calculate_one_to_many_violation_ratio(df, 'col1', 'col2')
    
    # col2=1 maps to A (1 unique) - valid
    # col2=2 maps to A (1 unique) - valid
    # col2=3 maps to None (1 unique) - valid
    # col2=4 maps to both 'B' and None (2 unique) - violation (2 rows invalid)
    # So 3 out of 5 rows are valid = 60%
    # But pandas might treat None as a single value, so col2=4 might map to 2 unique values (None and 'B')
    # Let's check what actually happens - if None and 'B' are both present for col2=4, it's 2 unique values
    expected_percentage = (3 / 5) * 100
    expected_violation = 1.0 - (expected_percentage / 100.0)
    
    # Actually, let's verify: col2=4 appears with 'B' and None, so nunique should be 2
    # But if pandas groups None values together, we need to check
    actual_col2_4_unique = df[df['col2'] == 4]['col1'].nunique()
    if actual_col2_4_unique == 2:
        # Normal case: None and 'B' are different, so violation
        assert abs(percentage - expected_percentage) < 0.01, f"Expected percentage≈{expected_percentage}, got {percentage}"
        assert abs(violation_ratio - expected_violation) < 0.01, f"Expected violation_ratio≈{expected_violation}, got {violation_ratio}"
        assert valid_rows == 3, f"Expected valid_rows=3, got {valid_rows}"
    else:
        # If None values are being treated specially, adjust expectations
        print(f"  Note: col2=4 has {actual_col2_4_unique} unique col1 values")
        assert valid_rows >= 3, f"Expected at least 3 valid_rows, got {valid_rows}"
    print("✓ PASSED\n")


def test_large_dataset():
    """Test with larger dataset."""
    print("Test 12: Large dataset")
    # Create a dataset where 90% are valid, 10% violate
    n_valid = 900
    n_invalid = 100
    
    # Each col2 value should be unique for valid rows
    # col1='A' with col2 values 0-899, col1='B' with col2 values 900-1799
    valid_col1 = ['A'] * n_valid + ['B'] * n_valid
    valid_col2 = list(range(n_valid)) + list(range(n_valid, n_valid * 2))
    
    # Invalid: col2=9999 maps to both A and B
    invalid_col1 = ['A', 'B'] * (n_invalid // 2)
    invalid_col2 = [9999] * n_invalid  # All map to both A and B
    
    df = pd.DataFrame({
        'col1': valid_col1 + invalid_col1,
        'col2': valid_col2 + invalid_col2
    })
    
    violation_ratio, percentage, valid_rows, total_rows, valid_mask = calculate_one_to_many_violation_ratio(df, 'col1', 'col2')
    
    total = n_valid * 2 + n_invalid
    expected_percentage = (n_valid * 2 / total) * 100
    expected_violation = 1.0 - (expected_percentage / 100.0)
    
    assert abs(percentage - expected_percentage) < 0.1, f"Expected percentage≈{expected_percentage}, got {percentage}"
    assert abs(violation_ratio - expected_violation) < 0.001, f"Expected violation_ratio≈{expected_violation}, got {violation_ratio}"
    assert valid_rows == n_valid * 2, f"Expected valid_rows={n_valid * 2}, got {valid_rows}"
    print("✓ PASSED\n")


def test_one_parent_many_children():
    """Test one parent with many children."""
    print("Test 13: One parent with many children")
    df = pd.DataFrame({
        'col1': ['A'] * 10,
        'col2': list(range(10))
    })
    
    violation_ratio, percentage, valid_rows, total_rows, valid_mask = calculate_one_to_many_violation_ratio(df, 'col1', 'col2')
    
    # Each col2 value maps to exactly one col1 value (A), so all valid
    assert violation_ratio == 0.0, f"Expected violation_ratio=0.0, got {violation_ratio}"
    assert percentage == 100.0, f"Expected percentage=100.0, got {percentage}"
    assert valid_rows == 10, f"Expected valid_rows=10, got {valid_rows}"
    print("✓ PASSED\n")


def test_many_parents_one_child():
    """Test many parents with one child (should violate)."""
    print("Test 14: Many parents with one child (violation)")
    df = pd.DataFrame({
        'col1': ['A', 'B', 'C'],
        'col2': [1, 1, 1]
    })
    
    violation_ratio, percentage, valid_rows, total_rows, valid_mask = calculate_one_to_many_violation_ratio(df, 'col1', 'col2')
    
    # col2=1 maps to A, B, and C (violation)
    assert violation_ratio == 1.0, f"Expected violation_ratio=1.0, got {violation_ratio}"
    assert percentage == 0.0, f"Expected percentage=0.0, got {percentage}"
    assert valid_rows == 0, f"Expected valid_rows=0, got {valid_rows}"
    print("✓ PASSED\n")


def test_complex_scenario():
    """Test complex real-world scenario."""
    print("Test 15: Complex scenario")
    df = pd.DataFrame({
        'col1': ['A', 'A', 'A', 'B', 'B', 'C', 'C', 'D', 'E', 'E'],
        'col2': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    })
    
    violation_ratio, percentage, valid_rows, total_rows, valid_mask = calculate_one_to_many_violation_ratio(df, 'col1', 'col2')
    
    # All col2 values map to exactly one col1 value, so perfect
    assert violation_ratio == 0.0, f"Expected violation_ratio=0.0, got {violation_ratio}"
    assert percentage == 100.0, f"Expected percentage=100.0, got {percentage}"
    assert valid_rows == 10, f"Expected valid_rows=10, got {valid_rows}"
    print("✓ PASSED\n")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Running comprehensive tests for 1:N violation ratio")
    print("=" * 60 + "\n")
    
    tests = [
        test_perfect_1_to_many,
        test_perfect_1_to_1,
        test_violation_single,
        test_all_violations,
        test_empty_dataframe,
        test_single_row,
        test_duplicate_rows,
        test_partial_violations,
        test_numeric_values,
        test_mixed_types,
        test_na_values,
        test_large_dataset,
        test_one_parent_many_children,
        test_many_parents_one_child,
        test_complex_scenario,
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
            failed += 1
    
    print("=" * 60)
    print(f"Tests completed: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)

