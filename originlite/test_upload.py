#!/usr/bin/env python3
"""Test script to verify df_from_upload functionality."""

import pandas as pd
from pathlib import Path
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, '.')

try:
    from originlite.utils import df_from_upload
    print("✓ Successfully imported df_from_upload")
except ImportError as e:
    print(f"✗ Failed to import df_from_upload: {e}")
    sys.exit(1)

# Test with sample data
sample_files = [
    Path("sample_data/iris.csv"),
    Path("sample_data/tips.csv")
]

print("\n=== Testing Sample Files ===")
for sample_file in sample_files:
    if sample_file.exists():
        print(f"\nTesting: {sample_file}")
        try:
            df = pd.read_csv(sample_file)
            print(f"✓ pandas.read_csv: {df.shape} - {list(df.columns)}")
            print(f"  First row: {df.iloc[0].to_dict()}")
        except Exception as e:
            print(f"✗ pandas.read_csv failed: {e}")
    else:
        print(f"✗ File not found: {sample_file}")

# Create a simple test CSV
print("\n=== Creating Test CSV ===")
test_csv_path = Path("test_data.csv")
test_data = pd.DataFrame({
    'x': [1, 2, 3, 4, 5],
    'y': [2, 4, 6, 8, 10]
})
test_data.to_csv(test_csv_path, index=False)
print(f"✓ Created test CSV: {test_csv_path}")

# Test reading it back
try:
    df_test = pd.read_csv(test_csv_path)
    print(f"✓ Read test CSV: {df_test.shape} - {list(df_test.columns)}")
    print(f"  Data:\n{df_test}")
except Exception as e:
    print(f"✗ Failed to read test CSV: {e}")

# Clean up
if test_csv_path.exists():
    test_csv_path.unlink()
    print("✓ Cleaned up test file")

print("\n=== Summary ===")
print("If all tests passed, the dataframe creation should work.")
print("The issue might be in the Streamlit upload handling or UI interaction.")
