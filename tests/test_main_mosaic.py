#!/usr/bin/env python3
"""Test script to verify mosaic issues exist on main branch."""

import sys
sys.path.insert(0, 'src')

import numpy as np
import geowombat as gw
from geowombat.data import (
    l8_224077_20200518_B2,
    l8_224078_20200518_B2,
)
import rasterio as rio

print("=" * 70)
print(f"Testing Mosaic on MAIN branch (geowombat {gw.__version__})")
print("=" * 70)

# Test: Issue #328 - Mosaic union bounds issue
print("\n" + "-" * 70)
print("TEST: Issue #328 - Mosaic with bounds_by='union'")
print("-" * 70)

try:
    filenames = [l8_224077_20200518_B2, l8_224078_20200518_B2]
    print(f"Opening files: {[f.split('/')[-1] for f in filenames]}")

    # Get individual bounds
    individual_bounds = []
    for fn in filenames:
        with rio.open(fn) as src_rio:
            individual_bounds.append(src_rio.bounds)
            print(f"  {fn.split('/')[-1]} bounds: {src_rio.bounds}")

    # Calculate expected union bounds
    expected_left = min(b.left for b in individual_bounds)
    expected_bottom = min(b.bottom for b in individual_bounds)
    expected_right = max(b.right for b in individual_bounds)
    expected_top = max(b.top for b in individual_bounds)
    print(f"  Expected union bounds: ({expected_left}, {expected_bottom}, {expected_right}, {expected_top})")

    with gw.open(
        filenames,
        band_names=["blue"],
        mosaic=True,
        overlap="max",
        bounds_by="union",
        nodata=0,
    ) as src:
        actual_bounds = src.gw.bounds
        print(f"  Actual mosaic bounds: {actual_bounds}")
        print(f"  Shape: {src.shape}")

        # Handle both tuple and BoundingBox formats
        if hasattr(actual_bounds, 'left'):
            ab_left, ab_bottom, ab_right, ab_top = actual_bounds.left, actual_bounds.bottom, actual_bounds.right, actual_bounds.top
        else:
            ab_left, ab_bottom, ab_right, ab_top = actual_bounds

        # Check if bounds approximately match (with some tolerance for pixel alignment)
        tol = 100  # 100m tolerance
        bounds_match = (
            abs(ab_left - expected_left) < tol and
            abs(ab_bottom - expected_bottom) < tol and
            abs(ab_right - expected_right) < tol and
            abs(ab_top - expected_top) < tol
        )

        if bounds_match:
            print("  ✓ Union bounds are correct")
        else:
            print("  ✗ Union bounds DON'T match expected - ISSUE EXISTS!")
            print(f"    Left diff: {abs(ab_left - expected_left)}")
            print(f"    Bottom diff: {abs(ab_bottom - expected_bottom)}")
            print(f"    Right diff: {abs(ab_right - expected_right)}")
            print(f"    Top diff: {abs(ab_top - expected_top)}")

except Exception as e:
    print(f"  ✗ FAILED with error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("Test completed!")
print("=" * 70)
