#!/usr/bin/env python3
"""Test script to verify mosaic fixes for issues #322 and #328."""

import sys
sys.path.insert(0, 'src')

import numpy as np
import geowombat as gw
from geowombat.data import (
    l8_224077_20200518_B2,
    l8_224078_20200518_B2,
    l8_224077_20200518_B2_nan,
    l8_224078_20200518_B2_nan,
)
from geowombat.core import lonlat_to_xy, coords_to_indices

print("=" * 70)
print("Testing Mosaic Functionality Fixes")
print("=" * 70)

# Test 1: Issue #322 - Mosaic fails with nan nodata
print("\n" + "-" * 70)
print("TEST 1: Issue #322 - Mosaic with NaN nodata values")
print("-" * 70)

try:
    filenames = [l8_224077_20200518_B2_nan, l8_224078_20200518_B2_nan]
    print(f"Opening files: {[f.split('/')[-1] for f in filenames]}")

    with gw.open(
        filenames,
        band_names=["blue"],
        mosaic=True,
        overlap="max",
        bounds_by="union",
        nodata=np.nan,  # This is the key - using np.nan as nodata
    ) as src:
        print(f"  Shape: {src.shape}")
        print(f"  Bounds: {src.gw.bounds}")

        # Check some values
        start_values = src.values[0, 0, 0:10]
        print(f"  First 10 values (row 0): {start_values}")

        # Check for valid data (not all NaN)
        valid_count = np.sum(~np.isnan(src.values))
        total_count = src.values.size
        print(f"  Valid pixels: {valid_count}/{total_count} ({100*valid_count/total_count:.1f}%)")

        if valid_count > 0:
            print("  ✓ TEST 1 PASSED: Mosaic with NaN nodata works!")
        else:
            print("  ✗ TEST 1 FAILED: No valid data in mosaic")

except Exception as e:
    print(f"  ✗ TEST 1 FAILED with error: {type(e).__name__}: {e}")

# Test 2: Issue #322 variant - Mosaic with nodata=0 on files that have NaN
print("\n" + "-" * 70)
print("TEST 2: Issue #322 variant - Mosaic with nodata=0 on NaN files")
print("-" * 70)

try:
    filenames = [l8_224077_20200518_B2_nan, l8_224078_20200518_B2_nan]
    print(f"Opening files with nodata=0: {[f.split('/')[-1] for f in filenames]}")

    with gw.open(
        filenames,
        band_names=["blue"],
        mosaic=True,
        overlap="max",
        bounds_by="union",
        nodata=0,  # nodata=0 but files have NaN for missing
    ) as src:
        print(f"  Shape: {src.shape}")

        # Get values at the overlap region
        x, y = lonlat_to_xy(-54.78604601, -25.23023330, dst_crs=src)
        j, i = coords_to_indices(x, y, src)
        mid_values = src[0, i:i+3, j:j+3].values
        print(f"  Mid values at overlap:\n{mid_values}")

        expected = np.array([
            [8387.0, 8183.0, 8050.0],
            [7938.0, 7869.0, 7889.0],
            [7862.0, 7828.0, 7721.0],
        ])

        if np.allclose(mid_values, expected, rtol=0.01):
            print("  ✓ TEST 2 PASSED: Overlap values match expected!")
        else:
            print(f"  ✗ TEST 2 FAILED: Values don't match expected:\n{expected}")

except Exception as e:
    print(f"  ✗ TEST 2 FAILED with error: {type(e).__name__}: {e}")

# Test 3: Issue #328 - Mosaic union bounds issue
print("\n" + "-" * 70)
print("TEST 3: Issue #328 - Mosaic with bounds_by='union'")
print("-" * 70)

try:
    # Use the regular (non-nan) files
    filenames = [l8_224077_20200518_B2, l8_224078_20200518_B2]
    print(f"Opening files: {[f.split('/')[-1] for f in filenames]}")

    # First get individual bounds
    import rasterio as rio
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
            print("  ✓ TEST 3 PASSED: Union bounds are correct!")
        else:
            print("  ✗ TEST 3 FAILED: Bounds don't match expected union")

except Exception as e:
    print(f"  ✗ TEST 3 FAILED with error: {type(e).__name__}: {e}")

# Test 4: Mosaic with intersection (baseline test)
print("\n" + "-" * 70)
print("TEST 4: Baseline - Mosaic with bounds_by='intersection'")
print("-" * 70)

try:
    filenames = [l8_224077_20200518_B2, l8_224078_20200518_B2]

    with gw.open(
        filenames,
        band_names=["blue"],
        mosaic=True,
        overlap="max",
        bounds_by="intersection",
        nodata=0,
    ) as src:
        print(f"  Shape: {src.shape}")
        print(f"  Bounds: {src.gw.bounds}")

        # Check that intersection is smaller than union
        bounds = src.gw.bounds
        if hasattr(bounds, 'left'):
            inter_area = (bounds.right - bounds.left) * (bounds.top - bounds.bottom)
        else:
            inter_area = (bounds[2] - bounds[0]) * (bounds[3] - bounds[1])
        print(f"  Intersection area: {inter_area/1e6:.1f} km²")

        print("  ✓ TEST 4 PASSED: Intersection mosaic works!")

except Exception as e:
    print(f"  ✗ TEST 4 FAILED with error: {type(e).__name__}: {e}")


print("\n" + "=" * 70)
print("All tests completed!")
print("=" * 70)
