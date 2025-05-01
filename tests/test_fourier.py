import pytest
import numpy as np
import xarray as xr
from geowombat.core.util import MapProcesses


def test_moving_fourier_single_band():
    # Create a single-band test image
    data = np.random.rand(100, 100)
    coords = {"y": np.arange(100), "x": np.arange(100)}
    single_band = xr.DataArray(data, dims=("y", "x"), coords=coords)

    # Apply moving Fourier Transform
    result = MapProcesses.moving_fourier(single_band, w=5)

    # Assert the result contains the expected keys
    assert "fourier_mean" in result
    assert "fourier_variance" in result

    # Assert the shapes match the input
    assert result["fourier_mean"].shape == single_band.shape
    assert result["fourier_variance"].shape == single_band.shape


def test_moving_fourier_rgb():
    # Create an RGB test image
    data = np.random.rand(3, 100, 100)
    coords = {
        "band": ["R", "G", "B"],
        "y": np.arange(100),
        "x": np.arange(100),
    }
    rgb_image = xr.DataArray(data, dims=("band", "y", "x"), coords=coords)

    # Apply moving Fourier Transform
    result = MapProcesses.moving_fourier(rgb_image, w=5)

    # Assert the result contains the expected keys
    assert "fourier_mean" in result
    assert "fourier_variance" in result

    # Assert the shapes match the input
    assert result["fourier_mean"].shape == rgb_image.shape
    assert result["fourier_variance"].shape == rgb_image.shape


def test_moving_fourier_greyscale():
    # Create an RGB test image
    data = np.random.rand(3, 100, 100)
    coords = {
        "band": ["R", "G", "B"],
        "y": np.arange(100),
        "x": np.arange(100),
    }
    rgb_image = xr.DataArray(data, dims=("band", "y", "x"), coords=coords)

    # Apply moving Fourier Transform with greyscale=True
    result = MapProcesses.moving_fourier(rgb_image, w=5, greyscale=True)

    # Assert the result contains the expected keys
    assert "fourier_mean" in result
    assert "fourier_variance" in result

    # Assert the output has a single band
    assert result["fourier_mean"].dims == ("y", "x")
    assert result["fourier_variance"].dims == ("y", "x")

    # Assert the shapes match the input spatial dimensions
    assert result["fourier_mean"].shape == (100, 100)
    assert result["fourier_variance"].shape == (100, 100)
