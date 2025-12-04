import unittest
import numpy as np
import xarray as xr
from geowombat.core.util import MapProcesses


class TestMovingFourier(unittest.TestCase):
    def setUp(self):
        # Create a single-band test image
        self.single_band_data = np.random.rand(100, 100)
        self.single_band_coords = {"y": np.arange(100), "x": np.arange(100)}
        self.single_band = xr.DataArray(
            self.single_band_data,
            dims=("y", "x"),
            coords=self.single_band_coords,
        )

        # Create an RGB test image
        self.rgb_data = np.random.rand(3, 100, 100)
        self.rgb_coords = {
            "band": ["R", "G", "B"],
            "y": np.arange(100),
            "x": np.arange(100),
        }
        self.rgb_image = xr.DataArray(
            self.rgb_data, dims=("band", "y", "x"), coords=self.rgb_coords
        )

    def test_moving_fourier_single_band(self):
        # Apply moving Fourier Transform
        result = MapProcesses.moving_fourier(self.single_band, w=5)

        # Assert the result contains the expected keys
        self.assertIn("fourier_mean", result)
        self.assertIn("fourier_variance", result)

        # Assert the shapes match the input
        self.assertEqual(result["fourier_mean"].shape, self.single_band.shape)
        self.assertEqual(
            result["fourier_variance"].shape, self.single_band.shape
        )

    def test_moving_fourier_rgb(self):
        # Apply moving Fourier Transform
        result = MapProcesses.moving_fourier(self.rgb_image, w=5)

        # Assert the result contains the expected keys
        self.assertIn("fourier_mean", result)
        self.assertIn("fourier_variance", result)

        # Assert the shapes match the input
        self.assertEqual(result["fourier_mean"].shape, self.rgb_image.shape)
        self.assertEqual(
            result["fourier_variance"].shape, self.rgb_image.shape
        )


if __name__ == "__main__":
    unittest.main()
