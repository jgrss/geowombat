import numpy as np
import geowombat as gw
from geowombat.data import l8_224078_20200518_B2


def test_moving_fourier_transform_single_band():
    """
    Test moving window Fourier transform via gw.moving API
    """
    # Open single-band Band 2 example
    with gw.open(l8_224078_20200518_B2) as src:
        # Apply moving Fourier transform using stat 'fourier_transform'
        res = gw.moving(src, stat="fourier_transform", w=15, nodata=32767.0)
        # Compute the result array
        arr = res.data.compute(num_workers=4)

    # Result should be a NumPy array matching input shape
    assert isinstance(arr, np.ndarray)
    assert arr.shape == src.shape
    # All values should be finite
    assert np.all(np.isfinite(arr))
