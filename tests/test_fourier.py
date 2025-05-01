import numpy as np
import geowombat as gw
from geowombat.data import l8_224078_20200518_B2
from geowombat.backends.dask_ import Cluster


def test_moving_fourier_transform_single_band():
    """
    Test moving window Fourier transform via gw.moving API
    """
    # Start a local Dask cluster
    cluster = Cluster(
        n_workers=4, threads_per_worker=2, scheduler_port=0, processes=False
    )
    cluster.start()

    # Open single-band Band 2 example
    with gw.open(l8_224078_20200518_B2, chunk=512) as src:
        # Apply moving Fourier transform using stat 'fourier_transform'
        res = gw.moving(src, stat="fourier_transform", w=15, nodata=0)
        # Compute the result array
        arr = res.data.compute()

    # Result should be a NumPy array matching input shape
    assert isinstance(arr, np.ndarray)
    assert arr.shape == src.shape
    # All values should be finite
    assert np.all(np.isfinite(arr))

    # Stop the cluster
    cluster.stop()
