import unittest

from geowombat.backends.dask_ import Cluster


class TestDaskCluster(unittest.TestCase):
    def test_cluster(self):
        cl = Cluster(n_workers=2, threads_per_worker=1)
        cl.start()
        self.assertTrue(cl.cluster is not None)
        self.assertTrue(cl.client is not None)
        cl.restart()
        self.assertTrue(cl.cluster is not None)
        self.assertTrue(cl.client is not None)
        cl.stop()
        self.assertTrue(cl.cluster is None)
        self.assertTrue(cl.client is None)


if __name__ == '__main__':
    unittest.main()
