from dask.distributed import Client, LocalCluster


class Cluster(object):

    def __init__(self, **kwargs):

        self.kwargs = kwargs
        self.cluster = None
        self.client = None

    def start(self):

        self.cluster = LocalCluster(**self.kwargs)
        self.client = Client(self.cluster)

    def stop(self):

        self.client.close()
        self.cluster.close()

        self.client = None
        self.cluster = None
