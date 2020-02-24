from abc import ABC, abstractmethod
from contextlib import ExitStack

from ..radiometry import BRDF, LinearAdjustments, RadTransforms

import geowombat as gw
import xarray as xr


rt = RadTransforms()
br = BRDF()
la = LinearAdjustments()


class BaseGeoTasks(ABC):

    @abstractmethod
    def __init__(self, inputs, outputs, tasks, config, kwargs, outkwargs):

        self.inputs = inputs
        self.outputs = outputs
        self.tasks = tasks
        self.config = config
        self.kwargs = kwargs
        self.outkwargs = outkwargs

    @abstractmethod
    def clean(self):
        """Clean intermediate data"""
        pass

    @abstractmethod
    def submit(self):
        """Submit a task pipeline"""
        raise NotImplementedError

    def _validate_methods(self, *args):

        if len(args) != len(self.processes):
            raise AttributeError('The lengths do not match.')

        for object_, proc_ in zip(*args, self.processes):

            if not hasattr(object_, proc_):
                raise NameError(f'The {proc_} process is not supported.')

    def __len__(self):
        return len(self.processes)


class GeoTasks(BaseGeoTasks):

    """
    Example:
        >>> import geowombat as gw
        >>> from geowombat.core import pipeline
        >>> from geowombat.radiometry import RadTransforms
        >>>
        >>> rt = RadTransforms()
        >>>
        >>> inputs = {rt.dn_to_sr: ('input.tif', 'sza.tif', 'saa.tif', 'vza.tif', 'vaa.tif'),
        >>>           gw.ndvi: 'dn_to_sr'}
        >>>
        >>> outputs = {rt.dn_to_sr: None,
        >>>            gw.ndvi: 'ndvi.tif'}
        >>>
        >>> kwargs = {rt.dn_to_sr: {'meta': 'meta.mtl'}}
        >>> config = {'sensor': 'l7', 'scale_factor': 0.0001}
        >>> outkwargs = {'compress': 'lzw', 'overwrite': True}
        >>>
        >>> tasks = (rt.dn_to_sr, gw.ndvi)
        >>>
        >>> task = pipeline.GeoTasks(inputs, outputs, tasks, config, kwargs, outkwargs)
        >>> task.submit()
    """

    def __init__(self, inputs, outputs, tasks, config, kwargs, outkwargs):
        super().__init__(inputs, outputs, tasks, config, kwargs, outkwargs)

    def submit(self):

        with gw.config.update(**self.config):

            for task in self.tasks:

                kwargs = self.kwargs[task] if task in self.kwargs else {}

                # TODO
                with ExitStack as stack:
                    files = [stack.enter_context([gw.open(image), gw.open(image)])]

                res = task(*self.inputs[task], **kwargs)

                if (task in self.outputs) and self.outputs[task]:
                    res.gw.to_raster(self.outputs[task], **self.outkwargs)



    def clean(self):
        pass


class LandsatBRDFPipeline(GeoPipeline):

    """
    A pipeline class for Landsat BRDF

    Args:
        processes (tuple): The spectral indices to process.

    Returns:
        ``xarray.DataArray``

    Example:
        >>> import geowombat as gw
        >>> from geowombat.core import pipeline
        >>> from geowombat.radiometry import RadTransforms
        >>>
        >>> rt = RadTransforms()
        >>> meta = rt.get_landsat_coefficients('file.MTL')
        >>>
        >>> task = pipeline.LandsatBRDFPipeline(('dn_to_sr', 'norm_brdf', 'bandpass'))
        >>>
        >>> with gw.open('image.tif') as src, \
        >>>     gw.open('sza.tif') as sza, \
        >>>         gw.open('saa.tif') as saa, \
        >>>             gw.open('vza.tif') as vza, \
        >>>                 gw.open('vaa.tif') as vaa:
        >>>
        >>>     res = task.submit(src, sza, saa, vza, vaa, sensor=meta.sensor, meta=meta)
    """

    def __init__(self, processes):

        super().__init__(processes)
        self._validate_methods([rt, br, la])

    def submit(self, data, *args, **kwargs):

        for i, func_ in enumerate(self.processes):

            if func_ == 'dn_to_sr':

                func = getattr(rt, func_)

                if i == 0:

                    res = func(data,
                               *args,
                               sensor=kwargs['sensor'],
                               meta=kwargs['meta'])

                else:

                    res = func(res,
                               *args,
                               sensor=kwargs['sensor'],
                               meta=kwargs['meta'])

            elif func_ == 'norm_brdf':

                func = getattr(br, func_)

                if i == 0:

                    res = func(data,
                               *args,
                               sensor=kwargs['sensor'],
                               wavelengths=data.band.values.tolist(),
                               out_range=10000.0,
                               nodata=65535)

                else:

                    res = func(res,
                               *args,
                               sensor=kwargs['sensor'],
                               wavelengths=data.band.values.tolist(),
                               out_range=10000.0,
                               nodata=65535)

            elif func_ == 'bandpass':

                func = getattr(la, func_)

                if i == 0:

                    if kwargs['sensor'].lower() in ['l5', 'l7']:

                        res = func(data,
                                   kwargs['sensor'].lower(),
                                   to='l8',
                                   scale_factor=0.0001)

                    else:

                        res = func(res,
                                   kwargs['sensor'].lower(),
                                   to='l8',
                                   scale_factor=0.0001)

        return res


class IndicesPipeline(GeoPipeline):

    """
    A pipeline class for spectral indices

    Args:
        processes (tuple): The spectral indices to process.

    Returns:
        ``xarray.DataArray``

    Example:
        >>> import geowombat as gw
        >>> from geowombat.core import pipeline
        >>>
        >>> task = pipeline.IndicesPipeline(('avi', 'evi2', 'evi', 'nbr', 'ndvi', 'tasseled_cap'))
        >>>
        >>> with gw.open('image.tif') as src:
        >>>     res = task.submit(src, scale_factor=0.0001)
    """

    def __init__(self, processes):

        super().__init__(processes)
        self._validate_methods([gw]*len(processes))

    def submit(self, data, *args, **kwargs):

        attrs = data.attrs.copy()
        results = []

        for vi in self.processes:

            vi_func = getattr(gw, vi)
            results.append(vi_func(data, *args, **kwargs))

        results = xr.concat(results, dim='band').astype('float64')

        return results.assign_attrs(**attrs)
