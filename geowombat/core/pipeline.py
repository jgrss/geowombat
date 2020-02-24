from abc import ABC, abstractmethod
from contextlib import ExitStack
from pathlib import Path

from ..errors import logger
from ..radiometry import BRDF, LinearAdjustments, RadTransforms

import geowombat as gw
import xarray as xr


rt = RadTransforms()
br = BRDF()
la = LinearAdjustments()


class BaseGeoTasks(ABC):

    @abstractmethod
    def __init__(self,
                 inputs,
                 outputs,
                 tasks,
                 clean,
                 config_args=None,
                 open_args=None,
                 func_args=None,
                 out_args=None):

        self.inputs = inputs
        self.outputs = outputs
        self.tasks = tasks
        self.clean = clean
        self.config_args = config_args if inputs else {}
        self.open_args = open_args if inputs else {}
        self.func_args = func_args if inputs else {}
        self.out_args = out_args if inputs else {}

    @abstractmethod
    def clean(self):
        """Clean intermediate data"""
        pass

    @abstractmethod
    def execute(self, task_id, task, src, **kwargs):
        """Execute a task"""
        pass

    @abstractmethod
    def submit(self):
        """Submit a task pipeline"""
        raise NotImplementedError

    @abstractmethod
    def cleanup(self):
        """Cleanup task outputs"""
        pass

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
        >>> rt = RadTransforms()
        >>>
        >>> tasks = (('A', rt.dn_to_sr), ('B', gw.ndvi))
        >>> clean = ('A')
        >>>
        >>> inputs = {'A': ('input.tif', 'sza.tif', 'saa.tif', 'vza.tif', 'vaa.tif'),
        >>>           'B': 'A'}
        >>>
        >>> # {'task': (func, output)}
        >>> outputs = {'A': 'sr.tif',
        >>>            'B': 'ndvi.tif'}
        >>>
        >>> func_args = {'A': {'meta': 'meta.mtl'}}
        >>>
        >>> open_args = {'chunks': 512}
        >>> config_args = {'sensor': 'l7', 'scale_factor': 0.0001}
        >>> out_args = {'compress': 'lzw', 'overwrite': True}
        >>>
        >>> task = pipeline.GeoTasks(inputs, outputs, tasks, clean, config_args, open_args, func_args, out_args)
        >>>
        >>> task.submit()
        >>> task.cleanup()
    """

    def __init__(self,
                 inputs,
                 outputs,
                 tasks,
                 clean,
                 config_args=None,
                 open_args=None,
                 func_args=None,
                 out_args=None):

        super().__init__(inputs,
                         outputs,
                         tasks,
                         clean,
                         config_args,
                         open_args,
                         func_args,
                         out_args)

    def execute(self, task_id, task, src, **kwargs):

        # Execute the task
        res = task(*src, **kwargs)

        # Write to file, if needed
        # TODO: how to handle in-memory results
        if task_id in self.outputs:
            res.gw.to_raster(self.outputs[task_id], **self.out_args)

    def submit(self):

        with gw.config.update(**self.config_args):

            for task_id, task in self.tasks:

                # Check task keywords
                kwargs = self.func_args[task_id] if task_id in self.func_args else {}

                # Check task input(s)
                if isinstance(self.inputs[task_id], str) and not Path(self.inputs[task_id]).is_file():

                    with gw.open(self.outputs[self.inputs[task_id]]) as src:
                        self.execute(task_id, task, src, **kwargs)

                if isinstance(self.inputs[task_id], str) and Path(self.inputs[task_id]).is_file():

                    with gw.open(self.inputs[task_id], **self.open_args) as src:
                        self.execute(task_id, task, src, **kwargs)

                else:

                    with ExitStack() as stack:

                        # Open input files for the task
                        src = [stack.enter_context(gw.open(fn, **self.open_args)) for fn in self.inputs[task_id]]
                        self.execute(task_id, task, src, **kwargs)

    def cleanup(self):

        for task_id in self.clean:

            fn = Path(self.outputs[task_id])

            if fn.is_file():

                try:
                    fn.unlink()
                except:
                    logger.warning(f'  Could not remove task {task_id} output.')

    def visualize(self):

        from graphviz import Digraph

        dot = Digraph()

        for task_id, task in self.tasks:
            dot.node(task_id, f'Task {task_id}')

        # dot.edges(['AB', 'BR'])

        dot.render(view=True)


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
