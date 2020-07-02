import os
from abc import ABC, abstractmethod
from contextlib import ExitStack
from pathlib import Path
from copy import copy
import string
import random
from datetime import datetime
# from inspect import signature

from ..errors import logger
from .. import config as gw_config
from .. import open as gw_open

import xarray as xr
import graphviz


PROC_NODE_ATTRS = {
    "shape": "oval",
    "color": "#3454b4",
    "fontcolor": "#131f43",
    "style": "filled",
    "fillcolor": "#c6d2f6"}

PROC_EDGE_ATTRS = {"color": "#3454b4", "style": "bold"}

CONFIG_NODE_ATTRS = {
    "shape": "diamond",
    "color": "black",
    "fontcolor": "#131f43",
    "style": "rounded,filled",
    "fillcolor": "none"}

CONFIG_EDGE_ATTRS = {"color": "grey", "style": "dashed"}

OUT_NODE_ATTRS = {
    "shape": "pentagon",
    "color": "black",
    "fontcolor": "#131f43",
    "style": "rounded,filled",
    "fillcolor": "none"}

OUT_EDGE_ATTRS = {"color": "#edcec6", "style": "dashed"}

INPUT_NODE_ATTRS = {
    "shape": "box",
    "color": "#b49434",
    "fontcolor": "#2d250d",
    "style": "filled",
    "fillcolor": "#f3e3b3"}

INPUT_EDGE_ATTRS = {"color": "#b49434"}

VAR_NODE_ATTRS = {"shape": "box", "color": "#555555", "fontcolor": "#555555", "style": "dashed"}

VAR_EDGE_ATTRS = {"color": "#555555"}


class BaseGeoTask(ABC):

    @abstractmethod
    def __init__(self,
                 inputs,
                 outputs,
                 tasks,
                 clean=None,
                 config_args=None,
                 open_args=None,
                 func_args=None,
                 out_args=None,
                 log_file=None):

        self.inputs = inputs
        self.outputs = outputs
        self.tasks = tasks
        self.clean = clean if clean else {}
        self.config_args = config_args if inputs else {}
        self.open_args = open_args if inputs else {}
        self.func_args = func_args if inputs else {}
        self.out_args = out_args if inputs else {}
        self.log_file = log_file

        _log_home = os.path.abspath(os.path.dirname(__file__))

        if not os.access(_log_home, os.W_OK):
            _log_home = os.path.expanduser('~')

        if not self.log_file:
            self.log_file = os.path.join(_log_home, 'task.log')

    def copy(self):
        return copy(self)

    def __add__(self, other):

        """
        Add another pipeline
        """

        self_inputs_copy = self.inputs.copy()
        self_outputs_copy = self.outputs.copy()
        self_func_args_copy = self.func_args.copy()
        self_config_args_copy = self.config_args.copy()
        self_open_args_copy = self.open_args.copy()
        self_out_args_copy = self.out_args.copy()
        self_clean_copy = self.clean.copy()

        tasks = list()

        for task_id, task in self.tasks:
            tasks.append((task_id, task))

        for task_id, task in other.tasks:
            tasks.append((task_id, task))

        self_inputs_copy.update(other.inputs)
        self_outputs_copy.update(other.outputs)
        self_func_args_copy.update(other.func_args)
        self_config_args_copy.update(other.config_args)
        self_open_args_copy.update(other.open_args)
        self_out_args_copy.update(other.out_args)
        self_clean_copy.update(other.clean)

        return GeoTask(self_inputs_copy,
                       self_outputs_copy,
                       tuple(tasks),
                       clean=self_clean_copy,
                       config_args=self_config_args_copy,
                       open_args=self_open_args_copy,
                       func_args=self_func_args_copy,
                       out_args=self_out_args_copy)

    @abstractmethod
    def execute(self, task_id, task, src, task_results, attrs, **kwargs):
        """Execute a task"""
        pass

    @abstractmethod
    def submit(self):
        """Submit a task pipeline"""
        raise NotImplementedError

    def _cleanup(self, level, task_id):

        clean_level = self.clean[task_id]

        if clean_level == level:

            fn = Path(self.outputs[task_id])

            if fn.is_file():

                try:
                    fn.unlink()
                except:
                    logger.warning(f'  Could not remove task {task_id} output.')

    def _check_task(self, task_id):
        return True if Path(self.outputs[task_id]).is_file() else False

    def _set_log(self, task_id):

        letters_digits = string.ascii_letters + string.digits
        random_id = ''.join(random.choice(letters_digits) for i in range(0, 9))

        task_output = self.outputs[task_id]
        when = datetime.now().strftime('%A, %d-%m-%Y at %H:%M:%S')

        task_log = f"{when} | {task_output} | task_id-{random_id}"

        return f'{task_log} ok\n' if self._check_task(task_id) else f'{task_log} failed\n'

    def _log_task(self, task_id):

        if Path(self.log_file).is_file():

            with open(self.log_file, mode='r') as f:
                lines = f.readlines()

        else:

            lines = []

            with open(self.log_file, mode='w') as f:
                f.writelines(lines)

        with open(self.log_file, mode='r+') as f:

            lines.append(self._set_log(task_id))
            f.writelines(lines)

    def __len__(self):
        return len(self.processes)


class GraphBuilder(object):

    """
    Reference:
        https://github.com/benbovy/xarray-simlab/blob/master/xsimlab/dot.py
    """

    def visualize(self, **kwargs):

        if not kwargs:
            kwargs = {'rankdir': 'LR'}

        self.seen = set()
        self.inputs_seen = set()
        self.outputs_seen = set()

        counter = 0

        self.g = graphviz.Digraph()
        self.g.subgraph(graph_attr=kwargs)

        for task_id, task in self.tasks:

            if task_id not in self.seen:

                self.seen.add(task_id)
                self.g.node(task_id, label=f'Task {task_id}: {task.__name__}', **PROC_NODE_ATTRS)

            if task_id != list(self.tasks)[0][0]:

                if isinstance(self.inputs[task_id], str):
                    self.g.edge(list(self.tasks)[counter-1][0], task_id, **PROC_EDGE_ATTRS)
                else:

                    task_list_ = list(list(zip(*self.tasks))[0])

                    for ctask in self.inputs[task_id]:
                        if ctask in task_list_:
                            cidx = task_list_.index(ctask)
                        else:
                            cidx = counter-1

                        self.g.edge(list(self.tasks)[cidx][0], task_id, **PROC_EDGE_ATTRS)

            for config_key, config_setting in self.config_args.items():

                with self.g.subgraph(name='cluster_0') as c:

                    c.attr(style='filled', color='lightgrey')
                    c.node(config_key, label=f'{config_key}: {config_setting}', **CONFIG_NODE_ATTRS)
                    c.attr(label='geowombat.config.update() args')

                self.g.edge(config_key, list(self.tasks)[counter-1][0], **CONFIG_EDGE_ATTRS)

            counter += 1

        counter = 0

        for task_id, output_ in self.outputs.items():

            if output_ not in self.outputs_seen:
                self.outputs_seen.add(output_)

            node_attrs = INPUT_NODE_ATTRS.copy()
            edge_attrs = INPUT_EDGE_ATTRS.copy()

            if task_id in self.clean:

                if self.clean[task_id] == 'task':
                    node_attrs['color'] = 'red'
                elif self.clean[task_id] == 'pipeline':
                    node_attrs['color'] = 'purple'

            if task_id == list(self.outputs.keys())[-1]:
                node_attrs['color'] = 'blue'

            node_attrs['style'] = 'dashed'
            edge_attrs['style'] = 'dashed'

            self.g.node(f'{task_id} {self.outputs[task_id]}', label=self.outputs[task_id], **node_attrs)
            self.g.edge(task_id, f'{task_id} {self.outputs[task_id]}', weight='200', **edge_attrs)

            for out_key, out_setting in self.out_args.items():

                if not output_.startswith('mem|'):

                    with self.g.subgraph(name='cluster_1') as c:

                        c.attr(style='filled', color='#a6d5ab')
                        c.node(out_key, label=f'{out_key}: {out_setting}', **OUT_NODE_ATTRS)
                        c.attr(label='geowombat.to_raster() args')

                    self.g.edge(out_key, f'{task_id} {self.outputs[task_id]}', **OUT_EDGE_ATTRS)

            if counter > 0:

                task_id_ = list(self.outputs.keys())[counter-1]

                if not self.outputs[task_id_].startswith('mem|'):
                    self.g.edge(f'{task_id_} {self.outputs[task_id_]}', task_id, weight='200', **edge_attrs)

            counter += 1

        counter = 0

        for task_id, inputs_ in self.inputs.items():

            if isinstance(inputs_, str):
                self._add_inputs(counter, task_id, [inputs_])
            else:
                self._add_inputs(counter, task_id, inputs_)

                counter += 1

        for task_id, params in self.func_args.items():

            for k, param in params.items():

                self.g.node(f'{task_id} {k}', label=f'{k}: {param}', **VAR_NODE_ATTRS)
                self.g.edge(f'{task_id} {k}', task_id, weight='200', **VAR_EDGE_ATTRS)

        return self.g

    def _add_inputs(self, counter, task_id, input_list):

        for input_ in input_list:

            if isinstance(input_, tuple) or isinstance(input_, list):
                gen = input_
            else:
                gen = (input_,)

            for gen_item in gen:

                if gen_item in self.outputs_seen:
                    task_id_b = None
                else:

                    if gen_item not in self.inputs_seen:

                        self.inputs_seen.add(gen_item)

                        gen_label = Path(gen_item).name if Path(gen_item).is_file() else gen_item

                        if gen_item not in self.outputs:
                            self.g.node(f'{task_id} {gen_item}', label=gen_label, **INPUT_NODE_ATTRS)

                        task_id_b = task_id

                    else:
                        task_id_b = list(self.tasks)[counter-1][0]

                if task_id_b:

                    if gen_item not in self.outputs:

                        for itask, iinputs in self.inputs.items():
                            if gen_item in iinputs:
                                task_id_b = itask
                                break

                        self.g.edge(f'{task_id_b} {gen_item}', task_id, weight='200', **INPUT_EDGE_ATTRS)


class GeoTask(BaseGeoTask, GraphBuilder):

    """
    A Geo-task scheduler

    Args:
        inputs (dict)
        outputs (dict)
        tasks (tuple)
        clean (Optional[dict])
        config_args (Optional[dict])
        open_args (Optional[dict])
        func_args (Optional[dict])
        out_args (Optional[dict])
        log_file (Optional[str])

    Example:
        >>> import geowombat as gw
        >>> from geowombat.data import l8_224078_20200518_B3, l8_224078_20200518_B4, l8_224078_20200518
        >>> from geowombat.tasks import GeoTask
        >>>
        >>> # Task a and b take 1 input file
        >>> # Task c takes 2 input files
        >>> # Task d takes the output of task c
        >>> # Task e takes the outputs of a, b, and d
        >>> inputs = {'a': l8_224078_20200518, 'b': l8_224078_20200518, 'c': (l8_224078_20200518_B3, l8_224078_20200518_B4), 'd': 'c', 'e': ('a', 'b', 'd')}
        >>>
        >>> # The output task names
        >>> # All tasks are in-memory DataArrays
        >>> outputs = {'a': 'mem|r1', 'b': 'mem|r2', 'c': 'mem|r3', 'd': 'mem|mean', 'e': 'mem|stack'}
        >>>
        >>> # Task a and b compute the `norm_diff`
        >>> # Task c concatenates two images
        >>> # Task d takes the mean of c
        >>> # Task e concatenates a, b, and d
        >>> tasks = (('a', gw.norm_diff), ('b', gw.norm_diff), ('c', xr.concat), ('d', xr.DataArray.mean), ('e', xr.concat))
        >>>
        >>> # Task a and b take band name arguments
        >>> # Tasks c, d, and e take the coordinate dimension name as an argument
        >>> func_args = {'a': {'b1': 'green', 'b2': 'red'}, 'b': {'b1': 'blue', 'b2': 'green'}, 'c': {'dim': 'band'}, 'd': {'dim': 'band'}, 'e': {'dim': 'band'}}
        >>> open_args = {'chunks': 512}
        >>> config_args = {'sensor': 'bgr', 'nodata': 0, 'scale_factor': 0.0001}
        >>>
        >>> # Setup a task
        >>> task_mean = GeoTask(inputs, outputs, tasks, config_args=config_args, open_args=open_args, func_args=func_args)
        >>>
        >>> # Visualize the task
        >>> task_mean.visualize()
        >>>
        >>> # Create a task that takes the output of task e and writes the mean to file
        >>> task_write = GeoTask({'f': 'e'}, {'f': 'mean.tif'}, (('f', xr.DataArray.mean),),
        >>>                      config_args=config_args,
        >>>                      func_args={'f': {'dim': 'band'}},
        >>>                      open_args=open_args,
        >>>                      out_args={'compress': 'lzw', 'overwrite': True})
        >>>
        >>> # Add the new task
        >>> new_task = task_mean + task_write
        >>>
        >>> new_task.visualize()
        >>>
        >>> # Write the task pipeline to file
        >>> new_task.submit()
    """

    def __init__(self,
                 inputs,
                 outputs,
                 tasks,
                 clean=None,
                 config_args=None,
                 open_args=None,
                 func_args=None,
                 out_args=None,
                 log_file=None):

        super().__init__(inputs,
                         outputs,
                         tasks,
                         clean=clean,
                         config_args=config_args,
                         open_args=open_args,
                         func_args=func_args,
                         out_args=out_args,
                         log_file=log_file)

    def execute(self, task_id, task, src, task_results, attrs, **kwargs):

        """
        Executes an individual task

        Args:
            task_id (str)
            task (func)
            src (DataArray | list)
            task_results (dict)
            attrs (dict)
            kwargs (Optional[dict])
        """

        # Execute the task
        if isinstance(src, tuple):
            res = task((task_results[i] for i in src), **kwargs)
        else:
            res = task(src, **kwargs)

        if not hasattr(res, 'band'):
            res = res.expand_dims(dim='band').assign_coords({'band': ['res']})

        # Write to file
        if task_id in self.outputs:
            if self.outputs[task_id].lower().endswith('.tif'):
                if not hasattr(res, 'crs'):
                    res.attrs = attrs
                res.gw.to_raster(self.outputs[task_id], **self.out_args)

        return res

    def submit(self):

        """
        Submits a pipeline task
        """

        task_results = {}
        attrs = None
        res = None

        with gw_config.update(**self.config_args):

            counter = 0

            for task_id, task in self.tasks:

                # Check task keywords
                kwargs = self.func_args[task_id] if task_id in self.func_args else {}

                # Check task input(s)
                if isinstance(self.inputs[task_id], tuple) or isinstance(self.inputs[task_id], list):

                    with ExitStack() as stack:

                        # Open input files for the task
                        src = (stack.enter_context(gw_open(fn, **self.open_args)) if Path(fn).is_file()
                               else task_results[fn] for fn in self.inputs[task_id])

                        res = self.execute(task_id, task, src, task_results, attrs, **kwargs)

                    # res = self.execute(task_id, task, self.inputs[task_id], task_results, attrs, **kwargs)

                elif isinstance(self.inputs[task_id], str) and not Path(self.inputs[task_id]).is_file():
                    res = self.execute(task_id, task, task_results[self.inputs[task_id]], task_results, attrs, **kwargs)
                elif isinstance(self.inputs[task_id], str) and Path(self.inputs[task_id]).is_file():

                    with gw_open(self.inputs[task_id], **self.open_args) as src:
                        attrs = src.attrs.copy()
                        res = self.execute(task_id, task, src, task_results, attrs, **kwargs)

                task_results[task_id] = res

                self._log_task(task_id)

                # if counter > 0:
                #     self._cleanup('task', self.tasks[counter-1][[0]])

                counter += 1

        # for task_id, __ in self.tasks:
        #     self._cleanup('pipeline', task_id)

        return res


# class IndicesStack(object):
#
#     """
#     A class for stacking spectral indices
#
#     Args:
#         processes (tuple): The spectral indices to process.
#
#     Returns:
#         ``xarray.DataArray``
#
#     Example:
#         >>> import geowombat as gw
#         >>> from geowombat.tasks import IndicesStack
#         >>>
#         >>> task = IndicesStack(('avi', 'evi2', 'evi', 'nbr', 'ndvi', 'tasseled_cap'))
#         >>>
#         >>> with gw.open('image.tif') as src:
#         >>>     res = task.submit(src, scale_factor=0.0001)
#     """
#
#     def __init__(self, processes):
#
#         self.processes = processes
#         self._validate_methods()
#
#     def _validate_methods(self):
#
#         args = [gw] * len(self.processes)
#
#         for object_, proc_ in zip(*args, self.processes):
#
#             if not hasattr(object_, proc_):
#                 raise NameError(f'The {proc_} process is not supported.')
#
#     def submit(self, data, *args, **kwargs):
#
#         attrs = data.attrs.copy()
#         results = []
#
#         for vi in self.processes:
#
#             vi_func = getattr(gw, vi)
#             results.append(vi_func(data, *args, **kwargs))
#
#         results = xr.concat(results, dim='band').astype('float64')
#
#         return results.assign_attrs(**attrs)
