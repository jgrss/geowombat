.. _tasks:

Pipeline tasks
==============

.. ipython:: python

    import geowombat as gw
    from geowombat.tasks import GeoTask
    from geowombat.data import l8_224078_20200518_B3, l8_224078_20200518_B4, l8_224078_20200518

    import xarray as xr
    import matplotlib.pyplot as plt

.. ipython:: python

    inputs = {'a': l8_224078_20200518, 'b': l8_224078_20200518, 'c': (l8_224078_20200518_B3, l8_224078_20200518_B4), 'd': 'c', 'e': ('a', 'b', 'd')}
    outputs = {'a': 'mem|r1', 'b': 'mem|r2', 'c': 'mem|r3', 'd': 'mem|mean', 'e': 'mem|stack'}
    tasks = (('a', gw.norm_diff), ('b', gw.norm_diff), ('c', xr.concat), ('d', xr.DataArray.mean), ('e', xr.concat))
    func_args = {'a': {'b1': 'green', 'b2': 'red'}, 'b': {'b1': 'blue', 'b2': 'green'}, 'c': {'dim': 'band'}, 'd': {'dim': 'band'}, 'e': {'dim': 'band'}}
    open_args = {'chunks': 512}
    config_args = {'sensor': 'bgr', 'nodata': 0, 'scale_factor': 0.0001}

    task_mean = GeoTask(inputs, outputs, tasks, config_args=config_args, open_args=open_args, func_args=func_args, out_args={})
    viz = task_mean.visualize()

    viz.format = 'PNG'
    viz.render(filename='task_mean', view=False)

    fig, ax = plt.subplots(dpi=200)
    @savefig task_mean.png
    plt.tight_layout(pad=1)
