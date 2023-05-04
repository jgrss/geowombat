.. _tasks:

Pipeline tasks
==============

Setup a task and visualize the steps
------------------------------------

.. code:: python

    import geowombat as gw
    from geowombat.tasks import GeoTask
    from geowombat.data import l8_224078_20200518_B3, l8_224078_20200518_B4, l8_224078_20200518

    import xarray as xr

    # Setup the input steps
    inputs = {
        'a': l8_224078_20200518,
        'b': l8_224078_20200518,
        'c': (l8_224078_20200518_B3, l8_224078_20200518_B4),
        'd': 'c',               # the input of 'd' is the output of 'c'
        'e': ('a', 'b', 'd')    # the input of 'e' is the output of 'a', 'b', and 'c'
    }

    # Setup the outputs of each step
    # Here, we could specify filenames to write or to process in-memory
    outputs = {
        'a': 'mem|r1',
        'b': 'mem|r2',
        'c': 'mem|r3',
        'd': 'mem|mean',
        'e': 'mem|stack'
    }

    # Setup the tasks to execute on each input step
    tasks = (
        ('a', gw.norm_diff),
        ('b', gw.norm_diff),
        ('c', xr.concat),
        ('d', xr.DataArray.mean),
        ('e', xr.concat)
    )

    # Setup the function keyword arguments of each step
    func_args = {
        'a': {'b1': 'green', 'b2': 'red'},
        'b': {'b1': 'blue', 'b2': 'green'},
        'c': {'dim': 'band'},
        'd': {'dim': 'band'},
        'e': {'dim': 'band'}
    }

    open_args = {'chunks': 512}
    config_args = {'sensor': 'bgr', 'nodata': 0, 'scale_factor': 0.0001}
    out_args = {}

    # Setup the task object
    task_mean = GeoTask(
        inputs,
        outputs,
        tasks,
        config_args=config_args,
        open_args=open_args,
        func_args=func_args,
        out_args=out_args
    )

    # Visualize the steps
    task_mean.visualize()

.. image:: _static/task_mean.png

Create a new task and add to initial task
-----------------------------------------

.. code:: python

    inputs = {'f': 'e'}         # 'f' takes the output of step 'e' from our previous task
    outputs = {'f': 'mean.tif'}
    tasks = (('f', xr.DataArray.mean),)
    func_args = {'f': {'dim': 'band'}}
    out_args = {'compress': 'lzw', 'overwrite': True}

    # Create a task that takes the output of task e and writes the mean to file
    task_write = GeoTask(
        inputs,
        outputs,
        tasks,
        config_args=config_args,
        func_args=func_args,
        open_args=open_args,
        out_args=out_args
    )

    # Add the new task
    new_task = task_mean + task_write

    new_task.visualize()

.. image:: _static/task_write.png

Execute the task pipeline
-------------------------

.. code:: python

    new_task.submit()

