.. _changelog:

Change Log
==========

1.0.0b (10 November 2019)
-------------------------

New
~~~

- Added :func:`download_cube` function.

1.0.0b (1 November 2019)
------------------------

Enhancements
~~~~~~~~~~~~

- Added `expand_by` user argument in :func:`geowombat.clip`.

1.0.0b (30 October 2019)
------------------------

New
~~~

- Added user functions as Xarray attributes. See :ref:`apply` for an example.

1.0.0b (24 October 2019)
------------------------

Enhancements
~~~~~~~~~~~~

- Implemented improvements from testing processes vs. threads for concurrent I/O in :func:`geowombat.to_raster`.

Bug fixes
~~~~~~~~~

- Changed BRDF normalization (:class:`geowombat.radiometry.BRDF`) from 1d to 2d in order to work with Dask arrays.

1.0.0b (23 October 2019)
------------------------

Changes
~~~~~~~

- Added fixes for surface reflectance

New
~~~

- Added support for band stacking (in addition to time stacking) in :func:`geowombat.open`. The new keyword argument is `stack_dim` and can be used like:

.. code:: python

    with gw.open(..., stack_dim='band') as ds:
        ...

1.0.0b (20 October 2019)
------------------------

Changes
~~~~~~~

- Block writing can now be done with `concurrent.futures` or with `dask.store`.

New
~~~

- Added automatic date parsing when concatenating a list of files.
- Added BRDF normalization using the c-factor method.

1.0.0a
------

History
~~~~~~~

- Examined concurrent writing workflows.
- Setup basic geo-spatial functionality.
