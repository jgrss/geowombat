.. _changelog:

Change Log
==========

1.0.0b (23 October 2019)
------------------------

Changes
~~~~~~~

- Added fixes for surface reflectance

New
~~~

- Added support for band stacking (in addition to time stacking) in :func:`geowombat.open`.

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
