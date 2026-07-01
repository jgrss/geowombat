.. _band_math:

Band math
=========

Band math combines two or more spectral bands into a single value per pixel that
highlights a physical property of the surface, such as vegetation vigor, water
content, or burn severity. GeoWombat exposes these operations through the
``.gw`` accessor, so any :class:`xarray.DataArray` opened with :func:`geowombat.open`
can compute an index directly, returning a new lazy :class:`xarray.DataArray`.

Most indices are defined on surface reflectance, so the examples below set a
``scale_factor`` to convert the stored digital numbers to reflectance. Passing
the ``sensor`` name (here ``'rgbn'``) lets GeoWombat map band positions to named
bands (red, nir, ...) automatically. Both can be supplied per call or, more
conveniently, through the configuration manager (see :ref:`tutorial-config`).

Vegetation indices
------------------

Vegetation indices contrast the strong absorption of red light by chlorophyll
with the strong reflectance of near-infrared light by leaf structure. Higher
values generally indicate denser, healthier vegetation.

Enhanced Vegetation Index (EVI)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

EVI improves on the classic NDVI by correcting for atmospheric and soil-background
effects, which makes it more reliable over dense canopies where NDVI saturates.

.. ipython:: python

    import geowombat as gw
    from geowombat.data import rgbn

Calculate a vegetation index, returning an :class:`xarray.DataArray`.

.. ipython:: python

    with gw.open(rgbn) as ds:
        print(ds)
        evi = ds.gw.evi(sensor='rgbn', scale_factor=0.0001)
        print(evi)

Use the configuration context to set parameters, so the ``sensor`` and
``scale_factor`` do not need to be repeated on every call.

.. ipython:: python

    with gw.config.update(sensor='rgbn', scale_factor=0.0001):
        with gw.open(rgbn) as ds:
            evi = ds.gw.evi()
            print(evi)

Two-band Enhanced Vegetation Index (EVI2)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

EVI2 is a two-band approximation of EVI that uses only the red and near-infrared
bands. It is useful for sensors that lack a blue band, or when the blue band is
noisy.

.. ipython:: python

    with gw.config.update(sensor='rgbn', scale_factor=0.0001):
        with gw.open(rgbn) as ds:
            evi2 = ds.gw.evi2()
            print(evi2)

Normalized difference indices
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Many indices share the normalized-difference form ``(b1 - b2) / (b1 + b2)``. Use
the generic :func:`xarray.DataArray.gw.norm_diff` function with any two-band
combination to build one on the fly. For example, a red/near-infrared pair
reproduces NDVI.

.. ipython:: python

    with gw.config.update(sensor='rgbn'):
        with gw.open(rgbn) as ds:
            d = ds.gw.norm_diff('red', 'nir')
            print(d)

Tasseled cap transformations
----------------------------

The tasseled cap transformation rotates the spectral bands into components that
correspond to physical scene characteristics, most commonly brightness,
greenness, and wetness. Because the rotation coefficients are sensor-specific,
supply the ``sensor`` name so GeoWombat applies the correct set.

.. ipython:: python

    with gw.config.update(sensor='qb', scale_factor=0.0001):
        with gw.open(rgbn) as ds:
            tcap = ds.gw.tasseled_cap()
            print(tcap)

Additional useful indices are available, such as the normalized burn ratio (NBR),
which measures fire severity, and the woody index (WI). For a full list of
indices and their expected bands, see the `API docs <https://geowombat.readthedocs.io/en/latest/api.html>`_.
