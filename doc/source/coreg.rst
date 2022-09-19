.. _coreg:

Co-registration
===============

Image co-registration is the ...

AROSICS method
--------------

The AROSICS (An Automated and Robust Open-Source Image Co-Registration Software for Multi-Sensor
Satellite Data) co-registration method was developed `by Scheffler et al. (2017) <https://www.mdpi.com/2072-4292/9/7/676>`_.
The authors made their algorithm open to the public in a Python package called
`AROSICS <https://danschef.git-pages.gfz-potsdam.de/arosics/doc/>`_.

In GeoWombat, we provide an interface to the AROSICS package. The co-registration method is called :func:`geowombat.coregister`,
and mainly acts as an interface between ``DataArrays`` and AROSICS.

Install AROSICS
---------------

The AROSICS package can either be installed separately from geowombat by following the
`installation instructions <https://danschef.git-pages.gfz-potsdam.de/arosics/doc/installation.html>`_ or by installing
with geowombat. For the latter, simply add the ``coreg`` extra when installing geowombat by::

    pip install geowombat[coreg]@git+https://github.com/jgrss/geowombat.git

Co-register an image
--------------------

To co-register an image using a secondary reference, open both as ``DataArrays`` and pass them
to the :func:`coregister` method.

.. note::

    If AROSICS is unable to adjust the target data, either because of insufficient valid data or because
    the data are not offset to each other, the returned ``DataArray`` will match the input target data.

.. code:: python

    import geowombat as gw
    from geowombat.data import l8_224077_20200518_B2
    from geowombat.data import l8_224077_20200518_B4

    with gw.open(l8_224077_20200518_B2) as target, gw.open(
        l8_224077_20200518_B4
    ) as reference:
        target_shifted = gw.coregister(
            target=target,
            reference=reference,
            ws=(256, 256),
            r_b4match=1,
            s_b4match=1,
            max_shift=5,
            resamp_alg_deshift='nearest',
            resamp_alg_calc='cubic',
            out_gsd=[target.gw.celly, reference.gw.celly],
            q=True,
            nodata=(0, 0),
            CPUs=1,
        )
