.. _moving:

Moving windows
==============

Examine the :func:`geowombat.moving` help.

.. ipython:: python

    import geowombat as gw

    print(help(gw.moving))

Calculate the local average within a 5x5 pixel window.

.. code:: python

    import geowombat as gw
    from geowombat.data import rgbn

    with gw.open(rgbn, chunks=512) as src:

        res = src.gw.moving(stat='mean', w=5, nodata=0)

        # Compute results
        res.data.compute(num_workers=4)
        # or save to file
        # res.gw.save('output.tif', num_workers=4)
