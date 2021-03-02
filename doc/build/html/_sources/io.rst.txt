.. _io:

Raster I/O
==========

File opening with GeoWombat uses the :func:`geowombat.open` function to open raster files.

.. ipython:: python

    # Import GeoWombat
    import geowombat as gw

    # Load image names
    from geowombat.data import l8_224077_20200518_B2, l8_224077_20200518_B3, l8_224077_20200518_B4
    from geowombat.data import l8_224078_20200518_B2, l8_224078_20200518_B3, l8_224078_20200518_B4, l8_224078_20200518

    from pathlib import Path
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe

To open individual images, GeoWombat uses :func:`xarray.open_rasterio` and :func:`rasterio.vrt.WarpedVRT`.

.. ipython:: python

    fig, ax = plt.subplots(dpi=200)
    with gw.open(l8_224078_20200518) as src:
        src.where(src != 0).sel(band=[3, 2, 1]).gw.imshow(robust=True, ax=ax)
    @savefig rgb_plot.png
    plt.tight_layout(pad=1)

Open a raster as a DataArray.

.. ipython:: python

    with gw.open(l8_224078_20200518) as src:
        print(src)

Force the output data type.

.. ipython:: python

    with gw.open(l8_224078_20200518, dtype='float64') as src:
        print(src.dtype)

Specify band names.

.. ipython:: python

    with gw.open(l8_224078_20200518, band_names=['blue', 'green', 'red']) as src:
        print(src.band)

Use the sensor name to set band names.

.. ipython:: python

    with gw.config.update(sensor='bgr'):
        with gw.open(l8_224078_20200518) as src:
            print(src.band)

To open multiple images stacked by bands, use a list of files with ``stack_dim='band'``.

Open a list of files as a DataArray, with all bands stacked.

.. ipython:: python

    with gw.open([l8_224078_20200518_B2, l8_224078_20200518_B3, l8_224078_20200518_B4],
                 band_names=['b', 'g', 'r'],
                 stack_dim='band') as src:
        print(src)

To open multiple images as a time stack, change the input to a list of files.

Open a list of files as a DataArray.

.. ipython:: python

    with gw.open([l8_224078_20200518, l8_224078_20200518],
                 band_names=['blue', 'green', 'red'],
                 time_names=['t1', 't2']) as src:
        print(src)

.. note::

    Xarray will handle alignment of images of varying sizes as long as the the resolutions are "target aligned". If images are not target aligned, Xarray might not concatenate a stack of images. With GeoWombat, we can use a context manager and a reference image to handle image alignment.

In the example below, we specify a reference image using GeoWombat's configuration manager.

.. note::

    The two images in this example are identical. The point here is just to illustrate the use of the configuration manager.

.. ipython:: python

    # Use an image as a reference for grid alignment and CRS-handling
    #
    # Within the configuration context, every image
    # in concat_list will conform to the reference grid.
    filenames = [l8_224078_20200518, l8_224078_20200518]
    with gw.config.update(ref_image=l8_224077_20200518_B2):
        with gw.open(filenames,
                     band_names=['blue', 'green', 'red'],
                     time_names=['t1', 't2']) as src:
            print(src)

    with gw.config.update(ref_image=l8_224078_20200518_B2):
        with gw.open(filenames,
                     band_names=['blue', 'green', 'red'],
                     time_names=['t1', 't2']) as src:
            print(src)

Stack the intersection of all images.

.. ipython:: python

    fig, ax = plt.subplots(dpi=200)
    filenames = [l8_224077_20200518_B2, l8_224078_20200518_B2]
    with gw.open(filenames,
                 band_names=['blue'],
                 mosaic=True,
                 bounds_by='intersection') as src:
        src.where(src != 0).sel(band='blue').gw.imshow(robust=True, ax=ax)
    @savefig blue_intersection_plot.png
    plt.tight_layout(pad=1)

Stack the union of all images.

.. ipython:: python

    fig, ax = plt.subplots(dpi=200)
    filenames = [l8_224077_20200518_B2, l8_224078_20200518_B2]
    with gw.open(filenames,
                 band_names=['blue'],
                 mosaic=True,
                 bounds_by='union') as src:
        src.where(src != 0).sel(band='blue').gw.imshow(robust=True, ax=ax)
    @savefig blue_union_plot.png
    plt.tight_layout(pad=1)

When multiple images have matching dates, the arrays are merged into one layer.

.. ipython:: python

    filenames = [l8_224077_20200518_B2, l8_224078_20200518_B2]
    band_names = ['blue']
    time_names = ['t1', 't1']
    with gw.open(filenames, band_names=band_names, time_names=time_names) as src:
        print(src)

Image mosaicking
----------------

Mosaic the two subsets into a single DataArray. If the images in the mosaic list have the same CRS, no configuration
is needed.

.. ipython:: python

    with gw.open([l8_224077_20200518_B2, l8_224078_20200518_B2],
                 band_names=['b'],
                 mosaic=True) as src:
        print(src)

If the images in the mosaic list have different CRSs, use a context manager to warp to a common grid.

.. note::

    The two images in this example have the same CRS. The point here is just to illustrate the use of the configuration manager.

.. ipython:: python

    # Use a reference CRS
    with gw.config.update(ref_image=l8_224077_20200518_B2):
        with gw.open([l8_224077_20200518_B2, l8_224078_20200518_B2],
                     band_names=['b'],
                     mosaic=True) as src:
            print(src)

Setup a plot function
~~~~~~~~~~~~~~~~~~~~~

.. ipython:: python

    def plot(bounds_by, ref_image=None, cmap='viridis'):
        fig, ax = plt.subplots(dpi=200)
        with gw.config.update(ref_image=ref_image):
            with gw.open([l8_224077_20200518_B4, l8_224078_20200518_B4],
                         band_names=['nir'],
                         chunks=256,
                         mosaic=True,
                         bounds_by=bounds_by) as srca:
                # Plot the NIR band
                srca.where(srca != 0).sel(band='nir').gw.imshow(robust=True, cbar_kwargs={'label': 'DN'}, ax=ax)
                # Plot the image chunks
                srca.gw.chunk_grid.plot(color='none', edgecolor='k', ls='-', lw=0.5, ax=ax)
                # Plot the image footprints
                srca.gw.footprint_grid.plot(color='none', edgecolor='orange', lw=2, ax=ax)
                # Label the image footprints
                for row in srca.gw.footprint_grid.itertuples(index=False):
                    ax.scatter(row.geometry.centroid.x, row.geometry.centroid.y,
                               s=50, color='red', edgecolor='white', lw=1)
                    ax.annotate(row.footprint.replace('.TIF', ''),
                                (row.geometry.centroid.x, row.geometry.centroid.y),
                                color='black',
                                size=8,
                                ha='center',
                                va='center',
                                path_effects=[pe.withStroke(linewidth=1, foreground='white')])
                # Set the display bounds
                ax.set_ylim(srca.gw.footprint_grid.total_bounds[1]-10, srca.gw.footprint_grid.total_bounds[3]+10)
                ax.set_xlim(srca.gw.footprint_grid.total_bounds[0]-10, srca.gw.footprint_grid.total_bounds[2]+10)
        title = f'Image {bounds_by}' if bounds_by else str(Path(ref_image).name.split('.')[0]) + ' as reference'
        size = 12 if bounds_by else 8
        ax.set_title(title, size=size)
        plt.tight_layout(pad=1)

Mosaic by the union of images
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The two plots below illustrate how two images can be mosaicked. The orange grids highlight the image
footprints while the black grids illustrate the ``DataArray`` chunks.

.. ipython:: python

    @savefig union_example.png
    plot('union')

.. ipython:: python

    @savefig intersection_example.png
    plot('intersection')

.. ipython:: python

    @savefig ref1_example.png
    plot(None, l8_224077_20200518_B4)

.. ipython:: python

    @savefig ref2_example.png
    plot(None, l8_224078_20200518_B4)

Writing DataArrays to file
--------------------------

GeoWombat's I/O can be accessed through the :func:`to_vrt` and :func:`to_raster` functions. These functions use
Rasterio's :func:`write` and Dask.array :func:`store` functions as I/O backends. In the examples below,
``src`` is an ``xarray.DataArray`` with the necessary transform information to write to an image file.

Write to a VRT file.

.. code:: python

    import geowombat as gw

    # Transform the data to lat/lon
    with gw.config.update(ref_crs=4326):

        with gw.open(l8_224077_20200518_B4, chunks=1024) as src:

            # Write the data to a VRT
            src.gw.to_vrt('lat_lon_file.vrt')

Write to a raster file.

.. code:: python

    import geowombat as gw

    with gw.open(l8_224077_20200518_B4, chunks=1024) as src:

        # Xarray drops attributes
        attrs = src.attrs.copy()

        # Apply operations on the DataArray
        src = src * 10.0

        src.attrs = attrs

        # Write the data to a GeoTiff
        src.gw.to_raster('output.tif',
                         verbose=1,
                         n_workers=4,    # number of process workers sent to ``concurrent.futures``
                         n_threads=2,    # number of thread workers sent to ``dask.compute``
                         n_chunks=200)   # number of window chunks to send as concurrent futures

See :ref:`io-distributed` for more examples describing concurrent file writing with GeoWombat.
