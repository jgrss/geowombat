.. _radiometrics:

BRDF normalization
==================

`GeoWombat` uses the global c-factor method to apply BRDF normalization on surface reflectance data
---------------------------------------------------------------------------------------------------

In the example below, we use :func:`norm_brdf` to normalize a Landsat 8 OLI TIRS surface reflectance image.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    import geowombat as gw

    # Landsat 8 surface reflectance image
    l8_image = 'oli_tirs_sr.tif'

    # Pixel angle images
    solar_za = 'solar_za.tif'
    solar_az = 'solar_az.tif'
    sensor_za = 'sensor_za.tif'
    sensor_az = 'sensor_az.tif'

    # Set global parameters
    with gw.config.update(sensor='l8', scale_factor=0.0001):

        # Open the surface reflectance file
        with gw.open(l8_image, chunks=512) as ds:

            # Open the pixel angle files
            with gw.open(solar_za, chunks=512) as sza,
                gw.open(solar_za, chunks=512) as saz,
                    gw.open(solar_za, chunks=512) as ssza,
                        gw.open(solar_za, chunks=512) as ssaz:

                # Normalize the surface reflectance
                sr_brdf = ds.gw.norm_brdf(sza, saz, ssza, ssaz)

                # Save the results to file
                sr_brdf.gw.to_raster('oli_tirs_sr_brdf.tif')
