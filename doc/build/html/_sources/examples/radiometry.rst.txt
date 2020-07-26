.. _radiometry:

Radiometry
==========

`GeoWombat` uses the global c-factor method to apply BRDF normalization on surface reflectance data
---------------------------------------------------------------------------------------------------

In the example below, we use :func:`norm_brdf` to normalize a Landsat 8 surface reflectance image.

.. code:: python

    import geowombat as gw
    from geowombat.radiometry import RadTransforms
    from geowombat.radiometry import BRDF

    # Pixel angle images
    solar_za = 'solar_za.tif'
    solar_az = 'solar_az.tif'
    sensor_za = 'sensor_za.tif'
    sensor_az = 'sensor_az.tif'

    # Metadata file
    metadata = 'LC08_L1TP_042034_20160121_20170224_01_T1_MTL.txt'

    sr = RadTransforms()
    brdf = BRDF()

    # Set global parameters
    with gw.config.update(sensor='l8'):

        # Open the surface reflectance files
        with gw.open(['LC08_L1TP_042034_20160121_20170224_01_T1_B4.TIF',
                      'LC08_L1TP_042034_20160121_20170224_01_T1_B5.TIF',
                      'LC08_L1TP_042034_20160121_20170224_01_T1_B6.TIF'],
                      stack_dim='band',
                      chunks=512) as dn:

            # Open the pixel angle files
            with gw.open(solar_za, chunks=512) as sza,
                gw.open(solar_az, chunks=512) as saz,
                    gw.open(sensor_za, chunks=512) as vza,
                        gw.open(sensor_az, chunks=512) as vaz:

                # DN --> surface reflectance
                sr_data = sr.dn_to_sr(dn, solar_za, sensor_za, meta=metadata)

                # Normalize the surface reflectance
                brdf_data = brdf.norm_brdf(sr_data, sza, saz, vza, vaz, wavelengths=dn.band.values.tolist())

                # Save the results to file
                brdf_data.gw.to_raster('l8_sr_brdf.tif')
