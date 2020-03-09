#!/usr/bin/env python

import os
import fnmatch
from pathlib import Path
import argparse

from geowombat.util import GeoDownloads

import geopandas as gpd
from shapely.geometry import Polygon, Point
import pyproj


PROJ_DICT = dict(saaeac=dict(wkt='PROJCS["South_America_Albers_Equal_Area_Conic",GEOGCS["GCS_South_American_1969",DATUM["D_South_American_1969",SPHEROID["GRS_1967_Truncated",6378160,298.25]],PRIMEM["Greenwich",0],UNIT["Degree",0.017453292519943295]],PROJECTION["Albers"],PARAMETER["False_Easting",0],PARAMETER["False_Northing",0],PARAMETER["central_meridian",-60],PARAMETER["Standard_Parallel_1",-5],PARAMETER["Standard_Parallel_2",-42],PARAMETER["latitude_of_origin",-32],UNIT["Meter",1]]',
                             proj4='+proj=aea +lat_1=-5 +lat_2=-42 +lat_0=-32 +lon_0=-60 +x_0=0 +y_0=0 +ellps=aust_SA +units=m +no_defs',
                             epsg=102033),
                 naaeac=dict(wkt='PROJCS["North_America_Albers_Equal_Area_Conic",GEOGCS["GCS_North_American_1983",DATUM["D_North_American_1983",SPHEROID["GRS_1980",6378137,298.257222101]],PRIMEM["Greenwich",0],UNIT["Degree",0.017453292519943295]],PROJECTION["Albers"],PARAMETER["False_Easting",0],PARAMETER["False_Northing",0],PARAMETER["central_meridian",-96],PARAMETER["Standard_Parallel_1",20],PARAMETER["Standard_Parallel_2",60],PARAMETER["latitude_of_origin",40],UNIT["Meter",1]]',
                             proj4='+proj=aea +lat_1=20 +lat_2=60 +lat_0=40 +lon_0=-96 +x_0=0 +y_0=0 +datum=NAD83 +units=m +no_defs',
                             epsg=102008),
                 euaeac=dict(wkt='PROJCS["Europe_Albers_Equal_Area_Conic",GEOGCS["GCS_European_1950",DATUM["D_European_1950",SPHEROID["International_1924",6378388,297]],PRIMEM["Greenwich",0],UNIT["Degree",0.017453292519943295]],PROJECTION["Albers"],PARAMETER["False_Easting",0],PARAMETER["False_Northing",0],PARAMETER["central_meridian",10],PARAMETER["Standard_Parallel_1",43],PARAMETER["Standard_Parallel_2",62],PARAMETER["latitude_of_origin",30],UNIT["Meter",1]]',
                             proj4='+proj=aea +lat_1=43 +lat_2=62 +lat_0=30 +lon_0=10 +x_0=0 +y_0=0 +ellps=intl +units=m +no_defs',
                             epsg=102013),
                 afaeac=dict(wkt='PROJCS["Africa_Albers_Equal_Area_Conic",GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137,298.257223563]],PRIMEM["Greenwich",0],UNIT["Degree",0.017453292519943295]],PROJECTION["Albers"],PARAMETER["False_Easting",0],PARAMETER["False_Northing",0],PARAMETER["central_meridian",25],PARAMETER["Standard_Parallel_1",20],PARAMETER["Standard_Parallel_2",-23],PARAMETER["latitude_of_origin",0],UNIT["Meter",1]]',
                             proj4='+proj=aea +lat_1=20 +lat_2=-23 +lat_0=0 +lon_0=25 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs',
                             epsg=102022),
                 anaeac=dict(wkt='PROJCS["Asia_North_Albers_Equal_Area_Conic",GEOGCS["GCS_WGS_1984",DATUM["WGS_1984",SPHEROID["WGS_1984",6378137,298.257223563]],PRIMEM["Greenwich",0],UNIT["Degree",0.017453292519943295]],PROJECTION["Albers_Conic_Equal_Area"],PARAMETER["False_Easting",0],PARAMETER["False_Northing",0],PARAMETER["longitude_of_center",95],PARAMETER["Standard_Parallel_1",15],PARAMETER["Standard_Parallel_2",65],PARAMETER["latitude_of_center",30],UNIT["Meter",1],AUTHORITY["EPSG","102025"]]',
                             proj4='+proj=aea +lat_1=15 +lat_2=65 +lat_0=30 +lon_0=95 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs',
                             epsg=102025),
                 asaeac=dict(wkt='PROJCS["Asia_South_Albers_Equal_Area_Conic",GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137,298.257223563]],PRIMEM["Greenwich",0],UNIT["Degree",0.017453292519943295]],PROJECTION["Albers"],PARAMETER["False_Easting",0],PARAMETER["False_Northing",0],PARAMETER["central_meridian",125],PARAMETER["Standard_Parallel_1",7],PARAMETER["Standard_Parallel_2",-32],PARAMETER["latitude_of_origin",-15],UNIT["Meter",1]]',
                             proj4='+proj=aea +lat_1=7 +lat_2=-32 +lat_0=-15 +lon_0=125 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs',
                             epsg=102028),
                 auaeac=dict(wkt='PROJCS["GDA94 / Australian Albers",GEOGCS["GDA94",DATUM["Geocentric_Datum_of_Australia_1994",SPHEROID["GRS 1980",6378137,298.257222101,AUTHORITY["EPSG","7019"]],TOWGS84[0,0,0,0,0,0,0],AUTHORITY["EPSG","6283"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4283"]],PROJECTION["Albers_Conic_Equal_Area"],PARAMETER["standard_parallel_1",-18],PARAMETER["standard_parallel_2",-36],PARAMETER["latitude_of_center",0],PARAMETER["longitude_of_center",132],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH],AUTHORITY["EPSG","3577"]]',
                             proj4='+proj=aea +lat_1=-18 +lat_2=-36 +lat_0=0 +lon_0=132 +x_0=0 +y_0=0 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs',
                             epsg=3577),
                 arabia=dict(wkt='PROJCS["Ain el Abd / Aramco Lambert",GEOGCS["Ain el Abd",DATUM["Ain_el_Abd_1970",SPHEROID["International 1924",6378388,297,AUTHORITY["EPSG","7022"]],TOWGS84[-143,-236,7,0,0,0,0],AUTHORITY["EPSG","6204"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4204"]],PROJECTION["Lambert_Conformal_Conic_2SP"],PARAMETER["standard_parallel_1",17],PARAMETER["standard_parallel_2",33],PARAMETER["latitude_of_origin",25.08951],PARAMETER["central_meridian",48],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH],AUTHORITY["EPSG","2318"]]',
                             proj4='+proj=lcc +lat_1=17 +lat_2=33 +lat_0=25.08951 +lon_0=48 +x_0=0 +y_0=0 +ellps=intl +towgs84=-143,-236,7,0,0,0,0 +units=m +no_def',
                             epsg=2318),
                 turkey=dict(wkt='PROJCS["ETRS89_LAEA_Europe",GEOGCS["GCS_ETRS_1989",DATUM["European_Terrestrial_Reference_System_1989",SPHEROID["GRS_1980",6378137,298.257222101,AUTHORITY["EPSG","7019"]],AUTHORITY["EPSG","6258"]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]],PROJECTION["Lambert_Azimuthal_Equal_Area"],PARAMETER["latitude_of_center",52],PARAMETER["longitude_of_center",10],PARAMETER["false_easting",4321000],PARAMETER["false_northing",3210000],UNIT["metre",1,AUTHORITY["EPSG","9001"]]]',
                             proj4='+proj=laea +lat_0=52 +lon_0=10 +x_0=4321000 +y_0=3210000 +ellps=GRS80 +units=m +no_defs'),
                 equal_earth=dict(proj4='+proj=eqearth +lon_0=0 +x_0=0 +y_0=0 +ellps=GRS80 +units=m +datum=WGS84 +wktext'),
                 wgs84=dict(wkt='GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]]',
                            proj4='+proj=longlat +datum=WGS84 +no_defs',
                            epsg=4326))

PROJ4_DICT = {'North America': '+proj=aea +lat_1=20 +lat_2=60 +lat_0=40 +lon_0=-96 +x_0=0 +y_0=0 +datum=NAD83 +units=m +no_defs',
              'Europe': '+proj=aea +lat_1=43 +lat_2=62 +lat_0=30 +lon_0=10 +x_0=0 +y_0=0 +ellps=intl +units=m +no_defs',
              'Africa': '+proj=aea +lat_1=20 +lat_2=-23 +lat_0=0 +lon_0=25 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs',
              'Asia North': '+proj=aea +lat_1=15 +lat_2=65 +lat_0=30 +lon_0=95 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs',
              'Asia South': '+proj=aea +lat_1=7 +lat_2=-32 +lat_0=-15 +lon_0=125 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs',
              'Australia': '+proj=aea +lat_1=-18 +lat_2=-36 +lat_0=0 +lon_0=132 +x_0=0 +y_0=0 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs',
              'South America': '+proj=aea +lat_1=-5 +lat_2=-42 +lat_0=-32 +lon_0=-60 +x_0=0 +y_0=0 +ellps=aust_SA +units=m +no_defs',
              'Russia': '+proj=aea +lat_1=50 +lat_2=70 +lat_0=56 +lon_0=100 +x_0=0 +y_0=0 +ellps=WGS84 +datum=WGS84 +units=m +no_defs'}

_projections = {}


def get_utm_zone(coordinates):

    if 56 <= coordinates[1] < 64 and 3 <= coordinates[0] < 12:
        return 32

    if 72 <= coordinates[1] < 84 and 0 <= coordinates[0] < 42:

        if coordinates[0] < 9:
            return 31
        elif coordinates[0] < 21:
            return 33
        elif coordinates[0] < 33:
            return 35

        return 37

    return int((coordinates[0] + 180.0) / 6.0) + 1


def get_utm_letter(coordinates):
    return 'CDEFGHJKLMNPQRSTUVWXX'[int((coordinates[1] + 80.0) / 8.0)]


def zone_to_epsg(lat, utm_zone):
    return int('326{:d}'.format(utm_zone)) if lat > 0 else int('327{:d}'.format(utm_zone))


def latlon_to_utm(x, y):

    coordinates = (x, y)

    z = get_utm_zone(coordinates)
    l = get_utm_letter(coordinates)

    if z not in _projections:
        _projections[z] = pyproj.Proj(proj='utm', zone=z, ellps='WGS84')

    x, y = _projections[z](coordinates[0], coordinates[1])

    if y < 0:
        y += 10000000.0

    epsg = zone_to_epsg(coordinates[1], z)

    return z, l, x, y, epsg


def unproject(z, l, x, y):

    if z not in _projections:
        _projections[z] = pyproj.Proj(proj='utm', zone=z, ellps='WGS84')

    if l < 'N':
        y -= 10000000.0

    lon, lat = _projections[z](x, y, inverse=True)

    return lon, lat


def bounds_to_df(bounds, crs):

    """
    Gets geographic bounds from a tuple

    Args:
        bounds (tuple)
        crs (object or str): The CRS of the bounds.

    Returns:
        ``GeoDataFrame``
    """

    poly_bounds = Polygon([(bounds[0], bounds[3]),
                           (bounds[2], bounds[3]),
                           (bounds[2], bounds[1]),
                           (bounds[0], bounds[1]),
                           (bounds[0], bounds[3])])

    return gpd.GeoDataFrame([0],
                            geometry=[poly_bounds],
                            crs=crs)


def centroid_to_bounds(centroid, crs, offset):

    """
    Gets geographic bounds from a tuple

    Args:
        centroid (tuple): The x, y coordinate.
        crs (object or str): The CRS of the bounds.
        offset (float): The centroid offset.

    Returns:
        ``GeoDataFrame``
    """

    gx, gy = centroid

    poly_bounds = Polygon([(gx - offset, gy + offset),
                           (gx + offset, gy + offset),
                           (gx + offset, gy - offset),
                           (gx - offset, gy - offset),
                           (gx - offset, gy + offset)])

    return df_to_bounds(gpd.GeoDataFrame([0],
                                         geometry=[poly_bounds],
                                         crs=crs))


def reproject_grid(grid, src_crs, dst_crs):

    """
    Reprojects a grid from lat/lon

    Args:
        grid (object)
        src_crs (object or str): The CRS of the bounds.
        dst_crs (object or str): The CRS of the bounds.

    Returns:
        ``GeoDataFrame``
    """

    dfs = gpd.GeoDataFrame([0],
                           geometry=[grid.geometry],
                           crs=src_crs)

    return dfs.to_crs(dst_crs)


def xy_to_bounds(lat_lon, grid_size):

    """
    Gets geographic bounds from a tuple

    Args:
        lat_lon (tuple)
        grid_size (float)

    Returns:
        ``tuple`` of (left, bottom, right, top)
    """

    point = Point([lat_lon[1], lat_lon[0]])

    df = gpd.GeoDataFrame([0],
                          geometry=[point],
                          crs=PROJ_DICT['wgs84']['proj4'])

    # Re-project from lat/lon
    df = df.to_crs(PROJ_DICT['equal_earth']['proj4'])

    left = df.geometry.x.values[0] - (grid_size / 2.0)
    right = df.geometry.x.values[0] + (grid_size / 2.0)
    bottom = df.geometry.y.values[0] - (grid_size / 2.0)
    top = df.geometry.y.values[0] + (grid_size / 2.0)

    bounds = (left, bottom, right, top)

    df = bounds_to_df(bounds, PROJ_DICT['equal_earth']['proj4'])

    df = df.to_crs(PROJ_DICT['wgs84']['proj4'])

    return df_to_bounds(df)


def df_to_bounds(df):
    """Converts a GeoDataFrame to a bounds tuple"""
    return df.bounds.values[0].tolist()


def _rmdir(pathdir):

    for child in pathdir.iterdir():

        if child.is_file():

            try:
                child.unlink()
            except:
                pass

    try:
        pathdir.rmdir()
    except:
        pass


def download_data(args):

    grid_ids = args.grid_ids

    if grid_ids:

        grid_ids = [int(idn) for idn in grid_ids]
    #     # Convert SLURM ids
    #     grid_ids = [int(str(idn).split('_')[-1]) for idn in grid_ids]

    gdl = GeoDownloads()

    wgs84_proj4 = PROJ_DICT['wgs84']['proj4']
    equal_earth_proj4 = PROJ_DICT['equal_earth']['proj4']

    if args.grid_file:

        df = gpd.read_file(args.grid_file)

        df.crs = PROJ_DICT['equal_earth']['proj4']

        for row in df.itertuples():

            if grid_ids:

                if int(row.UNQ) not in grid_ids:
                    continue

            # Project from Equal Earth to lat/lon
            df_grid = reproject_grid(row, equal_earth_proj4, wgs84_proj4)

            # Get the bounds in lat/lon (used to query the download location)
            bounds = df_to_bounds(df_grid)

            # Get the lon/lat coordinate
            centroid = (df_grid.geometry.values[0].centroid.x,
                        df_grid.geometry.values[0].centroid.y)

            # Set the UTM EPSG code for the output grid
            utm_epsg = latlon_to_utm(centroid[0], centroid[1])[-1]

            # Get the centroid in UTM coordinates
            df_grid_utm_centroid = reproject_grid(row, equal_earth_proj4, 'epsg:{:d}'.format(utm_epsg)).centroid
            gridx = df_grid_utm_centroid.values[0].x
            gridy = df_grid_utm_centroid.values[0].y

            # Set the output grid bounds
            out_bounds = centroid_to_bounds((gridx, gridy), 'epsg:{:d}'.format(utm_epsg), args.grid_size / 2.0)

            outdir = Path(args.outdir).joinpath('{:06d}'.format(int(row.UNQ)))

            outdir.mkdir(parents=True, exist_ok=True)

            # Cleanup previous sessions
            # for wildcard in ['L*.TIF', 'L*.txt', 'L*.xml', '*.tif', 'L*.jp2', '*.gstmp']:
            #
            #     for fn in outdir.glob(wildcard):
            #
            #         if fn.is_file():
            #             fn.unlink()
            #
            # for fn in outdir.joinpath('brdf').glob('*temp*'):
            #
            #     if fn.is_file():
            #         fn.unlink()
            #
            # for root, dirs, files in os.walk(outdir.as_posix()):
            #
            #     if dirs:
            #
            #         for subdir in dirs:
            #
            #             if subdir.startswith('angles'):
            #                 _rmdir(outdir.joinpath(subdir))

            gdl.download_cube(args.sensors,
                              args.dates,
                              bounds,
                              args.wavelengths,
                              bands_out=args.wavelengths_out,
                              l57_angles_path=args.l7_bin,
                              l8_angles_path=args.l8_bin,
                              mask_qa=args.mask_qa,
                              chunks=args.chunks,
                              num_threads=args.num_threads,
                              write_angle_files=True,
                              outdir=outdir.as_posix(),
                              crs=utm_epsg,
                              out_bounds=out_bounds,
                              ref_res=args.ref_res,
                              verbose=1,
                              separate=False,
                              n_workers=args.n_workers,
                              n_threads=args.n_threads,
                              n_chunks=500,
                              overwrite=True,
                              compress='lzw',
                              nodata=65535)

    else:

        bounds = xy_to_bounds(args.coords, args.grid_size)

        utm_epsg = latlon_to_utm(args.coords[1], args.coords[0])[-1]

        gdl.download_cube(args.sensors,
                          args.dates,
                          bounds,
                          args.wavelengths,
                          bands_out=args.wavelengths_out,
                          l57_angles_path=args.l7_bin,
                          l8_angles_path=args.l8_bin,
                          mask_qa=args.mask_qa,
                          chunks=args.chunks,
                          num_threads=args.num_threads,
                          write_angle_files=True,
                          outdir=args.outdir,
                          crs=utm_epsg,
                          verbose=1,
                          separate=False,
                          n_workers=args.n_workers,
                          n_threads=args.n_threads,
                          n_chunks=500,
                          overwrite=True,
                          nodata=65535)


def main():

    parser = argparse.ArgumentParser(description='Downloads and processes satellite cubes',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--coords', dest='coords', help='The geographic lat/lon coordinates',
                        default=None, nargs='+', type=float)
    parser.add_argument('--grid-file', dest='grid_file', help='The grid vector file (*overrides --coords)', default=None)
    parser.add_argument('--grid-ids', dest='grid_ids',
                        help='Grid ids to process (*the ids correspond to the --grid-file UNQ field)',
                        default=None, nargs='+', type=int)
    parser.add_argument('--grid-size', dest='grid_size', help='The grid size (in meters)', default=10000.0, type=float)
    parser.add_argument('--sensors', dest='sensors', help='The satellite sensors', default=['l8'], nargs='+')
    parser.add_argument('--ref-res', dest='ref_res', help='The reference cell resolution',
                        default=(10.0, 10.0), nargs='+', type=float)
    parser.add_argument('--dates', dest='dates',
                        help='The start and end dates, in yyyy-mm-dd (e.g., 2010-01-01 2011-01-01)',
                        default=None, nargs='+')
    parser.add_argument('--wavelengths', dest='wavelengths', help='The sensor wavelengths', default=None, nargs='+')
    parser.add_argument('--wavelengths-out', dest='wavelengths_out', help='The output sensor wavelengths',
                        default=None, nargs='+')
    parser.add_argument('--outdir', dest='outdir', help='The output directory', default=None)
    parser.add_argument('--n-workers', dest='n_workers', help='The number of parallel to_raster() workers',
                        default=2, type=int)
    parser.add_argument('--n-threads', dest='n_threads', help='The number of parallel to_raster() threads',
                        default=4, type=int)
    parser.add_argument('--l7-bin', dest='l7_bin', help='The Landsat 7 angle bin directory',
                        default='/nfs/dmab-data/code/bin/espa/landsat_angles')
    parser.add_argument('--l8-bin', dest='l8_bin', help='The Landsat 8 angle bin directory',
                        default='/nfs/dmab-data/code/bin/espa/l8_angles')
    parser.add_argument('--mask-qa', dest='mask_qa', help='Whether to mask data with the QA layer', action='store_true')
    parser.add_argument('--chunks', dest='chunks', help='The read chunk size', default=512, type=int)
    parser.add_argument('--num-threads', dest='num_threads', help='The number of GDAL warp threads', default=1, type=int)

    args = parser.parse_args()

    download_data(args)


if __name__ == '__main__':
    main()
