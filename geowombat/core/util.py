import os
from collections import namedtuple

from ..errors import logger

from affine import Affine


def get_geometry_info(geometry, res):

    """
    Gets information from a Shapely geometry object

    Args:
        geometry (object): A `shapely.geometry` object.
        res (float): The cell resolution for the affine transform.

    Returns:
        Geometry information (namedtuple)
    """

    GeomInfo = namedtuple('GeomInfo', 'left bottom right top shape transform')

    minx, miny, maxx, maxy = geometry.bounds
    out_shape = (int((maxy - miny) / res), int((maxx - minx) / res))

    return GeomInfo(left=minx,
                    bottom=miny,
                    right=maxx,
                    top=maxy,
                    shape=out_shape,
                    transform=Affine(res, 0.0, minx, 0.0, -res, maxy))


def get_file_extension(filename):

    """
    Gets file and directory name information

    Args:
        filename (str): The file name.

    Returns:
        Name information (namedtuple)
    """

    FileNames = namedtuple('FileNames', 'd_name f_name f_base f_ext')

    d_name, f_name = os.path.splitext(filename)
    f_base, f_ext = os.path.split(f_name)

    return FileNames(d_name=d_name, f_name=f_name, f_base=f_base, f_ext=f_ext)


def n_rows_cols(pixel_index, block_size, rows_cols):

    """
    Adjusts block size for the end of image rows and columns.

    Args:
        pixel_index (int): The current pixel row or column index.
        block_size (int): The image block size.
        rows_cols (int): The total number of rows or columns in the image.

    Returns:
        Adjusted block size as int.
    """

    return block_size if (pixel_index + block_size) < rows_cols else rows_cols - pixel_index


class Chunks(object):

    @staticmethod
    def get_chunk_dim(chunksize):
        return '{:d}d'.format(len(chunksize))

    @staticmethod
    def check_chunktype(chunksize, output='3d'):

        chunk_len = len(chunksize)
        output_len = int(output[0])

        if not isinstance(chunksize, tuple):
            logger.warning('  The chunksize parameter should be a tuple.')

        # TODO: make compatible with multi-layer predictions (e.g., probabilities)
        if chunk_len != output_len:
            logger.warning('  The chunksize should be two-dimensional.')

    @staticmethod
    def check_chunksize(chunksize, output='3d'):

        chunk_len = len(chunksize)
        output_len = int(output[0])

        if chunk_len != output_len:

            if (chunk_len == 2) and (output_len == 3):
                return (1,) + chunksize
            elif (chunk_len == 3) and (output_len == 2):
                return chunksize[1:]

        return chunksize
