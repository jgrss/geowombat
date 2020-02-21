from pathlib import Path

from osgeo import gdal


def warp(in_image,
         out_image,
         overwrite=False,
         delete_input=False,
         **kwargs):

    """
    Warps an image

    Args:
        in_image (str): The input image.
        out_image (str): The output image.
        overwrite (Optional[bool]): Whether to overwrite an existing output.
        delete_input (Optional[bool]): Whether to delete the input image after warping.
        kwargs (Optional[dict]):
            format=None, outputBounds=None (minX, minY, maxX, maxY),
            outputBoundsSRS=None, targetAlignedPixels=False,
            width=0, height=0, srcAlpha=False, dstAlpha=False, warpOptions=None,
            errorThreshold=None, warpMemoryLimit=None,
            creationOptions=None, outputType=0, workingType=0,
            resampleAlg=resample_dict[resample], srcNodata=None, dstNodata=None,
            multithread=False, tps=False, rpc=False, geoloc=False,
            polynomialOrder=None, transformerOptions=None, cutlineDSName=None,
            cutlineLayer=None, cutlineWhere=None, cutlineSQL=None,
            cutlineBlend=None, cropToCutline=False, copyMetadata=True,
            metadataConflictValue=None, setColorInterpretation=False,
            callback=None, callback_data=None
    """

    if overwrite:

        if Path(out_image).is_file():
            Path(out_image).unlink()

    warp_options = gdal.WarpOptions(**kwargs)

    out_ds = gdal.Warp(out_image,
                       in_image,
                       options=warp_options)

    out_ds = None

    if delete_input:
        Path(in_image).unlink()
