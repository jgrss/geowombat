import typing as T
from pathlib import Path


def warp(
    in_image: T.Union[str, Path],
    out_image: T.Union[str, Path],
    overwrite: bool = False,
    delete_input: bool = False,
    **kwargs
) -> None:

    """Warps an image.

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
    try:
        from osgeo import gdal
    except ImportError:
        from ..handler import GDAL_INSTALL_MSG

        raise ImportError(GDAL_INSTALL_MSG)

    if overwrite:
        if Path(out_image).is_file():
            Path(out_image).unlink()

    warp_options = gdal.WarpOptions(**kwargs)
    out_ds = gdal.Warp(
        str(Path(out_image).resolve()),
        str(Path(in_image).resolve()),
        options=warp_options,
    )
    out_ds = None

    if delete_input:
        Path(in_image).unlink()
