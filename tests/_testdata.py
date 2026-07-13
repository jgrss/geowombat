"""On-demand fetching of large test-only rasters.

These files are not shipped with the geowombat package (see issue #362); they
are downloaded from a GitHub Release asset on first use, hash-verified, and
cached locally by pooch. Import this module from a test to obtain a local path:

    from _testdata import fetch
    path = fetch("l8_224077_20200518_B2_nan.tif")
"""

import pooch

POOCH = pooch.create(
    path=pooch.os_cache("geowombat-testdata"),
    base_url=(
        "https://github.com/jgrss/geowombat/releases/download/test-data-v1/"
    ),
    registry={
        "l8_224077_20200518_B2_nan.tif": (
            "sha256:46900b4dba63f3215d42f10a28fb33c0b65c337f1284eca7729f3b3d57296bb0"
        ),
        "l8_224078_20200518_B2_nan.tif": (
            "sha256:90d72f714d8e1f2df91912943b8c5fa557c75d075697c30efc05a9bd23f8e133"
        ),
    },
)


def fetch(name):
    """Return a local path to ``name``, downloading and caching if needed."""
    return POOCH.fetch(name)
