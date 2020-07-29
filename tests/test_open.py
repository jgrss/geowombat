import pytest
from pathlib import Path

import geowombat as gw
from geowombat.data import l8_224078_20200518


def test_open_incomplete_path():

    with pytest.raises(OSError):
        gw.open(Path(l8_224078_20200518).name)


def test_open():

    with gw.open(l8_224078_20200518) as src:
        assert src.gw.nbands == 4


def test_open_path():

    with gw.open(Path(l8_224078_20200518)) as src:
        assert src.gw.nbands == 4
