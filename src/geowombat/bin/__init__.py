import os
from pathlib import Path
import tarfile
import typing as T
from dataclasses import dataclass


p = Path(os.path.abspath(os.path.dirname(__file__)))

espa_tarball = p / 'ESPA.tar.gz'


@dataclass
class AnglePaths:
    l8_angles_path: Path
    l57_angles_path: Path


def extract_espa_tools(out_path: T.Union[str, Path]) -> AnglePaths:
    with tarfile.open(espa_tarball) as f:
        f.extractall(str(out_path))

    espa_path = Path(out_path) / 'ESPA'

    angle_paths = AnglePaths(
        l8_angles_path=espa_path / 'l8_angles',
        l57_angles_path=espa_path / 'landsat_angles'
    )

    return angle_paths
