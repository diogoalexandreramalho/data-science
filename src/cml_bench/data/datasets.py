"""Dataset abstraction for the project's two UCI datasets.

Each subclass encapsulates where the raw file lives, how to fetch it from UCI,
how to read it into a canonical DataFrame, and any dataset-specific metadata.

Use via the DATASETS registry (keyed by the legacy "PD"/"CT" source codes used
by the CLI), or directly: `Parkinsons().read()`, `Covertype().download()`.
"""

from __future__ import annotations

import gzip
import shutil
import ssl
import subprocess
import urllib.request
import zipfile
from abc import ABC, abstractmethod
from functools import cached_property
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

DATA_RAW_DIR = Path(__file__).resolve().parents[3] / "data" / "raw"


class Dataset(ABC):
    name: str
    source_code: str
    filename: str
    target_column: str
    url: str

    @property
    def raw_dir(self) -> Path:
        return DATA_RAW_DIR / self.name

    @property
    def raw_path(self) -> Path:
        return self.raw_dir / self.filename

    @abstractmethod
    def download(self, insecure: bool = False) -> None: ...

    @abstractmethod
    def read(self) -> pd.DataFrame: ...

    def _fetch(self, dest: Path, insecure: bool) -> None:
        ctx = ssl._create_unverified_context() if insecure else None
        with urllib.request.urlopen(self.url, context=ctx) as r, dest.open("wb") as f:
            shutil.copyfileobj(r, f)


class Covertype(Dataset):
    name = "covertype"
    source_code = "CT"
    filename = "covtype.csv"
    target_column = "Cover_Type"
    url = "https://archive.ics.uci.edu/static/public/31/covertype.zip"

    # UCI ships the data headerless; names come from covtype.info in the same zip.
    column_names = (
        [
            "Elevation",
            "Aspect",
            "Slope",
            "Horizontal_Distance_To_Hydrology",
            "Vertical_Distance_To_Hydrology",
            "Horizontal_Distance_To_Roadways",
            "Hillshade_9am",
            "Hillshade_Noon",
            "Hillshade_3pm",
            "Horizontal_Distance_To_Fire_Points",
        ]
        + [f"Wilderness_Area{i}" for i in range(1, 5)]
        + [f"Soil_Type{i}" for i in range(1, 41)]
        + ["Cover_Type"]
    )

    def download(self, insecure: bool = False) -> None:
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            zip_path = tmp / "covertype.zip"
            self._fetch(zip_path, insecure)
            with zipfile.ZipFile(zip_path) as zf:
                zf.extract("covtype.data.gz", tmp)
            with (
                gzip.open(tmp / "covtype.data.gz", "rt") as src,
                self.raw_path.open("w") as out,
            ):
                out.write(",".join(self.column_names) + "\n")
                shutil.copyfileobj(src, out)

    def read(self) -> pd.DataFrame:
        return pd.read_csv(self.raw_path, sep=",", decimal=".")


class Parkinsons(Dataset):
    name = "parkinsons"
    source_code = "PD"
    filename = "pd_speech_features.csv"
    target_column = "class"
    url = "https://archive.ics.uci.edu/static/public/470/parkinson+s+disease+classification.zip"

    def download(self, insecure: bool = False) -> None:
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            zip_path = tmp / "parkinsons.zip"
            self._fetch(zip_path, insecure)
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(tmp)
            rar = next(tmp.glob("*.rar"))
            subprocess.run(["tar", "-xf", str(rar), "-C", str(tmp)], check=True)
            csv = next(tmp.glob("*.csv"))
            shutil.move(str(csv), self.raw_path)

    def read(self) -> pd.DataFrame:
        return pd.read_csv(self.raw_path, sep=",", decimal=".", skiprows=1)

    @cached_property
    def feature_groups(self) -> dict[str, list[str]]:
        """Group the 754 features by UCI's category header row.

        Reads the CSV without `skiprows=1` so the category labels (Baseline Features,
        MFCC, TQWT Features, ...) become column names; the actual feature names live
        on row 2 (= row index 0 of the resulting frame).
        """
        raw = pd.read_csv(self.raw_path, sep=",", decimal=".")
        groups: dict[str, list[str]] = {"Start": []}
        current = "Start"
        for col in raw.columns:
            if col.startswith("Unnamed"):
                groups[current].append(raw[col].iloc[0])
            else:
                groups[col] = [raw[col].iloc[0]]
                current = col
        return groups


DATASETS: dict[str, Dataset] = {ds.source_code: ds for ds in [Covertype(), Parkinsons()]}
