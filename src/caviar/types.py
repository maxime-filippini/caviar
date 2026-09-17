"""Common types and validators."""

import dataclasses
import pathlib
from typing import Annotated

import numpy as np
import polars as pl
from pydantic import AfterValidator

type FloatArray = np.ndarray[tuple[int], np.dtype[np.float64]]
type DateArray = np.ndarray[tuple[int], np.dtype[np.datetime64]]


@dataclasses.dataclass
class Dataset:
    """A grouping of data items used for computations."""

    df_xy: pl.DataFrame
    df_xy_train: pl.DataFrame
    df_xy_test: pl.DataFrame
    x_train: DateArray
    y_train: FloatArray
    x_test: DateArray
    y_test: FloatArray


type DatasetDict = dict[str, Dataset]


def _parquet_validator(path: pathlib.Path):
    if path.suffix != ".parquet":
        raise ValueError("Not a valid parquet file!")
    return path


type RequireParquet[T] = Annotated[T, AfterValidator(_parquet_validator)]
