"""Run EWMA computations."""

import argparse
import datetime
import json
import pathlib
from collections.abc import Sequence
from uuid import uuid4

import numpy as np
import polars as pl
from pydantic import BaseModel
from pydantic import DirectoryPath
from scipy import stats

from caviar.data import add_sample_markers
from caviar.data import build_datasets
from caviar.ewma import ewma_variance
from caviar.ewma import fit
from caviar.types import DatasetDict


class Args(BaseModel):
    path_prepared_data_dir: DirectoryPath
    path_dir_out: DirectoryPath
    train_from: datetime.date | None = None
    train_to: datetime.date
    test_from: datetime.date | None = None
    test_to: datetime.date | None = None


def _run_ewma_computations(
    data_dict: DatasetDict,
    path_dir_out: pathlib.Path,
):
    for symbol, data in data_dict.items():
        # Variance is computed on data.y_test
        # y_test: [y_0, y_1, y_2, ...]
        # h:      [h_0, h_1, h_2, ...]
        #         |     |
        #         |     | uses h_0 and y_0, relevant for y_1
        #         | via backcast

        lam, nll = fit(y=data.y_train, y_backcast=data.y_train[:60])
        h = ewma_variance(y=data.y_test, lam=lam, y_backcast=data.y_train[-60:])
        vols = np.sqrt(h)
        vars_ = np.sqrt(h) * stats.norm.ppf(0.01)

        key = uuid4()
        ewma_fit = {"symbol": symbol, "key": str(key), "lam": lam, "loss": -nll}

        df = pl.DataFrame(
            {
                "symbol": symbol,
                "fit_key": str(key),
                "date": data.x_test,
                "vol": vols[:-1],
                "next_vol": vols[1:],
                "var": vars_[:-1],
            }
        )

        with open(path_dir_out / f"{symbol}.json", "w") as fd:
            json.dump(ewma_fit, fd, indent=2)

        df.write_parquet(path_dir_out / f"{symbol}.parquet")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--path-prepared-data-dir", type=pathlib.Path)
    parser.add_argument("-o", "--path-dir-out", type=pathlib.Path)
    parser.add_argument("--train-from", type=str, default=None)
    parser.add_argument("--train-to", type=str)
    parser.add_argument("--test-from", type=str, default=None)
    parser.add_argument("--test-to", type=str, default=None)
    _args = parser.parse_args(argv)
    args = Args(**vars(_args))

    df_prepared = pl.read_parquet(args.path_prepared_data_dir / "*.parquet").pipe(
        add_sample_markers,
        train_dates=(args.train_from, args.train_to),
        test_dates=(args.test_from, args.test_to),
    )

    data_dict = build_datasets(df_prepared)
    _run_ewma_computations(data_dict, path_dir_out=args.path_dir_out)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
