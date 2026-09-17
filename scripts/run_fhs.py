"""Run FHS computations."""

import argparse
import json
import logging
import pathlib
from collections.abc import Sequence

import numpy as np
import polars as pl
from numpy.lib.stride_tricks import sliding_window_view
from pydantic import BaseModel
from pydantic import DirectoryPath

from caviar import logging_


class Args(BaseModel):
    path_prepared_data_dir: DirectoryPath
    path_dir_ewma: DirectoryPath
    path_dir_out: DirectoryPath


class EwmaFitData(BaseModel):
    symbol: str
    key: str
    lam: float
    loss: float


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--path-prepared-data-dir", type=pathlib.Path)
    parser.add_argument("-o", "--path-dir-out", type=pathlib.Path)
    parser.add_argument("--path-dir-ewma", type=pathlib.Path)
    _args = parser.parse_args(argv)
    args = Args(**vars(_args))

    logger = logging_.configure_logger(
        name="root",
        level=logging.INFO,
        log_file=args.path_dir_out / "run_fhs.log",
    )

    df_prepared = pl.read_parquet(args.path_prepared_data_dir / "*.parquet").rename(
        {"date": "d_t", "return": "r_t-1->t"}
    )

    symbols = df_prepared["symbol"].unique().to_list()

    fit_data: dict[str, EwmaFitData] = {}
    for symbol in symbols:
        dfs: list[pl.DataFrame] = []
        logger.info("Symbol [%s] - Now processing", symbol)
        with open(args.path_dir_ewma / f"{symbol}.json") as fd:
            d = json.load(fd)

        fit_data[symbol] = EwmaFitData(**d)

        df_raw = pl.read_parquet(args.path_dir_ewma / f"{symbol}.parquet")

        df_symbol = (
            df_raw.rename(
                {"vol": "vol_t-1->t", "next_vol": "vol_t->t+1", "date": "d_t"}
            )
            .drop("var")
            .join(df_prepared, on=["d_t", "symbol"])
            .group_by("symbol", "fit_key")
            .map_groups(
                lambda df: df.sort("d_t")
                .with_columns(pl.col("d_t").shift(-1).alias("d_t+1"))
                .with_columns(pl.col("r_t-1->t").shift(-1).alias("r_t->t+1"))
            )
            .sort("d_t")
        )

        fit_keys = df_symbol["fit_key"].unique().to_list()

        logger.info("Symbol [%s] - %i fit key(s) found", symbol, len(fit_keys))

        for fit_key in fit_keys:
            df_fit = df_symbol.filter(pl.col("fit_key").eq(fit_key))

            d_roll = sliding_window_view(df_fit["d_t"].to_numpy(), window_shape=500)
            r_roll = sliding_window_view(
                df_fit["r_t-1->t"].to_numpy(), window_shape=500
            )
            vol_roll = sliding_window_view(
                df_fit["vol_t-1->t"].to_numpy(), window_shape=500
            )
            next_vol_roll = sliding_window_view(
                df_fit["vol_t->t+1"].to_numpy(), window_shape=500
            )

            r_rescaled_roll = r_roll / vol_roll * next_vol_roll[:, [-1]]
            vars_ = np.quantile(r_rescaled_roll, q=0.01, axis=-1)
            d_vars = d_roll[:, -1]

            df_var = pl.DataFrame({"d_t": d_vars, "var_t->t+1": vars_})

            df_enriched = df_fit.join(df_var, on="d_t", how="left")

            dfs.append(df_enriched)

        pl.concat(dfs).write_parquet(args.path_dir_out / f"{symbol}.parquet")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
