"""Run CAViaR fits on returns data."""

import argparse
import dataclasses
import datetime
import json
import logging
import pathlib
from collections.abc import Sequence
from uuid import uuid4

import numpy as np
import polars as pl
from pydantic import BaseModel
from pydantic import DirectoryPath

from caviar import logging_
from caviar.data import add_sample_markers
from caviar.data import build_datasets
from caviar.models import CaviarApproach
from caviar.models import get_func
from caviar.optimization import fit
from caviar.types import DatasetDict
from caviar.types import DateArray
from caviar.types import FloatArray

DEFAULT_N_BETA_SIM = 10_000
DEFAULT_M_FIRST_PASS = 10
DEFAULT_RNG_SEED = 123
DEFAULT_N_PARAMS = 4
DEFAULT_IN_SAMPLE_END_DATE = datetime.date(2024, 12, 31)
DEFAULT_OPTIMIZATION_TOLERANCE = 10e-10
DEFAULT_MAX_ITER = 1000
DEFAULT_PATTERN = "*.parquet"


type CaviarComputationMap = dict[str, dict[CaviarApproach, CaviarComputation]]


class Args(BaseModel):
    path_prepared_data_dir: DirectoryPath
    path_dir_out: DirectoryPath
    n_beta_sims: int
    m_first_pass: int
    in_sample_end_date: datetime.date
    rng_seed: int
    tol: float
    max_iter: int
    pattern: str


@dataclasses.dataclass
class Data:
    df_xy: pl.DataFrame
    df_xy_train: pl.DataFrame
    df_xy_test: pl.DataFrame
    x_train: DateArray
    y_train: FloatArray
    x_test: DateArray
    y_test: FloatArray


@dataclasses.dataclass
class CaviarComputation:
    key: str
    beta: FloatArray
    loss: float
    dates: DateArray
    vars: FloatArray


def _run_caviar_computations(
    data_dict: DatasetDict,
    rng: np.random.Generator,
    n_initial_sims: int,
    m_first_pass: int,
    path_dir_out: pathlib.Path,
    max_iter: int,
    tol: float,
    logger: logging.Logger,
):
    caviar_computations: CaviarComputationMap = {}

    for symbol, fit_data in data_dict.items():
        logger.info("Symbol [%s] - Calculation starting", symbol)
        caviar_computations[symbol] = {}

        # We calibrate every sub-model specification to the symbol data
        for approach in CaviarApproach:
            logger.info(
                "Symbol [%s] - Approach [%s] - Calculation starting", symbol, approach
            )
            beta, loss = fit(
                approach=approach,
                y=fit_data.y_train,
                y_backcast=fit_data.y_train[:300],  # TODO: Parametrize
                q=0.01,
                rng=rng,
                n_initial_sims=n_initial_sims,
                m_first_pass=m_first_pass,
                max_iter=max_iter,
                tol=tol,
            )

            logger.info(
                "Symbol [%s] - Approach [%s] - Beta: %s", symbol, approach, beta
            )

            caviar_func = get_func(approach)

            yy = caviar_func(
                y=fit_data.y_test, beta=beta, q=0.01, y_backcast=fit_data.y_train[-300:]
            )
            key = uuid4()
            caviar_computations[symbol][approach] = CaviarComputation(
                beta=beta, loss=loss, dates=fit_data.x_test, vars=yy, key=str(key)
            )

        # Serialize the computations
        fit_out = [
            {
                "symbol": symbol,
                "key": computation.key,
                "approach": str(approach),
                "beta": computation.beta.tolist(),
                "loss": computation.loss,
            }
            for approach, computation in caviar_computations[symbol].items()
        ]

        df = pl.concat(
            [
                pl.DataFrame(
                    {
                        "symbol": symbol,
                        "approach": str(approach),
                        "fit_key": computation.key,
                        "d_t": computation.dates,
                        "var_t->t+1": computation.vars,
                        "r_t-1->t": fit_data.y_test,
                    }
                )
                for approach, computation in caviar_computations[symbol].items()
            ]
        )

        with open(path_dir_out / f"{symbol}.json", "w") as fd:
            json.dump(fit_out, fd, indent=2)

        df.write_parquet(path_dir_out / f"{symbol}.parquet")

    return caviar_computations


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--path-prepared-data-dir", type=pathlib.Path)
    parser.add_argument("-o", "--path-dir-out", type=pathlib.Path)
    parser.add_argument("--n-beta-sims", type=int, default=DEFAULT_N_BETA_SIM)
    parser.add_argument("--m-first-pass", type=int, default=DEFAULT_M_FIRST_PASS)
    parser.add_argument(
        "--in-sample-end-date", type=str, default=DEFAULT_IN_SAMPLE_END_DATE
    )
    parser.add_argument("--rng-seed", type=int, default=DEFAULT_RNG_SEED)
    parser.add_argument("--max-iter", type=int, default=DEFAULT_MAX_ITER)
    parser.add_argument("--tol", type=float, default=DEFAULT_OPTIMIZATION_TOLERANCE)
    parser.add_argument("-p", "--pattern", type=str, default=DEFAULT_PATTERN)
    _args = parser.parse_args(argv)
    args = Args(**vars(_args))

    logger = logging_.configure_logger(
        name="root",
        level=logging.INFO,
        log_file=args.path_dir_out / "run_caviar.log",
    )

    rng = np.random.default_rng(args.rng_seed)

    df_prepared = pl.read_parquet(args.path_prepared_data_dir / args.pattern).pipe(
        add_sample_markers,
        train_dates=(None, args.in_sample_end_date),
        test_dates=(None, None),
    )

    data_dict = build_datasets(df_prepared)
    _ = _run_caviar_computations(
        data_dict=data_dict,
        rng=rng,
        n_initial_sims=args.n_beta_sims,
        m_first_pass=args.m_first_pass,
        path_dir_out=args.path_dir_out,
        max_iter=args.max_iter,
        tol=args.tol,
        logger=logger,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
