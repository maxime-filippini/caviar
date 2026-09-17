"""Fetch some market data from EODHD for analysis."""

import argparse
import datetime
import pathlib
from collections.abc import Iterable
from collections.abc import Sequence

import httpx
import polars as pl
from pydantic import BaseModel
from pydantic import DirectoryPath
from pydantic import model_validator

from caviar import eodhd
from caviar.settings import Settings

ROOT_URL = "https://eodhd.com/api/eod"
DEFAULT_SYMBOLS = ["SPY", "VDE"]


class Args(BaseModel):
    symbols: list[str]
    date_start: datetime.date
    date_end: datetime.date
    path_output_dir: DirectoryPath

    @model_validator(mode="after")
    def validate_date_logic(self):
        if self.date_end < self.date_start:
            raise ValueError("date_end cannot be anterior to date_start!")
        return self


def _fetch_data(
    api_key: str,
    symbols: Iterable[str],
    date_range: tuple[datetime.date, datetime.date],
) -> dict[str, pl.DataFrame]:
    with httpx.Client() as client:
        all_data = eodhd.request_eod_data(
            client=client, api_key=api_key, symbols=symbols, date_range=date_range
        )

    return {
        symbol: pl.DataFrame(data.root).with_columns(symbol=pl.lit(symbol))
        for symbol, data in all_data.items()
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("symbols", nargs="*", type=str)
    parser.add_argument("-s", "--start", type=str)
    parser.add_argument("-e", "--end", type=str)
    parser.add_argument(
        "-o",
        "--output-dir",
        type=pathlib.Path,
    )
    _args = parser.parse_args(argv)

    args = Args(
        symbols=_args.symbols,
        date_start=_args.start,
        date_end=_args.end,
        path_output_dir=_args.output_dir,
    )

    settings = Settings()  # pyright: ignore[reportCallIssue]

    df_dict = _fetch_data(
        symbols=args.symbols,
        api_key=settings.MARKET_API_KEY.get_secret_value(),
        date_range=(args.date_start, args.date_end),
    )

    for symbol, df in df_dict.items():
        df.write_parquet(
            args.path_output_dir
            / f"eod_{symbol}_{args.date_start.strftime('%Y-%m-%d')}_{args.date_end.strftime('%Y-%m-%d')}.parquet"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
