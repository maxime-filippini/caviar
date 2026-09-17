import argparse
import pathlib
from collections.abc import Sequence

import polars as pl
from pydantic import BaseModel
from pydantic import DirectoryPath
from pydantic import model_validator

from caviar import data


class Args(BaseModel):
    path_input_dir: DirectoryPath
    pattern: str
    path_output_dir: DirectoryPath

    @model_validator(mode="after")
    def validate_pattern(self):
        if not self.pattern.endswith(".parquet"):
            raise ValueError("Not a valid parquet pattern")
        return self


def _prepare_returns(
    path_input_dir: pathlib.Path, pattern: str, path_output_dir: pathlib.Path
) -> None:
    for file in path_input_dir.rglob(pattern):
        df_eod = pl.read_parquet(file)
        df_prepared = data.prepare_returns_single_symbol(df_eod).drop("adjusted_close")
        df_prepared.write_parquet(path_output_dir / f"prepared_{file.name}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--path-input-dir",
        type=str,
    )
    parser.add_argument("-p", "--pattern", type=str, default="*")
    parser.add_argument(
        "-o",
        "--path-output-dir",
        type=pathlib.Path,
    )
    _args = parser.parse_args(argv)
    args = Args(
        path_input_dir=_args.path_input_dir,
        pattern=_args.pattern,
        path_output_dir=_args.path_output_dir,
    )

    _prepare_returns(
        path_input_dir=args.path_input_dir,
        pattern=args.pattern,
        path_output_dir=args.path_output_dir,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
