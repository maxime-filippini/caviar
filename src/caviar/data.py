import datetime

import polars as pl

from caviar.types import Dataset
from caviar.types import DatasetDict


def prepare_returns_single_symbol(df_eod: pl.DataFrame) -> pl.DataFrame:
    """Prepare returns from adjusted_close from a table holding a single symbol.

    Args:
        df_eod (pl.DataFrame): EOD adjusted close dataframe.

    Returns:
        pl.DataFrame: Long-format dataframe holding returns.
    """
    df = (
        df_eod.select("date", "adjusted_close", "symbol")
        .sort("date")
        .with_columns(pl.col("adjusted_close").forward_fill(limit=5))
    )

    assert df.filter(pl.col("adjusted_close").is_null()).is_empty()

    df_returns = df.with_columns(
        pl.col("adjusted_close").pct_change().alias("return")
    ).drop_nulls()

    return df_returns


def prepare_returns_wide(df_eod: pl.DataFrame) -> pl.DataFrame:
    """Prepare returns from a wide adjusted_close table.

    Args:
        df_eod (pl.DataFrame): Wide-format EOD adjusted close dataframe.

    Returns:
        pl.DataFrame: Wide-format returns dataframe.
    """
    df_wide = (
        df_eod.select("date", "adjusted_close", "symbol")
        .sort("date")
        .pivot("symbol", index="date")
        .with_columns(pl.exclude("date").forward_fill(limit=5))
    )

    assert df_wide.filter(pl.any_horizontal(pl.all().is_null())).is_empty()
    df_returns = df_wide.with_columns(pl.exclude("date").pct_change()).drop_nulls()
    return df_returns


def build_datasets(df_prepared_long: pl.DataFrame) -> DatasetDict:
    """Build train/test datasets from a long-format returns dataframe.

    The input dataframe is expected to have a `sample` column already marking
    the train/test rows.

    Args:
        df_prepared_long (pl.DataFrame): Long-format returns dataframe.

    Returns:
        DatasetDict: Dictionary holding a dataset per symbol.
    """
    data = {}
    for symbol in df_prepared_long.get_column("symbol").unique():
        df_xy = (
            df_prepared_long.filter(pl.col("symbol").eq(symbol))
            .sort("date")
            .drop("symbol")
        )

        df_xy_train = df_xy.filter(pl.col("sample").eq("train"))
        df_xy_test = df_xy.filter(pl.col("sample").eq("test"))

        x_train = df_xy_train.get_column("date").to_numpy()
        y_train = df_xy_train.get_column("return").to_numpy()
        x_test = df_xy_test.get_column("date").to_numpy()
        y_test = df_xy_test.get_column("return").to_numpy()

        data[symbol] = Dataset(
            df_xy=df_xy,
            df_xy_train=df_xy_train,
            df_xy_test=df_xy_test,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
        )

    return data


def add_sample_markers(
    df_prepared_long: pl.DataFrame,
    train_dates: tuple[datetime.date | None, datetime.date],
    test_dates: tuple[datetime.date | None, datetime.date | None],
) -> pl.DataFrame:
    """Mark dataset rows as belonging into `train` or `test` datasets.

    - If the first date of the training dataset is not provided, this function
    will take the first available date.
    - If the first date of the testing dataset is not provided, it will take the
    first date after the end of the training dataset.
    - If the last date of the testing dataset is not provided, it will take the
    last available date.

    Args:
        df_prepared_long (pl.DataFrame): Long-format returns dataframe
        train_dates (tuple[datetime.date | None, datetime.date]): Dates
            bookending the training dataset.
        test_dates (tuple[datetime.date | None, datetime.date | None]): Dates
            bookending the testing dataset.

    Raises:
        ValueError: _description_

    Returns:
        pl.DataFrame: _description_
    """
    train_start, train_end = train_dates
    test_start, test_end = test_dates

    include_test_start: bool = True

    if train_end is None:
        raise ValueError("End date for training set needs to be defined.")

    if test_start is None:
        test_start = train_end
        include_test_start = False

    train_filter_expr = (
        pl.col("date")
        .pipe(lambda e: e.ge(train_start) if train_start is not None else e)
        .pipe(lambda e: e.le(train_end))
    )

    test_filter_expr = (
        pl.col("date")
        .pipe(lambda e: e.ge(test_start) if include_test_start else e.gt(test_start))
        .pipe(lambda e: e.le(test_end) if test_end is not None else e)
    )

    return df_prepared_long.with_columns(
        pl.when(train_filter_expr)
        .then(pl.lit("train"))
        .when(test_filter_expr)
        .then(pl.lit("test"))
        .alias("sample")
    ).filter(pl.col("sample").is_not_null())
