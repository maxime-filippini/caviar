"""Types and functions involved in the retrieval of market data via EODHD."""

import datetime
from collections.abc import Iterable

import httpx
from pydantic import BaseModel
from pydantic import RootModel

ROOT_URL = "https://eodhd.com/api/eod"


class EodDataSingle(BaseModel):
    date: datetime.date
    open: float
    close: float
    high: float
    low: float
    adjusted_close: float
    volume: float


class EodData(RootModel[list[EodDataSingle]]):
    pass


def request_eod_data(
    client: httpx.Client,
    api_key: str,
    symbols: Iterable[str],
    date_range: tuple[datetime.date, datetime.date],
) -> dict[str, EodData]:
    """Request end-of-day data.

    Assumes that the API returns the following fields:
        - date
        - open
        - close
        - high
        - low
        - adjusted_close
        - volume

    Args:
        client (httpx.Client): The HTTP client used to make the request.
        api_key (str): The EODHD API key
        symbols (Iterable[str]): Symbols for which to retrieve data.
        date_range (tuple[datetime.date, datetime.date]): Start and end dates.

    Returns:
        dict[str, EodData]: Processed data.
    """
    date_start, date_end = date_range
    out: dict[str, EodData] = {}

    for symbol in symbols:
        resp = client.get(
            f"{ROOT_URL}/{symbol}",
            params={
                "api_token": api_key,
                "from": date_start.strftime("%Y-%m-%d"),
                "to": date_end.strftime("%Y-%m-%d"),
                "period": "d",
                "fmt": "json",
            },
        )

        resp.raise_for_status()
        out[symbol] = EodData.model_validate(resp.json())

    return out
