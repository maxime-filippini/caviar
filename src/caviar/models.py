"""CAViaR specifications and computation methods."""

from enum import StrEnum
from typing import Protocol
from typing import assert_never

import numpy as np

from caviar.types import FloatArray


class CaviarFunc(Protocol):
    """Interface defining a CAViaR-producing function."""

    def __call__(
        self, y: FloatArray, beta: FloatArray, q: float, y_backcast: FloatArray
    ) -> FloatArray:
        """Produce an array of CAViaR forecasts.

        Args:
            y (FloatArray): Returns array.
            beta (FloatArray): Parameter vector.
            q (float): Probability associated with the quantile to be modelled.
            y_backcast (FloatArray): Array used to compute the initial quantile.

        Returns:
            FloatArray: Array of CAViaR forecasts.
        """
        ...


class CaviarApproach(StrEnum):
    """Keys for implemented CAViaR approaches."""

    AS = "AS"
    IG = "IG"
    SAV = "SAV"


def caviar_asymmetric(
    y: FloatArray, beta: FloatArray, q: float, y_backcast: FloatArray
) -> FloatArray:
    """Produce an array of CAViaR forecasts using the Asymmetric Slope spec.

    Args:
        y (FloatArray): Returns array.
        beta (FloatArray): Parameter vector.
        q (float): Probability associated with the quantile to be modelled.
        y_backcast (FloatArray): Array used to compute the initial quantile.

    Returns:
        FloatArray: Array of CAViaR forecasts.
    """
    T = len(y)

    f = np.empty((T, *beta.shape[1:]))
    f[0] = np.quantile(y_backcast, q=q)

    for t in range(1, T):
        y_pos = max(y[t - 1], 0.0)
        y_neg = max(-y[t - 1], 0.0)

        f[t] = beta[0] + beta[1] * f[t - 1] + beta[2] * y_pos + beta[3] * y_neg

    return f


def caviar_indirect_garch(
    y: FloatArray, beta: FloatArray, q: float, y_backcast: FloatArray
) -> FloatArray:
    """Produce an array of CAViaR forecasts using the Indirect GARCH spec.

    Args:
        y (FloatArray): Returns array.
        beta (FloatArray): Parameter vector.
        q (float): Probability associated with the quantile to be modelled.
        y_backcast (FloatArray): Array used to compute the initial quantile.

    Returns:
        FloatArray: Array of CAViaR forecasts.
    """
    T = len(y)

    f = np.empty((T, *beta.shape[1:]))
    f[0] = np.quantile(y_backcast, q=q)

    for t in range(1, T):
        f[t] = -np.sqrt(beta[0] + beta[1] * f[t - 1] ** 2 + beta[2] * y[t - 1] ** 2)

    return f


def caviar_symmetric_absolute_value(
    y: FloatArray,
    beta: FloatArray,
    q: float,
    y_backcast: FloatArray,
) -> FloatArray:
    """Produce an array of CAViaR forecasts using the SAV spec.

    Args:
        y (FloatArray): Returns array.
        beta (FloatArray): Parameter vector.
        q (float): Probability associated with the quantile to be modelled.
        y_backcast (FloatArray): Array used to compute the initial quantile.

    Returns:
        FloatArray: Array of CAViaR forecasts.
    """
    T = len(y)

    f = np.empty((T, *beta.shape[1:]))
    f[0] = np.quantile(y_backcast, q=q)

    for t in range(1, T):
        f[t] = beta[0] + beta[1] * f[t - 1] + beta[2] * np.abs(y[t - 1])

    return f


def get_func(approach: CaviarApproach) -> CaviarFunc:
    """Get the compute function associated with a given approach.

    Args:
        approach (CaviarApproach): The approach for which to retrieve the
            function.

    Returns:
        CaviarFunc: The resulting function.
    """
    match approach:
        case CaviarApproach.AS:
            return caviar_asymmetric
        case CaviarApproach.IG:
            return caviar_indirect_garch
        case CaviarApproach.SAV:
            return caviar_symmetric_absolute_value
        case _:
            assert_never(approach)


def get_n_params(approach: CaviarApproach) -> int:
    """Get the parameter vector size associated with a given approach.

    Args:
        approach (CaviarApproach): The approach for which to retrieve the
            parameter vector size.

    Returns:
        int: The resulting parameter vector size.
    """
    match approach:
        case CaviarApproach.AS:
            return 4
        case CaviarApproach.IG:
            return 3
        case CaviarApproach.SAV:
            return 3
        case _:
            assert_never(approach)
