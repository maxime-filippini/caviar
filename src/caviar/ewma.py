from collections.abc import Callable

import numpy as np
from scipy.optimize import minimize_scalar

from caviar.types import FloatArray


def ewma_variance(
    y: FloatArray,
    lam: float,
    y_backcast: FloatArray,
) -> FloatArray:
    """Compute EWMA variances for a set of returns.

    Args:
        y (FloatArray): Returns array.
        lam (float): Decay factor (model parameter).
        y_backcast (FloatArray): Array used to compute the initial variance.

    Returns:
        FloatArray: Array of EWMA variances.
    """
    T = y.shape[0]
    v = np.empty((T + 1, *y.shape[1:]), dtype=float)

    # Forecast variance for y[0].
    v[0] = np.var(y_backcast, axis=0, ddof=1)

    for t in range(T):
        # After observing y[t], construct the forecast for y[t + 1].
        v[t + 1] = lam * v[t] + (1.0 - lam) * y[t] ** 2

    return v


def ewma_objective_func(
    y: FloatArray,
    y_backcast: FloatArray,
) -> Callable[[float], float]:
    """Factory function for creating EWMA loss functions based on pre-set data.

    Args:
        y (FloatArray): Returns array.
        y_backcast (FloatArray): Array used to compute the initial variance.

    Returns:
        Callable[[float], float]: Function producing negative log-likelihood as
            a function of the decay factor parameter.
    """
    y = np.asarray(y, dtype=float)

    def inner(lam: float) -> float:
        h = ewma_variance(
            y=y,
            lam=lam,
            y_backcast=y_backcast,
        )

        h_ll = np.maximum(h[1:-1], 1e-12)
        y_ll = y[1:]

        nll = 0.5 * np.sum(np.log(2.0 * np.pi) + np.log(h_ll) + y_ll**2 / h_ll)

        return float(nll)

    return inner


def fit(
    y: FloatArray,
    y_backcast: FloatArray,
    xatol: float = 1e-10,
    fatol: float | None = None,
) -> tuple[float, float]:
    """Fit an EWMA model on an array of returns.

    Args:
        y (FloatArray): Returns array.
        y_backcast (FloatArray): Array used to compute the initial variance.
        xatol (float, optional): Absolute tolerance on the lambda parameter to
            define convergence. Defaults to 1e-10.
        fatol (float | None, optional): Absolute tolerance on the loss output to
            define convergence. Defaults to None.

    Raises:
        RuntimeError: If the optimization fails.

    Returns:
        tuple[float, float]: (lambda, loss) tuple
    """
    objective_func = ewma_objective_func(
        y=y,
        y_backcast=y_backcast,
    )

    opts = {
        **{"xatol": xatol},
        **({"fatol": fatol} if fatol is not None else {}),
    }

    result = minimize_scalar(
        objective_func,
        method="bounded",
        bounds=(1e-6, 1.0 - 1e-6),
        options=opts,
    )

    if not result.success:
        raise RuntimeError(f"Optimization failed: {result.message}")

    return float(result.x), float(result.fun)
