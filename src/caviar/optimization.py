"""Optimization routines and utilities for CAViaR approaches."""

import logging
from collections.abc import Callable
from typing import cast

import numpy as np
from scipy.optimize import minimize

from caviar.models import CaviarApproach
from caviar.models import CaviarFunc
from caviar.models import get_func
from caviar.models import get_n_params
from caviar.types import FloatArray

logger = logging.getLogger("caviar.optimization")


def compute_loss(y: FloatArray, f: FloatArray, q: float) -> FloatArray:
    """Loss function for quantile forecast estimates.

    This function is implemented to allow for loss computations on batches. It
    is assumed that both y and f are of arbitrary dimensions (but the same)
    where losses are aggregated over the first dimension. This means the first
    dimension represents Time, while all other dimensions represent computation
    batches.

    Args:
        y (FloatArray): Returns array.
        f (FloatArray): Quantile forecasts array.
        q (float): Probability associated with the quantile

    Returns:
        FloatArray: Array of losses.
    """
    y_b = y.reshape((y.size,) + (1,) * (f.ndim - 1))
    u = y_b - f
    loss = np.where(u >= 0, q * u, (q - 1.0) * u)
    return np.sum(loss, axis=0)


def loss_func_factory(
    caviar_func: CaviarFunc, y: FloatArray, q: float, y_backcast: FloatArray
) -> Callable[[FloatArray], float]:
    """Objective function factory.

    Args:
        caviar_func (CaviarFunc): The CAViaR computation function.
        y (FloatArray): Array of returns.
        q (float): Probability associated with the quantile being modelled.
        y_backcast (FloatArray): Array of returns used to compute the initial
            quantile.

    Returns:
        Callable[[FloatArray], float]: Loss function.
    """

    def inner(beta: FloatArray) -> float:
        with np.errstate(over="ignore", invalid="ignore"):
            f = caviar_func(y=y, beta=beta, q=q, y_backcast=y_backcast)

        if not np.all(np.isfinite(f)):
            return 1e100

        value = compute_loss(y, f, q)

        if not np.isfinite(value):
            return 1e100

        return float(value)

    return inner


def fit(
    approach: CaviarApproach,
    y: FloatArray,
    y_backcast: FloatArray,
    q: float,
    rng: np.random.Generator,
    n_initial_sims: int,
    m_first_pass: int,
    max_iter: int,
    tol: float,
) -> tuple[FloatArray, float]:
    """Fit a CAViaR model using returns as sole observable array.

    Args:
        approach (CaviarApproach): The CAViaR approach (e.g. IG, SAV, SA).
        y (FloatArray): Array of lagged portfolio returns.
        y_backcast (FloatArray): Array of lagged returns used to compute the starting quantile.
        q (float): The probability associated with the quantile being modelled.
        rng (np.random.Generator): Generator used for simulating random numbers.
        n_initial_sims (int): Number of initial parameter vectors to be simulated.
        m_first_pass (int): Number of parameter vectors to retain after first pass.
        max_iter (int): Maximum number of Simplex -> QN loops to perform for a given parameter vector.
        tol (float): Tolerance used for both parameter and function outputs.

    Returns:
        tuple[float, float]: Optimal (beta, loss) tuple.
    """
    # Determine the CAViaR DGP and the size of the beta vector based on the
    # selected approach.
    caviar_func = get_func(approach)
    n_params = get_n_params(approach)

    # 1. Simulate N random beta vectors sampling a U([0,1]) distribution
    cand_params = rng.uniform(0.0, 1.0, size=(n_params, n_initial_sims))

    # 2. Compute the loss for each of these vectors. Here, the `caviar_func`
    #    is written so as to run on an array of beta vectors.
    f_init = caviar_func(beta=cand_params, y=y, q=q, y_backcast=y_backcast)
    ll = compute_loss(y=y, f=f_init, q=q)

    # 3. Select the best parameters based on the M lowest loss values
    idx = np.argpartition(ll, kth=m_first_pass - 1)[:m_first_pass]
    best_params_first_pass = cand_params[:, idx]  # (n_params, m_first_pass)

    # 4. Define the objective function for the implementation. This function is
    #    responsible for computing the loss associated with a given beta vector.
    results: list[tuple[FloatArray, float]] = []
    objective = loss_func_factory(
        caviar_func=caviar_func, y=y, q=q, y_backcast=y_backcast
    )

    # 5. We perform the optimization routine starting from all retained vectors
    for beta0 in best_params_first_pass.T:
        beta = beta0.copy()
        loss = objective(beta)

        for i_iter in range(max_iter):
            logger.info("Loop %i/(%i)", i_iter + 1, max_iter)
            beta_old = beta.copy()
            loss_old = loss

            # 5.1. Start with a Nelder-Mead pass
            nm = minimize(
                objective,
                x0=beta,
                method="Nelder-Mead",
                options={
                    "maxiter": 10000,
                    "xatol": tol,
                    "fatol": tol,
                },
            )

            # 5.2. Refine with a Quasi-Newton algorithm
            qn = minimize(
                objective,
                x0=nm.x,
                method="BFGS",
                options={
                    "maxiter": 10000,
                    "gtol": tol,
                },
            )

            beta = cast(FloatArray, qn.x)
            loss = float(qn.fun)

            # 5.3. Determine whether to continue based on pre-defined tolerance
            param_change = np.max(np.abs(beta - beta_old))
            fun_change = abs(loss - loss_old)

            logger.info("Parameter change: %e", param_change)
            logger.info("Loss change: %e", fun_change)

            if param_change < tol and fun_change < tol:
                logger.info("Convergence reached after %i iterations!", i_iter + 1)
                break

        results.append((beta, loss))

    # Keep the beta associated with the lowest overall loss
    return min(results, key=lambda x: x[1])
