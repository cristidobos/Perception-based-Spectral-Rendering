import jax.numpy as jnp
from jax import grad, hessian, jit


@jit
def cumulative_trapezoid_jax(y: jnp.ndarray, x: jnp.ndarray, initial: float = 0.0) -> jnp.ndarray:
    """
    Compute the cumulative integral of `y` with respect to `x` using the trapezoidal rule.
    - y: 1D jnp.ndarray of function values.
    - x: 1D jnp.ndarray of coordinates (must be same length as y).
    - initial: scalar to prepend as the zeroth‐element of the integral.

    Returns a jnp.ndarray of length len(y), where
    result[0] == initial, and
    result[i] = integral from x[0] to x[i] of y via trapezoidal rule.
    """
    dx = x[1:] - x[:-1]

    trapezoids = (y[:-1] + y[1:]) * 0.5 * dx

    cumsum_traps = jnp.cumsum(trapezoids)

    init = jnp.array(initial, dtype=cumsum_traps.dtype)
    return jnp.concatenate([init[jnp.newaxis], cumsum_traps])