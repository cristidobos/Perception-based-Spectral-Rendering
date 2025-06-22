import jax.numpy as jnp
from jax import jit, custom_jvp

def to_domain_100_jax(Lab):
    # assume Lab in [0,1]/[-1,1] scale → scale to [0,100]/[-100,100]
    L, a, b = Lab
    return L * 100, a * 100, b * 100

def tsplit_jax(Matrix): # Generic name
    # split last axis into three arrays
    # Ensure it handles inputs that might not be just (3,)
    if Matrix.ndim == 0: # Should not happen for color values
        return Matrix, Matrix, Matrix # Or raise error
    if Matrix.shape[-1] != 3:
        raise ValueError(f"Last dimension must be 3 for tsplit_jax, got shape {Matrix.shape}")
    return Matrix[..., 0], Matrix[..., 1], Matrix[..., 2]


@jit
def delta_E_CIE2000_jax(Lab1, Lab2, textiles: bool = False):
    # 1) scale & split
    L1, a1, b1 = tsplit_jax(Lab1)
    L2, a2, b2 = tsplit_jax(Lab2)

    kL = 2.0 if textiles else 1.0
    kC = 1.0
    kH = 1.0

    # 2) chroma
    C1 = jnp.hypot(a1, b1)
    C2 = jnp.hypot(a2, b2)
    Cbar = 0.5 * (C1 + C2)
    Cbar7 = Cbar**7

    G = 0.5 * (1 - jnp.sqrt(Cbar7 / (Cbar7 + 25.0**7)))

    ap1 = (1 + G) * a1
    ap2 = (1 + G) * a2

    Cp1 = jnp.hypot(ap1, b1)
    Cp2 = jnp.hypot(ap2, b2)

    hp1 = jnp.where((b1 == 0) & (ap1 == 0),
                    0.0,
                    (jnp.degrees(stable_arctan2(b1, ap1)) % 360.0))  # <--- USE STABLE ARCTAN2
    hp2 = jnp.where((b2 == 0) & (ap2 == 0),
                    0.0,
                    (jnp.degrees(stable_arctan2(b2, ap2)) % 360.0))  # <--- USE STABLE ARCTAN2

    dLp = L2 - L1
    dCp = Cp2 - Cp1
    hp2s1 = hp2 - hp1
    Cp1m2 = Cp1 * Cp2
    dhp = jnp.select(
        [
            Cp1m2 == 0.0,
            jnp.fabs(hp2s1) <= 180,
            hp2s1 > 180,
            hp2s1 < -180
        ],
        [
            0,
            hp2s1,
            hp2s1 - 360,
            hp2s1 + 360
        ]
    )

    dhp = 2 * jnp.sqrt(Cp1m2) * jnp.sin(jnp.deg2rad(dhp / 2))

    L_bar_p = (L1 + L2) / 2
    C_bar_p = (Cp1 + Cp2) / 2

    a_h_p_1_s_2 = jnp.fabs(hp1 - hp2)
    h_p_1_a_2 = hp1 + hp2
    h_bar_p = jnp.select(
        [
            Cp1m2 == 0,
            a_h_p_1_s_2 <= 180,
            jnp.logical_and(a_h_p_1_s_2 > 180, h_p_1_a_2 < 360),
            jnp.logical_and(a_h_p_1_s_2 > 180, h_p_1_a_2 >= 360),
        ],
        [
            h_p_1_a_2,
            h_p_1_a_2 / 2,
            (h_p_1_a_2 + 360) / 2,
            (h_p_1_a_2 - 360) / 2,
        ],
    )

    T = (
            1
            - 0.17 * jnp.cos(jnp.deg2rad(h_bar_p - 30))
            + 0.24 * jnp.cos(jnp.deg2rad(2 * h_bar_p))
            + 0.32 * jnp.cos(jnp.deg2rad(3 * h_bar_p + 6))
            - 0.20 * jnp.cos(jnp.deg2rad(4 * h_bar_p - 63))
    )

    delta_theta = 30 * jnp.exp(-(((h_bar_p - 275) / 25) ** 2))

    C_bar_p_7 = C_bar_p ** 7
    R_C = 2 * jnp.sqrt(
        C_bar_p_7 / (C_bar_p_7 + jnp.array(25.0) ** 7)
    )

    L_bar_p_2 = (L_bar_p - 50) ** 2
    S_L = 1 + ((0.015 * L_bar_p_2) / jnp.sqrt(20 + L_bar_p_2))

    S_C = 1 + 0.045 * C_bar_p

    S_H = 1 + 0.015 * C_bar_p * T

    R_T = -jnp.sin(jnp.deg2rad(2 * delta_theta)) * R_C
    argument = ((dLp / (kL * S_L)) ** 2
                + (dCp / (kC * S_C)) ** 2
                + (dhp / (kH * S_H)) ** 2
                + R_T * (dCp / (kC * S_C)) * (dhp / (kH * S_H)))

    safe_argument = jnp.maximum(argument, 1e-10)
    d_E = jnp.sqrt(safe_argument)

    return jnp.float64(d_E)

def tstack_jax(L, a, b):
    """Stack three (...,) arrays into one (...,3) array."""
    return jnp.stack([L, a, b], axis=-1)

def to_domain_1_jax(XYZ):
    """
    Input XYZ in [0,1] scale is assumed;
    if your inputs are [0,100] you’d divide by 100 here.
    """
    X, Y, Z = XYZ
    return X / 100, Y / 100, Z / 100

def from_range_100_jax(Lab):
    """
    Convert Lab in [0,100]/[-100,100] back to [0,1]/[-1,1].
    """
    return Lab / jnp.array([100.0, 100.0, 100.0])

def xy_to_xyY_jax(xy):
    """Given illuminant xy, return [x, y, Y=1.0]"""
    x, y = xy
    return jnp.array([x, y, 1.0])

def xyY_to_XYZ_jax(xyY):
    """Convert CIE xyY → XYZ."""
    x, y, Y = xyY
    X = x * (Y / y)
    Z = (1 - x - y) * (Y / y)
    return jnp.array([X, Y, Z])

@jit
def _safe_cuberoot(x):
    eps = 1e-24
    return jnp.sign(x) * jnp.power(jnp.abs(x) + eps, 1.0 / 3.0)

@jit
def intermediate_lightness_function_CIE1976_jax(t):
    e = 216.0 / 24389.0
    κ = 24389.0 / 27.0
    return jnp.where(t > e,
                     stable_cbrt(t),
                     (κ * t + 16.0) / 116.0)

@jit
def XYZ_to_Lab_jax(XYZ, illuminant_xy=jnp.array([0.3127, 0.3290])):
    """
    XYZ:      array shape (...,3) in [0,1]
    illuminant_xy: 2-vector, e.g. D65 = [0.3127, 0.3290]
    returns: Lab in [0,1] scale, shape (...,3)
    """
    X, Y, Z    = tsplit_jax(XYZ)
    white_xyY  = xy_to_xyY_jax(illuminant_xy)
    Xn, Yn, Zn = tsplit_jax(xyY_to_XYZ_jax(white_xyY))

    fX = intermediate_lightness_function_CIE1976_jax(X / Xn)
    fY = intermediate_lightness_function_CIE1976_jax(Y / Yn)
    fZ = intermediate_lightness_function_CIE1976_jax(Z / Zn)

    L = 116.0 * fY - 16.0
    a = 500.0 * (fX - fY)
    b = 200.0 * (fY - fZ)

    return tstack_jax(L, a, b)

@custom_jvp
def stable_cbrt(x):
    """
    Numerically stable cube root function that is safe for gradient-based optimization.
    The gradient is clipped to prevent it from exploding near zero.
    """
    # Add a small epsilon for the forward pass to avoid NaN at x=0 if x has a zero-valued tangent.
    # The main stabilization happens in the jvp rule.
    return jnp.cbrt(x + 1e-24)


@stable_cbrt.defjvp
def stable_cbrt_jvp(primals, tangents):
    """
    Defines the custom Jacobian-vector product (i.e., the derivative).
    """
    (x,) = primals
    (t,) = tangents

    y = stable_cbrt(x)

    grad_clip_threshold = 1e-12
    safe_x = jnp.maximum(jnp.abs(x), grad_clip_threshold)

    grad = (1.0 / 3.0) * jnp.power(safe_x, -2.0 / 3.0)

    return y, grad * t

@custom_jvp
def stable_arctan2(y, x):
    """
    A numerically stable arctan2 with a "clipped" gradient for optimization.
    It returns the true angle but prevents its derivative from exploding when
    both x and y are close to zero.
    """
    return jnp.arctan2(y, x)

@stable_arctan2.defjvp
def stable_arctan2_jvp(primals, tangents):
    y, x = primals
    ty, tx = tangents

    res = stable_arctan2(y, x)

    denom = jnp.maximum(x**2 + y**2, 1e-12)

    grad_x = -y / denom
    grad_y =  x / denom
    tangent_out = grad_y * ty + grad_x * tx

    return res, tangent_out