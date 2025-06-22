from colour import SpectralShape, MSDS_CMFS
import pandas as pd
import numpy as np
import jax
import colour
from scipy.integrate import cumulative_trapezoid
from colour.models import RGB_COLOURSPACE_ADOBE_RGB1998
from scipy.optimize import minimize
import jax.numpy as jnp
from jax import device_get
from jax import grad, hessian
from jax import jit, vmap
from scipy.optimize import OptimizeResult
from color import delta_E_CIE2000_jax, XYZ_to_Lab_jax
from util import cumulative_trapezoid_jax
from evosax.algorithms import DifferentialEvolution, DiffusionEvolution
from evaluate import plot_color_bar


N = 4
lambda_min = 380
lambda_max = 780
lambda_step = 5.0
U = 100000
spd_path = "spds/resampled_fluorescent.csv"

# Import color matching functions
cmfs = MSDS_CMFS['CIE 1931 2 Degree Standard Observer']
cmfs = cmfs.copy().align(SpectralShape(lambda_min, lambda_max, lambda_step))
lambdas = cmfs.wavelengths
x_bar, y_bar, z_bar = cmfs.values.T

# Objective function
def core(flat):
    raw = jnp.exp(flat.reshape((N, len(lambdas))))

    It_all = vmap(lambda row: jnp.trapezoid(row, lambdas))(raw)
    nonzero = It_all != 0.0

    # For rows where It_all == 0, we leave pdf as zero; otherwise normalize:
    pdfs = jnp.where(
        nonzero[:, None],
        raw / (It_all[:, None]),
        jnp.zeros_like(raw)
    )

    # Compute CDFs via cumulative trapezoid (assumed jax‐jit’d)
    cdfs = jnp.stack([
        cumulative_trapezoid_jax(pdfs[i], lambdas, initial=0.0)
        for i in range(N)
    ])

    xs = jnp.linspace(0, 1,
                      U)  # We can include the bounds 0,1 if we don't use the Hessian which is unstable at the bounds
    # Compute λ values by interpolating xs against each CDF
    lmb = jnp.vstack([
        jnp.interp(xs, cdfs[i], lambdas)
        for i in range(N)
    ])

    # Interpolate color matching functions at each λ
    h_x = jnp.vstack([
        jnp.interp(lmb[i], lambdas, x_bar)
        for i in range(N)
    ])
    h_y = jnp.vstack([
        jnp.interp(lmb[i], lambdas, y_bar)
        for i in range(N)
    ])
    h_z = jnp.vstack([
        jnp.interp(lmb[i], lambdas, z_bar)
        for i in range(N)
    ])

    # Interpolate illuminant contributions at each λ
    illum = jnp.vstack([
        jnp.interp(lmb[i], lambdas, I)
        for i in range(N)
    ])

    # Compute denominator: sum over all PDFs interpolated at each λ
    denom = jnp.stack([
        jnp.sum(
            jnp.vstack([
                jnp.interp(lmb[i], lambdas, pdfs[k])
                for k in range(N)
            ]),
            axis=0
        )
        for i in range(N)
    ])
    eps = 1e-7
    denom += eps
    # Compute integral sums along axis=0
    sum_x = jnp.sum(h_x * illum / denom, axis=0)
    sum_y = jnp.sum(h_y * illum / denom, axis=0)
    sum_z = jnp.sum(h_z * illum / denom, axis=0)

    # Convert to CIE L*a*b (vectorize via vmap)
    labs = vmap(
        lambda X, Y, Z: XYZ_to_Lab_jax(jnp.array([X, Y, Z]))
    )(sum_x, sum_y, sum_z)

    # Compute perceptual error (ΔE) via vmap
    dEs = vmap(
        lambda lab: delta_E_CIE2000_jax(lab, Lab_gt_ref)
    )(labs)
    # Final integral via trapezoidal rule
    return jnp.trapezoid(dEs, xs)

def compute_XYZ_ground_truth(path):
    df = pd.read_csv(path, comment='#')
    wavelengths = df['wavelength'].to_numpy()
    values = df[' intensity'].to_numpy()

    # Create spectral distribution
    sd = colour.SpectralDistribution(dict(zip(wavelengths, values)))

    # Convert to XYZ (CIE 1931 2° observer)
    xyz = colour.sd_to_XYZ(sd)
    return xyz

def load_spd(spd_path: str):
    df = pd.read_csv(spd_path, comment='#')
    wavelengths = df['wavelength'].to_numpy()
    values = df[' intensity'].to_numpy()

    if wavelengths[0] > lambda_min or wavelengths[-1] < lambda_max:
        raise RuntimeError(f'The SPD from {spd_path} does not fully cover the wavelength range {lambda_min}-{lambda_max}nm.')

    I = np.zeros((len(lambdas), ))
    w = lambda_min
    idx = 0
    while w <= lambda_max:
        I[idx] = np.interp(w, wavelengths, values)
        # print(w, I[idx])
        w += lambda_step
        idx += 1
    return I

I = load_spd(spd_path)
I += 1e-4       # Add tiny number so it's never zero, it messes up the gradients in the optimizer
x_gt, y_gt, z_gt = compute_XYZ_ground_truth(spd_path)
Lab_gt_ref = XYZ_to_Lab_jax(jnp.array([x_gt, y_gt, z_gt]))

def get_initial_densities():
    initial_densities = np.zeros((N, len(lambdas)))

    illuminant_integral = np.trapezoid(I, lambdas)
    q = I / illuminant_integral
    cdf = cumulative_trapezoid(q, lambdas, initial=0.0)

    ranges = []
    j = 0
    a = np.interp(j / N, cdf, lambdas)
    while j < N:
        b = np.interp((j + 1) / N, cdf, lambdas)
        ranges.append((a, b))
        a = b
        j += 1

    P0 = 1 / N
    for i in range(N):
        (a, b) = ranges[i]
        initial_densities[i] = np.where((a <= lambdas) & (lambdas <= b),
                                q / P0,
                                0)
    # Interpolation errors result in pdfs not integrating to exactly 1, so we normalize again.
    for i in range(N):
        I0 = np.trapezoid(initial_densities[i], lambdas)
        if I0 != 0:
            initial_densities[i] /= I0

    # Add a small uniform baseline to avoid starting with hard zeros
    # and then re-normalize the densities so their integrals are still 1.
    smoothing_factor = 1e-3
    initial_densities += smoothing_factor

    for i in range(N):
        integral = np.trapezoid(initial_densities[i], lambdas)
        initial_densities[i] /= integral
    return initial_densities

# Get initial densities
initial_densities = get_initial_densities()
epsilon = 1e-10
initial = np.log(np.maximum(initial_densities.ravel(), epsilon))

obj = jit(core)
grad_fn = jit(grad(core))
hess_fn = jit(hessian(core))

def scipy_obj(x_np: np.ndarray) -> float:
    x_jnp = jnp.array(x_np)
    loss_jnp = obj(x_jnp)
    return float(device_get(loss_jnp))

def scipy_grad(x_np: np.ndarray) -> np.ndarray:
    x_jnp = jnp.array(x_np)
    g_jnp = grad_fn(x_jnp)
    return np.array(device_get(g_jnp))

def scipy_hess(x_np: np.ndarray) -> np.ndarray:
    x_jnp = jnp.array(x_np)
    H_jnp = hess_fn(x_jnp)
    return np.array(device_get(H_jnp))

def step_fn(carry, key):
    state, params = carry
    key, ask_key, tell_key = jax.random.split(key, 3)
    population, state = strategy.ask(ask_key, state, params)
    fitness = vmap(obj)(population)
    state, metrics = strategy.tell(tell_key, population, fitness, state, params)
    return (state, params), metrics


def callback_every_10_iterations(xk, state: OptimizeResult):
    """
    A callback function that only prints its output every 10 iterations.
    """
    current_iteration = state.nit

    # Check if the iteration number is a multiple of 10
    if current_iteration % 10 == 0:
        current_fun_value = state.fun
        # You can also access other useful info from the state object
        # e.g., gradient norm, trust-region radius
        grad_norm = np.linalg.norm(state.grad)
        trust_radius = state.tr_radius

        print(
            f"Iter: {current_iteration:4d} | Obj: {current_fun_value:.6e} | Grad Norm: {grad_norm:.4e} | TR Radius: {trust_radius:.4e}")

population_size = 200
TOTAL_VARS = 324
iterations = 50
generations = 500

strategy = DiffusionEvolution(population_size=population_size, solution=initial)
params = strategy.default_params

# Start optimization
print("Starting optimization...")

key = jax.random.PRNGKey(0)

X = initial
obj_X = obj(X)

for i in range(iterations):
    print("ITERATION ", i)
    print("################################################")
    key, subkey = jax.random.split(key)
    initial_population = jax.random.uniform(
        subkey,
        shape=(population_size, TOTAL_VARS),
        minval=-10,
        maxval=10
    )

    initial_fitness = jax.vmap(obj)(initial_population)

    key, subkey = jax.random.split(key)
    state = strategy.init(subkey, initial_population, initial_fitness, params)

    keys = jax.random.split(key, generations)
    print("Starting global search....")
    (final_de_state, _), metrics = jax.lax.scan(step_fn, (state, params), keys)
    best_de_fitness = metrics["best_fitness"][-1]
    best_de_solution_raw = final_de_state.best_solution
    print(f"Global search found a candidate solution with fitness: {best_de_fitness:.6f}")
    print("-----------------------------------------------------------------------------")
    print("Starting optimization using Scipy...")

    x0 = best_de_solution_raw
    res = minimize(
        fun=scipy_obj,
        x0=x0,
        method='trust-constr',
        jac=scipy_grad,
        # hess=scipy_hess,
        callback=callback_every_10_iterations,
        options={'disp': True, 'maxiter': 10000, 'xtol': 1e-16, 'gtol': 1e-10, 'verbose': 3},
    )

    print(f"Optimizer finished with objective value: {res.fun}")
    if res.fun < obj_X:
        print(f"Value {res.fun} lower than best value so far {obj_X}.")
        X = res.x
        obj_X = res.fun
    else:
        print(f"Value {res.fun} higher than best value so far {obj_X}.")

    # print("Starting IPOPT optimization...")
    #
    # # jax_cons_hess_funcs = [
    # #     jit(jax.hessian(lambda x, i=i: cons_func(x)[i]))
    # #     for i in range(N)
    # # ]
    #
    # class IpoptProblemDefinition:
    #     def __init__(self, h):
    #         self.tri_r, self.tri_c = h
    #
    #     def objective(self, x):
    #         return float(device_get(obj(jnp.array(x))))
    #
    #     def gradient(self, x):
    #         return np.array(device_get(grad_fn(jnp.array(x))))
    #
    #     # def hessianstructure(self):
    #     #     return self.tri_r, self.tri_c  # length = n_vars*(n_vars+1)//2
    #     #
    #     # def hessian(self, x, lagrange, obj_factor):
    #     #     H_tot = obj_factor * hess_fn(jnp.array(x))
    #     #     # lower-triangle, flattened
    #     #     return np.asarray(H_tot)[self.tri_r, self.tri_c]
    #
    #     # def constraints(self, x):
    #         # return np.array(device_get(cons_func(jnp.array(x))))
    #
    #     # def jacobianstructure(self):
    #     #     return self.jac_r, self.jac_c  # length = n_cons * n_vars
    #     #
    #     # def jacobian(self, x):
    #     #     J = cons_jac_func(jnp.array(x))
    #     #     return np.asarray(J)[self.jac_r, self.jac_c]  # pick only needed entries
    #
    #     # def hessianstructure(self):
    #     #     return self.tri_r, self.tri_c  # length = n_vars*(n_vars+1)//2
    #     #
    #     # def hessian(self, x, lagrange, obj_factor):
    #     #     H_obj = obj_factor * hess_fn(jnp.array(x))
    #     #     H_cons = sum(lagrange[i] * h(jnp.array(x))
    #     #                  for i, h in enumerate(self.cons_hess_funcs))
    #     #     H_tot = H_obj + H_cons
    #     #     # lower-triangle, flattened
    #     #     return np.asarray(H_tot)[self.tri_r, self.tri_c]
    #
    # n_vars = N * len(lambdas)
    # n_cons = 0
    # x0 = best_de_solution_raw
    #
    # tri_r, tri_c = np.tril_indices(n_vars)  # Hessian (lower-triangle)
    #
    # problem_instance = IpoptProblemDefinition((tri_r, tri_c))
    #
    # nlp = Problem(
    #     n_vars,
    #     n_cons,
    #     problem_obj=problem_instance
    # )
    #
    # nlp.add_option('linear_solver', 'ma86')
    # nlp.add_option("hessian_approximation", "limited-memory")
    # nlp.add_option('limited_memory_max_history', 100)
    # nlp.add_option("mu_strategy", "adaptive")
    # nlp.add_option('max_iter', 1000)
    # nlp.add_option('print_level', 5)
    #
    # final_x, info = nlp.solve(x0)
    #
    # print(f"IPOPT finished with objective value: {info['obj_val']}")
    # if info['obj_val'] < obj_X:
    #     print(f"Value {info['obj_val']} lower than best value so far {obj_X}.")
    #     X = final_x
    #     obj_X = info['obj_val']
    # else:
    #     print(f"Value {info['obj_val']} higher than best value so far {obj_X}.")
    print("############################################################################")

X = initial

def write_to_file(filename: str = "output.csv"):
    """
    - flat_vec: 1D array-like of length N*M (e.g. Python list or NumPy array).
    - N, M: dimensions of the target matrix (rows=N, columns=M).
    - filename: path to the CSV file to create.

    Behavior:
      1. Converts flat_vec into a 1D NumPy array.
      2. Reshapes it into shape (N, M).
      3. Saves as CSV with M comma-separated values per row.
    """
    if X is None:
        raise RuntimeError("Optimization function has not been called yet.")
    # Convert to NumPy array
    flat_np = np.asarray(X)

    M = len(lambdas)
    # Validate length
    if flat_np.size != N * M:
        raise ValueError(f"Expected {N * M} elements, but got {flat_np.size}.")

    # Reshape to (N, M)
    mat = flat_np.reshape((N, M))

    It_all = np.array([np.trapezoid(row, lambdas) for row in mat])
    nonzero = It_all != 0.0

    # For rows where It_all == 0, we leave pdf as zero; otherwise normalize:
    pdfs = np.where(
        nonzero[:, None],
        mat / (It_all[:, None]),
        np.zeros_like(mat)
    )


    np.savetxt(filename, pdfs, delimiter=",", fmt="%.18e")
    print(f"Wrote output densities matrix of shape ({N}, {M}) to '{filename}'.")


pdfs = initial_densities
cdfs = np.stack([
    cumulative_trapezoid(pdfs[i], lambdas, initial=0.0)
    for i in range(N)
])

xs = np.linspace(0, 1.0, U)

lmb = np.vstack([
    np.interp(xs, cdfs[i], lambdas)
    for i in range(N)
])

h_x = np.vstack([
    np.interp(lmb[i], lambdas, x_bar)
    for i in range(N)
])
h_y = np.vstack([
    np.interp(lmb[i], lambdas, y_bar)
    for i in range(N)
])
h_z = np.vstack([
    np.interp(lmb[i], lambdas, z_bar)
    for i in range(N)
])

illum = np.vstack([
    np.interp(lmb[i], lambdas, I)
    for i in range(N)
])

denom = np.stack([
    np.sum(
        np.vstack([
            np.interp(lmb[i], lambdas, pdfs[k])
            for k in range(N)
        ]),
        axis=0
    )
    for i in range(N)
])
eps = 1e-7
denom = np.maximum(denom, eps)

sum_x = np.sum(h_x * illum / denom, axis=0)
sum_y = np.sum(h_y * illum / denom, axis=0)
sum_z = np.sum(h_z * illum / denom, axis=0)

xyz_array = np.stack((sum_x, sum_y, sum_z), axis=1)
rgb_array = np.apply_along_axis(
    lambda xyz: colour.XYZ_to_RGB(xyz, RGB_COLOURSPACE_ADOBE_RGB1998),
    axis=1,
    arr=xyz_array
)
plot_color_bar(rgb_array, "color_bars/initial_densities_fluorescent-2.png")




