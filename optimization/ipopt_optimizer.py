from colour import SpectralShape, MSDS_CMFS
import pandas as pd
import numpy as np
import jax
import colour
from scipy.integrate import cumulative_trapezoid
from colour.models import RGB_COLOURSPACE_ADOBE_RGB1998
from colour.difference import delta_E_CIE2000
from scipy.optimize import minimize, LinearConstraint, Bounds
import jax.numpy as jnp
from jax import device_get
from jax import grad, hessian
from jax import jit, vmap, nn
from jax.debug import print as jax_print

from color import delta_E_CIE2000_jax, XYZ_to_Lab_jax
from util import cumulative_trapezoid_jax
from optimizer import is_matrix_of_shape
from cyipopt import Problem

class IPOPT_Optimizer:
    def __init__(self,
                 spd_path: str,
                 num_pdfs: int = 4,
                 lambda_min: float = 380,
                 lambda_max: float = 780,
                 lambda_step: float = 5,
                 u_samples: int = 100000,
                 debug: bool = False,
                 initial_densities = None):
        self.N = num_pdfs
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.lambda_step = lambda_step
        self.U = u_samples
        self.z_bar = None
        self.y_bar = None
        self.x_bar = None
        self.c1 = 0
        self.c2 = 0
        self.c3 = 0
        self.X = None
        self.lambdas = None
        self.initial_densities = initial_densities
        self.import_cmfs()
        self.I = self.load_spd(spd_path)
        self.I += 1e-4
        # self.x_gt, self.y_gt, self.z_gt = self.compute_XYZ_ground_truth(spd_path)
        #
        # self.Lab_gt_ref = XYZ_to_Lab_jax(jnp.array([self.x_gt, self.y_gt, self.z_gt]))
        self.compute_ground_truth()

        self.XYZ_gt_np = np.array([self.c1, self.c2, self.c3])
        self.scale_factor = self.compute_scale_factor(self.XYZ_gt_np)
        self.XYZ_gt_np *= self.scale_factor
        self.I *= self.scale_factor

        self.Lab_gt_ref = XYZ_to_Lab_jax(jnp.array(self.XYZ_gt_np))

        if initial_densities is None:
            self.get_initial_densities()

        def _core(flat):
            pdfs = flat.reshape((self.N, len(self.lambdas)))

            cdfs = jnp.stack([
                cumulative_trapezoid_jax(pdfs[i], self.lambdas, initial=0.0)
                for i in range(self.N)
            ])

            epsilon = 1e-4
            xs = jnp.linspace(0, 1.0, self.U)

            lmb = jnp.vstack([
                jnp.interp(xs, cdfs[i], self.lambdas)
                for i in range(self.N)
            ])

            h_x = jnp.vstack([
                jnp.interp(lmb[i], self.lambdas, self.x_bar)
                for i in range(self.N)
            ])
            h_y = jnp.vstack([
                jnp.interp(lmb[i], self.lambdas, self.y_bar)
                for i in range(self.N)
            ])
            h_z = jnp.vstack([
                jnp.interp(lmb[i], self.lambdas, self.z_bar)
                for i in range(self.N)
            ])

            illum = jnp.vstack([
                jnp.interp(lmb[i], self.lambdas, self.I)
                for i in range(self.N)
            ])

            denom = jnp.stack([
                jnp.sum(
                    jnp.vstack([
                        jnp.interp(lmb[i], self.lambdas, pdfs[k])
                        for k in range(self.N)
                    ]), axis=0
                )
                for i in range(self.N)
            ])
            denom = jnp.maximum(denom, 1e-8)

            sum_x = jnp.sum(h_x * illum / denom, axis=0)
            sum_y = jnp.sum(h_y * illum / denom, axis=0)
            sum_z = jnp.sum(h_z * illum / denom, axis=0)

            labs = vmap(lambda X, Y, Z: XYZ_to_Lab_jax(jnp.array([X, Y, Z])))(sum_x, sum_y, sum_z)

            dEs = vmap(lambda lab: delta_E_CIE2000_jax(lab, self.Lab_gt_ref))(labs)

            final_error = jnp.trapezoid(dEs, xs)

            return final_error

        def _constraints_func(x_flat):
            """
            Calculates the value of the equality constraints.
            We want the integral of each PDF to be 1.
            The constraint function should return a value that is zero when the
            constraint is satisfied, so we return `integral - 1`.
            """
            pdfs = x_flat.reshape((self.N, len(self.lambdas)))

            integrals = vmap(lambda pdf: jnp.trapezoid(pdf, self.lambdas))(pdfs)

            return integrals - 1.0

        if debug:
            self._o = _core
            self.grad_fn = grad(_core)
            self.hess_fn = hessian(_core)
            self.cons_func = _constraints_func
            self.cons_jac_func = jax.jacobian(_constraints_func)
        else:
            self._o = jit(_core)
            self.grad_fn = jit(grad(_core))
            self.hess_fn = jit(hessian(_core))
            self.cons_func = jit(_constraints_func)
            self.cons_jac_func = jit(jax.jacobian(_constraints_func))

    def compute_XYZ_ground_truth(self, path):
        df = pd.read_csv(path, comment='#')
        wavelengths = df['wavelength'].to_numpy()
        values = df[' intensity'].to_numpy()

        sd = colour.SpectralDistribution(dict(zip(wavelengths, values)))

        xyz = colour.sd_to_XYZ(sd)
        print(f"XYZ: {xyz}")
        return xyz

    # def test(self):
    #     rgb = colour.XYZ_to_RGB((self.x_gt, self.y_gt, self.z_gt), RGB_COLOURSPACE_ADOBE_RGB1998)
    #     return rgb

    def test(self):
        # x, y, z = self.compute_raw_xyz_values(self.I)
        rgb = colour.XYZ_to_RGB(self.XYZ_gt_np, RGB_COLOURSPACE_ADOBE_RGB1998)
        return rgb

    def import_cmfs(self):
        cmfs = MSDS_CMFS['CIE 1931 2 Degree Standard Observer']
        cmfs = cmfs.copy().align(SpectralShape(self.lambda_min, self.lambda_max, self.lambda_step))
        self.lambdas = cmfs.wavelengths
        self.x_bar, self.y_bar, self.z_bar = cmfs.values.T

    def compute_scale_factor(self, XYZ: np.ndarray):
        RGB = colour.XYZ_to_RGB(XYZ, RGB_COLOURSPACE_ADOBE_RGB1998)
        maxi = RGB.max()
        if maxi <= 100:
            return 1 / 100
        return 1 / maxi


    def load_spd(self, spd_path: str):
        df = pd.read_csv(spd_path, comment='#')
        wavelengths = df['wavelength'].to_numpy()
        values = df[' intensity'].to_numpy()

        I = np.zeros((len(self.lambdas), ))
        w = self.lambda_min
        idx = 0
        while w <= self.lambda_max:
            I[idx] = np.interp(w, wavelengths, values)
            # print(w, I[idx])
            w += self.lambda_step
            idx += 1
        return I


    def compute_raw_xyz_values(self, spectra):
        p1 = self.x_bar * spectra
        p2 = self.y_bar * spectra
        p3 = self.z_bar * spectra

        x = np.trapezoid(p1, self.lambdas)
        y = np.trapezoid(p2, self.lambdas)
        z = np.trapezoid(p3, self.lambdas)
        return x, y, z

    def compute_ground_truth(self):
        self.c1, self.c2, self.c3 = self.compute_raw_xyz_values(self.I)

    def set_initial_densities(self, densities):
        if not is_matrix_of_shape(densities, self.N, len(self.lambdas)):
            raise RuntimeError(f"Initial densities must be a matrix of shape {self.N} x {len(self.lambdas)}.")
        else:
            self.initial_densities = densities

    def get_initial_densities(self):
        self.initial_densities = np.zeros((self.N, len(self.lambdas)))

        illuminant_integral = np.trapezoid(self.I * (self.x_bar + self.y_bar + self.z_bar), self.lambdas)
        q = (self.I * (self.x_bar + self.y_bar + self.z_bar)) / illuminant_integral
        cdf = cumulative_trapezoid(q, self.lambdas, initial=0.0)

        ranges = []
        j = 0
        a = np.interp(j / self.N, cdf, self.lambdas)
        while j < self.N:
            b = np.interp((j + 1) / self.N, cdf, self.lambdas)
            ranges.append((a, b))
            a = b
            j += 1

        P0 = 1 / self.N
        for i in range(self.N):
            (a, b) = ranges[i]
            self.initial_densities[i] = np.where((a <= self.lambdas) & (self.lambdas <= b),
                                    q / P0,
                                    0)
        # Interpolation errors result in pdfs not integrating to exactly 1, so we normalize again.
        for i in range(self.N):
            I0 = np.trapezoid(self.initial_densities[i], self.lambdas)
            if I0 != 0:
                self.initial_densities[i] /= I0

        # Add a small uniform baseline to avoid starting with hard zeros
        # and then re-normalize the densities so their integrals are still 1.
        smoothing_factor = 1e-3
        self.initial_densities += smoothing_factor

        for i in range(self.N):
            integral = np.trapezoid(self.initial_densities[i], self.lambdas)
            self.initial_densities[i] /= integral


    def o(self, flattened_pdfs):
        return self._o(flattened_pdfs)

    def optimize(self):
        jax_cons_hess_funcs = [
            jit(jax.hessian(lambda x, i=i: self.cons_func(x)[i]))
            for i in range(self.N)
        ]
        class IpoptProblemDefinition:
            def __init__(self, optimizer_instance, cons_hess_funcs, j, h):
                self.optimizer = optimizer_instance
                self.cons_hess_funcs = cons_hess_funcs
                self.jac_r, self.jac_c = j
                self.tri_r, self.tri_c = h

            # @staticmethod
            # def _finite(name, arr):
            #     if not np.all(np.isfinite(arr)):
            #         raise FloatingPointError(f"{name} produced NaN/Inf")

            def objective(self, x):
                return float(device_get(self.optimizer._o(jnp.array(x))))

            def gradient(self, x):
                return np.array(device_get(self.optimizer.grad_fn(jnp.array(x))))

            def constraints(self, x):
                return np.array(device_get(self.optimizer.cons_func(jnp.array(x))))

            def jacobianstructure(self):
                return self.jac_r, self.jac_c  # length = n_cons * n_vars

            def jacobian(self, x):
                J = self.optimizer.cons_jac_func(jnp.array(x))
                return np.asarray(J)[self.jac_r, self.jac_c]  # pick only needed entries

            # def hessianstructure(self):
            #     return self.tri_r, self.tri_c  # length = n_vars*(n_vars+1)//2
            #
            # def hessian(self, x, lagrange, obj_factor):
            #     H_obj = obj_factor * self.optimizer.hess_fn(jnp.array(x))
            #     H_cons = sum(lagrange[i] * h(jnp.array(x))
            #                  for i, h in enumerate(self.cons_hess_funcs))
            #     H_tot = H_obj + H_cons
            #     # lower-triangle, flattened
            #     return np.asarray(H_tot)[self.tri_r, self.tri_c]

        n_vars = self.N * len(self.lambdas)
        n_cons = self.N
        x0 = self.initial_densities.ravel()

        tri_r, tri_c = np.tril_indices(n_vars)
        jac_rows = np.repeat(np.arange(n_cons), n_vars)
        jac_cols = np.tile(np.arange(n_vars), n_cons)

        lb = [0.0] * n_vars
        ub = [np.inf] * n_vars
        cl = [0.0] * n_cons
        cu = [0.0] * n_cons

        problem_instance = IpoptProblemDefinition(self, jax_cons_hess_funcs, (jac_rows, jac_cols), (tri_r, tri_c))

        nlp = Problem(
            n=n_vars,
            m=n_cons,
            problem_obj=problem_instance,
            lb=lb, ub=ub, cl=cl, cu=cu
        )


        nlp.add_option('linear_solver', 'ma86')
        nlp.add_option("hessian_approximation", "limited-memory")
        nlp.add_option('limited_memory_max_history', 100)
        nlp.add_option("nlp_scaling_method", "gradient-based")
        nlp.add_option("mu_strategy", "adaptive")
        nlp.add_option("barrier_tol_factor", 2.0)
        nlp.add_option("mu_min", 1e-9)
        nlp.add_option('max_soc', 25)
        nlp.add_option('watchdog_trial_iter_max', 20)
        nlp.add_option('acceptable_tol', 1e-4)
        nlp.add_option('max_iter', 5000)
        nlp.add_option('print_level', 5)
        nlp.add_option('tol', 1e-6)


        print('Optimizing with IPOPT...')


        final_x, info = nlp.solve(x0)

        self.X = final_x

        class OptimizeResult:
            def __init__(self, x, info):
                self.x = x
                self.fun = info['obj_val']
                self.success = (info['status'] == 0)
                self.message = info['status_msg']
                self.nit = info.get('iter', -1)
                self.keys = lambda: ['x', 'fun', 'success', 'message', 'nit']

        return OptimizeResult(final_x, info)

    def write_to_file(self, filename: str = "output.csv"):
        """
        - flat_vec: 1D array-like of length N*M (e.g. Python list or NumPy array).
        - N, M: dimensions of the target matrix (rows=N, columns=M).
        - filename: path to the CSV file to create.
        # Add very small value so none of them is zero

        Behavior:
          1. Converts flat_vec into a 1D NumPy array.
          2. Reshapes it into shape (N, M).
          3. Saves as CSV with M comma-separated values per row.
        """
        if self.X is None:
            raise RuntimeError("Optimization function has not been called yet.")
        # Convert to NumPy array
        flat_np = np.asarray(self.X)

        M = len(self.lambdas)
        # Validate length
        if flat_np.size != self.N * M:
            raise ValueError(f"Expected {self.N * M} elements, but got {flat_np.size}.")

        # Reshape to (N, M)
        mat = flat_np.reshape((self.N, M))

        # Write to CSV (one row per line, comma-delimited)
        # Adjust fmt as needed; here we use "%.18e" for high precision.
        np.savetxt(filename, mat, delimiter=",", fmt="%.18e")
        print(f"Wrote output densities matrix of shape ({self.N}, {M}) to '{filename}'.")

