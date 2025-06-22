from colour import SpectralShape, MSDS_CMFS
import pandas as pd
import numpy as np
import jax
import colour
from scipy.integrate import cumulative_trapezoid
from colour.models import RGB_COLOURSPACE_ADOBE_RGB1998
from colour.difference import delta_E_CIE2000
from scipy.optimize import minimize, LinearConstraint, Bounds, NonlinearConstraint
import jax.numpy as jnp
from jax import device_get
from jax import grad, hessian
from jax import jit, vmap
from scipy.optimize import OptimizeResult

from color import delta_E_CIE2000_jax, XYZ_to_Lab_jax
from util import cumulative_trapezoid_jax


def is_matrix_of_shape(x, N, M):
    try:
        a = np.asarray(x)
    except Exception:
        return False
    # must be 2-D and exactly (N, M)
    return (a.ndim == 2) and (a.shape == (N, M))

class Optimizer:
    PATH_STANDARD_ILLUMINANT = 'spds/2661-StandardIlluminant-D65-StandardIlluminant.csv'

    def __init__(self,
                 spd_path: str,
                 num_pdfs: int = 4,
                 lambda_min: float = 380,
                 lambda_max: float = 780,
                 lambda_step: float = 5,
                 u_samples: int = 10000):
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
        self.initial_densities = None
        self.import_cmfs()
        self.I = self.load_spd(spd_path)
        self.I += 1e-8
        # self.x_gt, self.y_gt, self.z_gt = self.compute_XYZ_ground_truth(spd_path)

        # self.compute_ground_truth()
        #
        # XYZ_gt_np = np.array([self.c1, self.c2, self.c3])
        # self.scale_factor = self.compute_scale_factor(XYZ_gt_np)
        # XYZ_gt_np *= self.scale_factor
        # self.I *= self.scale_factor
        #
        # self.Lab_gt_ref = XYZ_to_Lab_jax(jnp.array(XYZ_gt_np))

        # self.x_gt, self.y_gt, self.z_gt = self.compute_XYZ_ground_truth(spd_path)
        #
        # self.Lab_gt_ref = XYZ_to_Lab_jax(jnp.array([self.x_gt, self.y_gt, self.z_gt]))
        # print(self.x_gt, self.y_gt, self.z_gt)

        self.compute_ground_truth()

        self.XYZ_gt_np = np.array([self.c1, self.c2, self.c3])
        self.scale_factor = self.compute_scale_factor(self.XYZ_gt_np)
        self.XYZ_gt_np *= self.scale_factor
        self.I *= self.scale_factor

        self.Lab_gt_ref = XYZ_to_Lab_jax(jnp.array(self.XYZ_gt_np))
        print(self.XYZ_gt_np)

        self.get_initial_densities()

        def _core(flat):
            pdfs = flat.reshape((self.N, len(self.lambdas)))

            cdfs = jnp.stack([
                cumulative_trapezoid_jax(pdfs[i], self.lambdas, initial=0.0)
                for i in range(self.N)
            ])

            epsilon = 1e-5
            xs = jnp.linspace(epsilon, 1.0 - epsilon, self.U)

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
                    ]),
                    axis=0
                )
                for i in range(self.N)
            ])
            eps = 1e-7
            denom = jnp.maximum(denom, eps)

            sum_x = jnp.sum(h_x * illum / denom, axis=0)
            sum_y = jnp.sum(h_y * illum / denom, axis=0)
            sum_z = jnp.sum(h_z * illum / denom, axis=0)

            labs = vmap(
                lambda X, Y, Z: XYZ_to_Lab_jax(jnp.array([X, Y, Z]))
            )(sum_x, sum_y, sum_z)

            dEs = vmap(
                lambda lab: delta_E_CIE2000_jax(lab, self.Lab_gt_ref)
            )(labs)
            return jnp.trapezoid(dEs, xs)

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

        self._o = jit(_core)
        self.grad_fn = jit(grad(_core))
        self.hess_fn = jit(hessian(_core))
        self.cons_func = jit(_constraints_func)
        self.cons_jac_func = jit(jax.jacobian(_constraints_func))

    def test(self):
        # x, y, z = self.compute_raw_xyz_values(self.I)
        rgb = colour.XYZ_to_RGB(self.XYZ_gt_np, RGB_COLOURSPACE_ADOBE_RGB1998)
        return rgb

    # def test(self):
    #     # x, y, z = self.compute_raw_xyz_values(self.I)
    #     rgb = colour.XYZ_to_RGB(self.XYZ_gt_np, RGB_COLOURSPACE_ADOBE_RGB1998)
    #     return rgb

    # def test(self):
    #     rgb = colour.XYZ_to_RGB((self.x_gt, self.y_gt, self.z_gt), RGB_COLOURSPACE_ADOBE_RGB1998)
    #     return rgb

    def import_cmfs(self):
        cmfs = MSDS_CMFS['CIE 1931 2 Degree Standard Observer']
        cmfs = cmfs.copy().align(SpectralShape(self.lambda_min, self.lambda_max, self.lambda_step))
        self.lambdas = cmfs.wavelengths
        self.x_bar, self.y_bar, self.z_bar = cmfs.values.T

    # def compute_scale_factor(self, XYZ: np.ndarray):
    #     RGB = colour.XYZ_to_RGB(XYZ, RGB_COLOURSPACE_ADOBE_RGB1998)
    #     maxi = RGB.max()
    #     if maxi <= 1:
    #         return 1
    #     elif maxi <= 100:
    #         return 1 / 100
    #     return 1 / maxi

    def compute_scale_factor(self, XYZ: np.ndarray):
        RGB = colour.XYZ_to_RGB(XYZ, RGB_COLOURSPACE_ADOBE_RGB1998)
        maxi = RGB.max()
        if maxi <= 100:
            return 1 / 100
        return 1 / maxi

    def compute_XYZ_ground_truth(self, path):
        df = pd.read_csv(path, comment='#')
        wavelengths = df['wavelength'].to_numpy()
        values = df[' intensity'].to_numpy()

        sd = colour.SpectralDistribution(dict(zip(wavelengths, values)))

        xyz = colour.sd_to_XYZ(sd)
        return xyz

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
        self.densities = np.zeros((self.N, len(self.lambdas)))

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
            self.densities[i] = np.where((a <= self.lambdas) & (self.lambdas <= b),
                                    q / P0,
                                    0)
        # Interpolation errors result in pdfs not integrating to exactly 1, so we normalize again.
        for i in range(self.N):
            I0 = np.trapezoid(self.densities[i], self.lambdas)
            if I0 != 0:
                self.densities[i] /= I0


    def o(self, flattened_pdfs):
        return self._o(flattened_pdfs)

    def optimize(self):
        def scipy_obj(x_np: np.ndarray) -> float:
            x_jnp = jnp.array(x_np)
            loss_jnp = self._o(x_jnp)
            return float(device_get(loss_jnp))

        def scipy_grad(x_np: np.ndarray) -> np.ndarray:
            x_jnp = jnp.array(x_np)
            g_jnp = self.grad_fn(x_jnp)
            return np.array(device_get(g_jnp))

        def scipy_hess(x_np: np.ndarray) -> np.ndarray:
            x_jnp = jnp.array(x_np)
            H_jnp = self.hess_fn(x_jnp)
            return np.array(device_get(H_jnp))

        print('Optimizing...')

        x0 = self.densities.ravel()

        def callback_every_10_iterations(xk, state: OptimizeResult):
            """
            A callback function that only prints its output every 10 iterations.
            """
            current_iteration = state.nit

            if current_iteration % 10 == 0:
                current_fun_value = state.fun
                grad_norm = np.linalg.norm(state.grad)
                trust_radius = state.tr_radius

                print(
                    f"Iter: {current_iteration:4d} | Obj: {current_fun_value:.6e} | Grad Norm: {grad_norm:.4e} | TR Radius: {trust_radius:.4e}")

        equality_constraint = NonlinearConstraint(
            fun=lambda x: np.array(device_get(self.cons_func(jnp.array(x)))),
            jac=lambda x: np.array(device_get(self.cons_jac_func(jnp.array(x)))),
            lb=np.zeros(self.N),
            ub=np.zeros(self.N)
        )

        # cobyla_constraints = []
        # for i in range(self.N):
        #     # Constraint 1: self.cons_func(x)[i] >= 0
        #     cobyla_constraints.append({
        #         'type': 'ineq',
        #         'fun': lambda x, idx=i: np.array(device_get(self.cons_func(jnp.array(x))[idx]))
        #     })
        #
        #     # Constraint 2: -self.cons_func(x)[i] >= 0
        #     cobyla_constraints.append({
        #         'type': 'ineq',
        #         'fun': lambda x, idx=i: -np.array(device_get(self.cons_func(jnp.array(x))[idx]))
        #     })

        res = minimize(
            fun=scipy_obj,
            x0=self.densities.ravel(),
            method='trust-constr',
            jac=scipy_grad,
            hess=scipy_hess,
            bounds=Bounds(0, np.inf),
            callback=callback_every_10_iterations,
            constraints=[equality_constraint],
            options={'disp': True, 'maxiter': 3500} # , 'ftol': 1e-12, 'gtol': 1e-7, 'maxls': 100
        )

        self.X = res.x
        return res

    def write_to_file(self, filename: str = "output.csv"):
        """
        - flat_vec: 1D array-like of length N*M (e.g. Python list or NumPy array).
        - N, M: dimensions of the target matrix (rows=N, columns=M).
        - filename: path to the CSV file to create.

        Behavior:
          1. Converts flat_vec into a 1D NumPy array.
          2. Reshapes it into shape (N, M).
          3. Saves as CSV with M comma-separated values per row.
        """
        if self.X is None:
            raise RuntimeError("Optimization function has not been called yet.")
        flat_np = np.asarray(self.X)

        M = len(self.lambdas)
        if flat_np.size != self.N * M:
            raise ValueError(f"Expected {self.N * M} elements, but got {flat_np.size}.")

        mat = flat_np.reshape((self.N, M))

        It_all = np.array([np.trapezoid(row, self.lambdas) for row in mat])
        nonzero = It_all != 0.0

        pdfs = np.where(
            nonzero[:, None],
            mat / (It_all[:, None]),
            np.zeros_like(mat)
        )

        np.savetxt(filename, pdfs, delimiter=",", fmt="%.18e")
        print(f"Wrote output densities matrix of shape ({self.N}, {M}) to '{filename}'.")

