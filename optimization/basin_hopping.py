from colour import SpectralShape, MSDS_CMFS
import pandas as pd
import numpy as np
import jax
import colour
from scipy.integrate import cumulative_trapezoid
from colour.models import RGB_COLOURSPACE_ADOBE_RGB1998
import jax.numpy as jnp
from jax import device_get
from jax import grad, hessian
from jax import jit, vmap

from color import delta_E_CIE2000_jax, XYZ_to_Lab_jax
from util import cumulative_trapezoid_jax
from scipy.optimize import OptimizeResult, basinhopping



def is_matrix_of_shape(x, N, M):
    try:
        a = np.asarray(x)
    except Exception:
        return False
    # must be 2-D and exactly (N, M)
    return (a.ndim == 2) and (a.shape == (N, M))

class BasinHoppingOptimizer:
    def __init__(self,
                 spd_path: str,
                 num_pdfs: int = 4,
                 lambda_min: float = 380,
                 lambda_max: float = 780,
                 lambda_step: float = 5,
                 u_samples: int = 100000):
        self.N = num_pdfs
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.lambda_step = lambda_step
        self.U = u_samples
        self.z_bar = None
        self.y_bar = None
        self.x_bar = None
        self.X = None
        self.lambdas = None
        self.initial_densities = None
        self.import_cmfs()
        self.I = self.load_spd(spd_path)
        self.I += 1e-4
        self.x_gt, self.y_gt, self.z_gt = self.compute_XYZ_ground_truth(spd_path)

        self.Lab_gt_ref = XYZ_to_Lab_jax(jnp.array([self.x_gt, self.y_gt, self.z_gt]))

        self.get_initial_densities()

        def _core(flat):
            raw = jnp.exp(flat.reshape((self.N, len(self.lambdas))))

            It_all = vmap(lambda row: jnp.trapezoid(row, self.lambdas))(raw)
            nonzero = It_all != 0.0

            pdfs = jnp.where(
                nonzero[:, None],
                raw / (It_all[:, None]),
                jnp.zeros_like(raw)
            )

            cdfs = jnp.stack([
                cumulative_trapezoid_jax(pdfs[i], self.lambdas, initial=0.0)
                for i in range(self.N)
            ])

            xs = jnp.linspace(0, 1, self.U)
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
            denom += eps
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

        self._o = jit(_core)
        self.grad_fn = jit(grad(_core))
        self.hess_fn = jit(hessian(_core))

    def compute_XYZ_ground_truth(self, path):
        df = pd.read_csv(path, comment='#')
        wavelengths = df['wavelength'].to_numpy()
        values = df[' intensity'].to_numpy()

        sd = colour.SpectralDistribution(dict(zip(wavelengths, values)))

        xyz = colour.sd_to_XYZ(sd)
        print(f"XYZ: {xyz}")
        return xyz

    def test(self):
        rgb = colour.XYZ_to_RGB((self.x_gt, self.y_gt, self.z_gt), RGB_COLOURSPACE_ADOBE_RGB1998)
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

        if wavelengths[0] > self.lambda_min or wavelengths[-1] < self.lambda_max:
            raise RuntimeError(f'The SPD from {spd_path} does not fully cover the wavelength range {self.lambda_min}-{self.lambda_max}nm.')

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

        target = self.I * (self.x_bar + self.y_bar + self.z_bar)
        illuminant_integral = np.trapezoid(target, self.lambdas)
        q = target / illuminant_integral
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

        def callback_every_10_iterations(xk, state: OptimizeResult):
            current_iteration = state.nit
            if current_iteration % 10 == 0:
                fun_value = state.fun
                grad_norm = np.linalg.norm(state.grad)
                print(f"  [Local] Iter: {current_iteration:4d} | Obj: {fun_value:.6e} | Grad Norm: {grad_norm:.4e}")

        global_step_counter = [0]
        def log_global_step(x, f, accept):
            global_step_counter[0] += 1
            status = "Accepted" if accept else "Rejected"
            print(
                f"\n--- Global Step: {global_step_counter[0]:3d} | New Best Found: {status:10s} | Global Min: {f:.6e} ---")

        print('Optimizing...')

        minimizer_kwargs = {
            "method": "trust-constr",
            "jac": scipy_grad,
            "hess": scipy_hess,
            "callback": callback_every_10_iterations,
            "options": {
                'disp': False,
                'maxiter': 200,
                'xtol': 1e-16,
                'gtol': 1e-10,
                'verbose': 3
            }
        }

        epsilon = 1e-10
        x0_unconstrained = np.log(np.maximum(self.densities.ravel(), epsilon))

        res = basinhopping(
            func=scipy_obj,
            x0=x0_unconstrained,
            niter=100,
            minimizer_kwargs=minimizer_kwargs,
            callback=log_global_step,
            disp=True,
            stepsize=50.0
        )

        self.X = np.exp(res.x)
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
        # Validate length
        if flat_np.size != self.N * M:
            raise ValueError(f"Expected {self.N * M} elements, but got {flat_np.size}.")

        mat = flat_np.reshape((self.N, M))

        It_all = np.array([np.trapezoid(row, self.lambdas) for row in mat])
        nonzero = It_all != 0.0

        # For rows where It_all == 0, we leave pdf as zero; otherwise normalize:
        pdfs = np.where(
            nonzero[:, None],
            mat / (It_all[:, None]),
            np.zeros_like(mat)
        )


        # Write to CSV
        np.savetxt(filename, pdfs, delimiter=",", fmt="%.18e")
        print(f"Wrote output densities matrix of shape ({self.N}, {M}) to '{filename}'.")

