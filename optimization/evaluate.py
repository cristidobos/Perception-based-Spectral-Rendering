import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import cumulative_trapezoid
import colour
from colour.models import RGB_COLOURSPACE_ADOBE_RGB1998

from optimizer import Optimizer
from unconstrained_optimizer import UnconstrainedOptimizer


def plot_single_pixel(pixel_value, filename):
    """
    Plots a single pixel value. Pixel value must be an ndarray of shape (3, ).
    """
    fig, ax = plt.subplots(1, 1)
    ax.imshow(np.array([[pixel_value]]))

    ax.axis('off')
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    fig.savefig(filename, bbox_inches='tight', pad_inches=0, dpi=1)
    print(f"Saved single pixel to {filename}")

def plot_color_bar(color_bar, filename):
    """
    Plots a vector of pixels.
    color_bar must be an ndarray of shape (N, 3).
    """
    fig, ax = plt.subplots(figsize=(6, 1))
    ax.imshow(np.array([color_bar]), aspect='auto')

    ax.get_yaxis().set_visible(False)
    ax.set_xticks([0, len(color_bar) - 1])
    ax.set_xticklabels(['Start', 'End'])

    plt.savefig(filename)
    print(f"Saved color bar to {filename}")

def evaluate_optimization(optimizer, gt_filename, color_bar_filename, densities_filename):
    gt = optimizer.test()
    pdfs = np.loadtxt(densities_filename, delimiter=',')

    cdfs = np.stack([
        cumulative_trapezoid(pdfs[i], optimizer.lambdas, initial=0.0)
        for i in range(optimizer.N)
    ])

    xs = np.linspace(0, 1.0, optimizer.U)

    lmb = np.vstack([
        np.interp(xs, cdfs[i], optimizer.lambdas)
        for i in range(optimizer.N)
    ])

    h_x = np.vstack([
        np.interp(lmb[i], optimizer.lambdas, optimizer.x_bar)
        for i in range(optimizer.N)
    ])
    h_y = np.vstack([
        np.interp(lmb[i], optimizer.lambdas, optimizer.y_bar)
        for i in range(optimizer.N)
    ])
    h_z = np.vstack([
        np.interp(lmb[i], optimizer.lambdas, optimizer.z_bar)
        for i in range(optimizer.N)
    ])

    illum = np.vstack([
        np.interp(lmb[i], optimizer.lambdas, optimizer.I)
        for i in range(optimizer.N)
    ])

    denom = np.stack([
        np.sum(
            np.vstack([
                np.interp(lmb[i], optimizer.lambdas, pdfs[k])
                for k in range(optimizer.N)
            ]),
            axis=0
        )
        for i in range(optimizer.N)
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
    plot_single_pixel(gt, gt_filename)
    plot_color_bar(rgb_array, color_bar_filename)

