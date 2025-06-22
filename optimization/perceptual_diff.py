import imageio.v2 as imageio
import numpy as np
import colour
from colour.difference import delta_E_CIE2000
import OpenEXR
import Imath


def read_exr_with_openexr(filepath: str) -> np.ndarray:
    """Reads an EXR file using the robust OpenEXR library."""
    exr_file = OpenEXR.InputFile(filepath)

    header = exr_file.header()
    dw = header['dataWindow']
    size = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)

    if not ('R' in header['channels'] and 'G' in header['channels'] and 'B' in header['channels']):
        raise ValueError(f"EXR file '{filepath}' does not contain R, G, B channels.")

    float_type = Imath.PixelType(Imath.PixelType.FLOAT)
    r_str = exr_file.channel('R', float_type)
    g_str = exr_file.channel('G', float_type)
    b_str = exr_file.channel('B', float_type)

    r = np.frombuffer(r_str, dtype=np.float32).reshape(size)
    g = np.frombuffer(g_str, dtype=np.float32).reshape(size)
    b = np.frombuffer(b_str, dtype=np.float32).reshape(size)

    rgb = np.stack([r, g, b], axis=-1)

    exr_file.close()
    return rgb, header

def perceptual_difference_sum(image_path1: str, image_path2: str) -> float:
    """
    Computes the sum of the perceptual difference between two EXR images
    using the CIE2000 metric.

    Args:
        image_path1: The file path to the first EXR image.
        image_path2: The file path to the second EXR image.

    Returns:
        The sum of the CIE2000 perceptual difference for all corresponding
        pixels.
    """
    print(f"Reading {image_path1}...")
    image1_rgb, _ = read_exr_with_openexr(image_path1)
    print(f"Reading {image_path2}...")
    image2_rgb, _ = read_exr_with_openexr(image_path2)

    if image1_rgb.shape != image2_rgb.shape:
        raise ValueError("Input images must have the same dimensions.")

    xyz1 = colour.sRGB_to_XYZ(image1_rgb)
    xyz2 = colour.sRGB_to_XYZ(image2_rgb)

    image1_lab = colour.XYZ_to_Lab(xyz1)
    image2_lab = colour.XYZ_to_Lab(xyz2)

    delta_e_2000 = delta_E_CIE2000(image1_lab, image2_lab)

    return np.average(delta_e_2000), np.median(delta_e_2000)

if __name__ == '__main__':
    their_avg, their_median = perceptual_difference_sum('renders/cornell-box-theirs-fluorescent.exr', 'renders/cornell-box-gt-fluorescent.exr')
    our_avg, our_median = perceptual_difference_sum('renders/cornell-box-ours-fluorescent-2.exr', 'renders/cornell-box-gt-fluorescent.exr')

    print(f"Their average difference: {their_avg}")
    print(f"Our average difference: {our_avg}")
    print(f"Their median difference: {their_median}")
    print(f"Our median difference: {our_median}")