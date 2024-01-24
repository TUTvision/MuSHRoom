"""
borrowed from RadianceMapping https://github.com/seanywang0408/RadianceMapping/blob/main/rasterizer/rasterizer.py#L13
"""
import torch

from . import _C


def get_bin_size(img_size):
    if img_size <= 256:
        return 16
    elif img_size <= 512:
        return 32
    else:
        return 64


def rasterize(xyz_ndc, hw, radius):
    """
    This function implements rasterization.
    Args:
        xyz_ndc: ndc coordinates of point cloud
        hw: height and width of rasterization
        radius: radius of points
    Output:
        idx: buffer of points index
        zbuf: buffer of points depth
    """

    N = xyz_ndc.size(0)
    points_per_pixel = 1
    bin_size = get_bin_size(hw[0])
    cloud_to_packed_first_idx = torch.tensor([0], device=xyz_ndc.device)
    num_points_per_cloud = torch.tensor([N], device=xyz_ndc.device)
    radius = radius * torch.ones([N], device=xyz_ndc.device)

    idx, zbuf, _ = _C._rasterize(
        xyz_ndc,
        cloud_to_packed_first_idx,
        num_points_per_cloud,
        hw,
        radius,
        points_per_pixel,
        bin_size,
        N,
    )

    return idx, zbuf
