"""Spatial metrics of attribution maps on a periodic grid.

Every map is first turned into the measure |A| / sum |A|, as WassersteinGrad does.
Positions are (x, y) in pixels, x along columns and y along rows, and distances
use the minimum-image convention of the periodic domain.
"""

import math

import torch


def measure(a):
    return a.abs() / a.abs().sum()


def _offsets(shape, point):
    """Offsets (dx, dy) from ``point`` to every pixel of a periodic grid of the given shape."""
    n_y, n_x = shape
    y, x = torch.meshgrid(torch.arange(n_y, dtype=torch.float64), torch.arange(n_x, dtype=torch.float64),
                          indexing="ij")
    return (x - point[0] + n_x / 2) % n_x - n_x / 2, (y - point[1] + n_y / 2) % n_y - n_y / 2


def distance(p, q, shape):
    """Minimum-image distance between the points p and q."""
    n_y, n_x = shape
    return math.hypot((p[0] - q[0] + n_x / 2) % n_x - n_x / 2, (p[1] - q[1] + n_y / 2) % n_y - n_y / 2)


def centroid(a):
    """Centre of mass of |A| on the periodic grid (circular mean along each axis)."""
    mu = measure(a)
    centre = []
    for axis, n in ((-1, a.shape[-1]), (-2, a.shape[-2])):  # x, then y
        angle = 2 * math.pi * torch.arange(n, dtype=mu.dtype) / n
        marginal = mu.sum(dim=-2 if axis == -1 else -1)
        mean = math.atan2(float((marginal * angle.sin()).sum()), float((marginal * angle.cos()).sum()))
        centre.append(mean / (2 * math.pi) * n % n)
    return tuple(centre)


def centroid_error(a, point):
    """Distance from the centroid of |A| to ``point``, in pixels."""
    return distance(centroid(a), point, a.shape)


def peak(a):
    """Position (x, y) of the maximum of |A|."""
    index = int(a.abs().flatten().argmax())
    return index % a.shape[-1], index // a.shape[-1]


def peak_error(a, point):
    """Distance from the maximum of |A| to ``point``, in pixels."""
    return distance(peak(a), point, a.shape)


def rms_radius(a, point):
    """Root-mean-square distance of the attribution mass to ``point``, in pixels.

    Its square is the squared centroid error plus the spatial variance of |A|: it
    grows when the map is displaced from ``point`` and when it is spread out.
    """
    dx, dy = _offsets(a.shape, point)
    return float((measure(a) * (dx**2 + dy**2)).sum().sqrt())


def gini(a):
    """Gini index of |A| (the Sparsity metric of the paper): 0 for a uniform map, 1 for a single pixel."""
    values = a.abs().flatten().sort().values
    n = len(values)
    rank = torch.arange(1, n + 1, dtype=values.dtype)
    return float(((2 * rank - n - 1) * values).sum() / (n * values.sum()))
