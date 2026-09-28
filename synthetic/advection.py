"""Extrapolation nowcast of a moving storm cell: the model explained in the toy experiment.

The input is the current observation q of a transported field (for instance a rain
rate) on an n x n periodic grid; the observation one step earlier, q_prev, is fixed
context. The forecast is made as in extrapolation (Lagrangian persistence) nowcasting:

1. Motion estimation by block matching: the velocity m, in pixels per step, is the
   shift d maximising the cross-correlation C(d) = sum_x q(x) q_prev(x - d),
   searched on a grid of 1 / upsample pixel.
2. Extrapolation: q is advected by m and diffused for `lead` steps,
       dq/dt + m . grad q = nu lap q,
   solved exactly in Fourier space.
3. Target: mean of the forecast over a box, the region of interest (ROI).

Step 1 selects the motion with an argmax, like max pooling or hard attention
routing: the selected motion is piecewise constant in q, so gradients only flow
through the extrapolation. The input gradient of the target is therefore the ROI
carried back to the current time by the adjoint of steps 2-3, i.e. the region the
forecast air comes from, whose position is known. Input noise changes the selected
motion and displaces this gradient by lead x (motion error) without deforming it.
Everything is in float64 on the CPU, so that results are reproducible.
"""

import math
from dataclasses import dataclass

import torch

DTYPE = torch.float64


@dataclass
class Event:
    """A Gaussian storm cell of radius ``radius`` at ``source`` = (x, y), moving at ``velocity``."""

    source: tuple  # position at the current time, in pixels
    velocity: tuple  # (u, v), in pixels per step
    radius: float  # in pixels


class Nowcast:
    """Block-matching extrapolation nowcast on an n x n periodic grid (x along columns, y along rows)."""

    def __init__(self, n=64, lead=24, diffusion=0.15, roi_half_width=3, upsample=100):
        self.n, self.lead, self.diffusion = n, lead, diffusion
        self.roi_half_width, self.upsample = roi_half_width, upsample
        k = 2 * math.pi * torch.fft.fftfreq(n, dtype=DTYPE)  # angular wavenumbers, radians per pixel
        self.kx, self.ky = k[None, :], k[:, None]
        coords = torch.arange(n, dtype=DTYPE)
        self.y, self.x = torch.meshgrid(coords, coords, indexing="ij")

    # -- model -------------------------------------------------------------------

    @torch.no_grad()
    def estimate_motion(self, q, q_prev):
        """Shift (dx, dy) maximising sum_x q(x) q_prev(x - d), for q of shape (B, n, n): (B, 2).

        The integer maximum of the FFT cross-correlation is refined on a grid of
        1 / upsample pixel within one pixel of it, by evaluating the Fourier
        interpolant of the cross-correlation with matrix products (Guizar-Sicairos
        et al., Opt. Lett. 2008).
        """
        cross = torch.fft.fft2(q) * torch.conj(torch.fft.fft2(q_prev))  # (B, n, n), [ky, kx]
        peak = torch.fft.ifft2(cross).real.flatten(1).argmax(dim=1)
        coarse = torch.stack([peak % self.n, peak // self.n], dim=1).to(DTYPE)
        coarse = (coarse + self.n / 2) % self.n - self.n / 2
        steps = torch.arange(-self.upsample, self.upsample + 1, dtype=DTYPE) / self.upsample
        dx, dy = coarse[:, :1] + steps, coarse[:, 1:] + steps  # candidate shifts, (B, S)
        ex = torch.exp(1j * dx[..., None] * self.kx[0])  # (B, S, n)
        ey = torch.exp(1j * dy[..., None] * self.ky[:, 0])
        corr = (ey @ cross @ ex.transpose(1, 2)).real  # (B, S, S), [dy, dx]
        best = corr.flatten(1).argmax(dim=1)
        rows = torch.arange(len(q))
        return torch.stack([dx[rows, best % len(steps)], dy[rows, best // len(steps)]], dim=1)

    def extrapolate(self, q, velocity):
        """Solution at t = lead of dq/dt + m . grad q = nu lap q, for q (B, n, n) and m (B, 2)."""
        u, v = velocity[:, 0, None, None], velocity[:, 1, None, None]
        advection = self.kx * u + self.ky * v
        diffusion = self.diffusion * (self.kx**2 + self.ky**2)
        propagator = torch.exp(-self.lead * (1j * advection + diffusion))
        return torch.fft.ifft2(torch.fft.fft2(q) * propagator).real

    def target(self, event):
        """Scalar forecast f(q): mean of the extrapolated field over the ROI, batched over dim 0."""
        q_prev = self.observation(event, time=-1)
        weights = self.roi(event)

        def f(q):
            forecast = self.extrapolate(q, self.estimate_motion(q, q_prev))
            return (forecast * weights).sum(dim=(-2, -1))

        return f

    # -- geometry ----------------------------------------------------------------

    def offsets(self, point):
        """Offsets (dx, dy) from ``point`` to every pixel, with the minimum-image convention."""
        half = self.n / 2
        return (self.x - point[0] + half) % self.n - half, (self.y - point[1] + half) % self.n - half

    def observation(self, event, time=0):
        """Observed field ``time`` steps from now (negative: past): the cell carried by its velocity."""
        centre = [s + time * v for s, v in zip(event.source, event.velocity)]
        dx, dy = self.offsets(centre)
        return torch.exp(-(dx**2 + dy**2) / (2 * event.radius**2))

    def roi_centre(self, event):
        """Pixel nearest to the position of the cell at t = lead."""
        return tuple(round(s + self.lead * v) % self.n for s, v in zip(event.source, event.velocity))

    def roi(self, event):
        """Averaging weights over the (2 h + 1)^2 pixel box centred on ``roi_centre``."""
        dx, dy = self.offsets(self.roi_centre(event))
        box = ((dx.abs() <= self.roi_half_width) & (dy.abs() <= self.roi_half_width)).to(DTYPE)
        return box / box.sum()

    def source_centre(self, event):
        """Ground truth: the ROI centre carried back to the current time by the true velocity."""
        return tuple(c - self.lead * v for c, v in zip(self.roi_centre(event), event.velocity))
