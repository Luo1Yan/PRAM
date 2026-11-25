"""Register frozen patch grids using covariance on paired fitting frames."""

from math import isqrt

import numpy as np
import torch
from torch import Tensor


def _grid(value: Tensor) -> Tensor:
    if not isinstance(value, Tensor) or value.is_complex():
        raise ValueError("Expected a real feature tensor")
    if value.ndim == 3:
        n, cells, width = value.shape
        side = isqrt(cells)
        if side * side != cells:
            raise ValueError("A flattened spatial grid must be square")
        value = value.reshape(n, side, side, width)
    if value.ndim != 4 or min(value.shape) < 1:
        raise ValueError("Expected a nonempty [N, H, W, D] grid")
    return value


@torch.no_grad()
def estimate_spatial_offset(
    grid_x: Tensor, grid_y: Tensor, max_shift: int = 3,
    max_frames: int = 2000, seed: int = 0,
) -> tuple[tuple[int, int], Tensor]:
    """Match x[r, c] to y[r + dy, c + dx]. Map indices start at -max_shift."""
    x, y = _grid(grid_x), _grid(grid_y)
    if x.shape[:3] != y.shape[:3] or x.device != y.device:
        raise ValueError("Paired grids must share frame count, spatial shape and device")
    if type(max_shift) is not int or max_shift < 0:
        raise ValueError("max_shift must be a nonnegative integer")
    if type(max_frames) is not int or max_frames < 1:
        raise ValueError("max_frames must be a positive integer")

    n, h, w, da = x.shape
    db = y.shape[-1]
    chosen = np.random.default_rng(seed).choice(n, min(n, max_frames), replace=False)
    ids = torch.as_tensor(np.sort(chosen), device=x.device)
    batch = max(1, (1 << 21) // (h * w * (da + db)))
    blocks = ids.split(batch)

    def cell_mean(grid):
        total = torch.zeros(grid.shape[1:], device=x.device, dtype=torch.float64)
        for rows in blocks:
            total.add_(grid.index_select(0, rows).sum(0, dtype=torch.float64))
        return total / len(ids)

    mx, my = cell_mean(x), cell_mean(y)
    side = 2 * max_shift + 1
    strengths = mx.new_zeros(side, side)
    for dy in range(-min(max_shift, h - 1), min(max_shift, h - 1) + 1):
        ry = slice(max(0, -dy), h - max(0, dy))
        sy = slice(max(0, dy), h - max(0, -dy))
        for dx in range(-min(max_shift, w - 1), min(max_shift, w - 1) + 1):
            rx = slice(max(0, -dx), w - max(0, dx))
            sx = slice(max(0, dx), w - max(0, -dx))
            left, right = x[:, ry, rx], y[:, sy, sx]
            cross = mx.new_zeros(da, db)
            ca, cb = mx.new_zeros(da, da), my.new_zeros(db, db)
            for rows in blocks:
                a = (left.index_select(0, rows).double() - mx[ry, rx]).reshape(-1, da)
                b = (right.index_select(0, rows).double() - my[sy, sx]).reshape(-1, db)
                cross.addmm_(a.T, b)
                ca.addmm_(a.T, a)
                cb.addmm_(b.T, b)
            count = len(ids) * (h - abs(dy)) * (w - abs(dx))
            for matrix in (cross, ca, cb):
                matrix.div_(count)
            denominator = (ca.norm() * cb.norm()).sqrt().clamp_min(1e-12)
            strengths[dy + max_shift, dx + max_shift] = cross.norm() / denominator

    if strengths.max().item() == 0:
        return (0, 0), strengths
    row, col = divmod(int(strengths.argmax().item()), side)
    return (row - max_shift, col - max_shift), strengths


def align_spatial_grids(
    grid_a: Tensor, grid_p: Tensor, grid_b: Tensor,
    offset_ap: tuple[int, int], offset_pb: tuple[int, int], pool_size: int = 1,
) -> tuple[Tensor, Tensor, Tensor]:
    """Crop corresponding cells in A, P and B, then average complete square tiles."""
    if type(pool_size) is not int or pool_size < 1:
        raise ValueError("pool_size must be a positive integer")
    grids = tuple(_grid(grid) for grid in (grid_a, grid_p, grid_b))
    ay, ax = offset_ap
    by, bx = offset_pb
    offsets = ((0, 0), (ay, ax), (ay + by, ax + bx))
    top = max(-dy for dy, _ in offsets)
    left = max(-dx for _, dx in offsets)
    bottom = min(t.shape[1] - dy for t, (dy, _) in zip(grids, offsets))
    right = min(t.shape[2] - dx for t, (_, dx) in zip(grids, offsets))
    rows, cols = (bottom - top) // pool_size, (right - left) // pool_size
    if min(rows, cols) < 1:
        raise ValueError("The common overlap must contain at least one pooling tile")

    def crop(grid, shift):
        y, x = top + shift[0], left + shift[1]
        part = grid[:, y:y + rows * pool_size, x:x + cols * pool_size]
        if pool_size > 1:
            if not part.is_floating_point():
                part = part.float()
            part = part.reshape(len(part), rows, pool_size, cols, pool_size, part.size(-1))
            part = part.mean(dim=(2, 4))
        return part.flatten(1, 2).contiguous()

    return tuple(crop(grid, shift) for grid, shift in zip(grids, offsets))
