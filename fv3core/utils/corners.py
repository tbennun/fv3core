from gt4py import gtscript
from gt4py.gtscript import PARALLEL, computation, horizontal, interval, region

import fv3core._config as spec
from fv3core.decorators import gtstencil
from fv3core.utils.typing import FloatField


@gtscript.function
def fill_corners_2cells_mult_x(
    q: FloatField,
    q_corner: FloatField,
    sw_mult: float,
    se_mult: float,
    nw_mult: float,
    ne_mult: float,
):
    """
    Fills cell quantity q using corners from q_corner and multipliers in x-dir.
    """
    from __externals__ import i_end, i_start, j_end, j_start

    # Southwest
    with horizontal(region[i_start - 1, j_start - 1]):
        q = sw_mult * q_corner[0, 1, 0]
    with horizontal(region[i_start - 2, j_start - 1]):
        q = sw_mult * q_corner[1, 2, 0]

    # Southeast
    with horizontal(region[i_end + 1, j_start - 1]):
        q = se_mult * q_corner[0, 1, 0]
    with horizontal(region[i_end + 2, j_start - 1]):
        q = se_mult * q_corner[-1, 2, 0]

    # Northwest
    with horizontal(region[i_start - 1, j_end + 1]):
        q = nw_mult * q_corner[0, -1, 0]
    with horizontal(region[i_start - 2, j_end + 1]):
        q = nw_mult * q_corner[1, -2, 0]

    # Northeast
    with horizontal(region[i_end + 1, j_end + 1]):
        q = ne_mult * q_corner[0, -1, 0]
    with horizontal(region[i_end + 2, j_end + 1]):
        q = ne_mult * q_corner[-1, -2, 0]

    return q


@gtscript.function
def fill_corners_2cells_x(q: FloatField):
    """
    Fills cell quantity q in x-dir.
    """
    return fill_corners_2cells_mult_x(q, q, 1.0, 1.0, 1.0, 1.0)


@gtscript.function
def fill_corners_3cells_mult_x(
    q: FloatField,
    q_corner: FloatField,
    sw_mult: float,
    se_mult: float,
    nw_mult: float,
    ne_mult: float,
):
    """
    Fills cell quantity q using corners from q_corner and multipliers in x-dir.
    """
    from __externals__ import i_end, i_start, j_end, j_start

    q = fill_corners_2cells_mult_x(q, q_corner, sw_mult, se_mult, nw_mult, ne_mult)

    # Southwest
    with horizontal(region[i_start - 3, j_start - 1]):
        q = sw_mult * q_corner[2, 3, 0]

    # Southeast
    with horizontal(region[i_end + 3, j_start - 1]):
        q = se_mult * q_corner[-2, 3, 0]

    # Northwest
    with horizontal(region[i_start - 3, j_end + 1]):
        q = nw_mult * q_corner[2, -3, 0]

    # Northeast
    with horizontal(region[i_end + 3, j_end + 1]):
        q = ne_mult * q_corner[-2, -3, 0]

    return q


@gtscript.function
def fill_corners_2cells_mult_y(
    q: FloatField,
    q_corner: FloatField,
    sw_mult: float,
    se_mult: float,
    nw_mult: float,
    ne_mult: float,
):
    """
    Fills cell quantity q using corners from q_corner and multipliers in y-dir.
    """
    from __externals__ import i_end, i_start, j_end, j_start

    # Southwest
    with horizontal(region[i_start - 1, j_start - 1]):
        q = sw_mult * q_corner[1, 0, 0]
    with horizontal(region[i_start - 1, j_start - 2]):
        q = sw_mult * q_corner[2, 1, 0]

    # Southeast
    with horizontal(region[i_end + 1, j_start - 1]):
        q = se_mult * q_corner[-1, 0, 0]
    with horizontal(region[i_end + 1, j_start - 2]):
        q = se_mult * q_corner[-2, 1, 0]

    # Northwest
    with horizontal(region[i_start - 1, j_end + 1]):
        q = nw_mult * q_corner[1, 0, 0]
    with horizontal(region[i_start - 1, j_end + 2]):
        q = nw_mult * q_corner[2, -1, 0]

    # Northeast
    with horizontal(region[i_end + 1, j_end + 1]):
        q = ne_mult * q_corner[-1, 0, 0]
    with horizontal(region[i_end + 1, j_end + 2]):
        q = ne_mult * q_corner[-2, -1, 0]

    return q


@gtscript.function
def fill_corners_2cells_y(q: FloatField):
    """
    Fills cell quantity q in y-dir.
    """
    return fill_corners_2cells_mult_y(q, q, 1.0, 1.0, 1.0, 1.0)


@gtscript.function
def fill_corners_3cells_mult_y(
    q: FloatField,
    q_corner: FloatField,
    sw_mult: float,
    se_mult: float,
    nw_mult: float,
    ne_mult: float,
):
    """
    Fills cell quantity q using corners from q_corner and multipliers in y-dir.
    """
    from __externals__ import i_end, i_start, j_end, j_start

    q = fill_corners_2cells_mult_y(q, q_corner, sw_mult, se_mult, nw_mult, ne_mult)

    # Southwest
    with horizontal(region[i_start - 1, j_start - 3]):
        q = sw_mult * q_corner[3, 2, 0]

    # Southeast
    with horizontal(region[i_end + 1, j_start - 3]):
        q = se_mult * q_corner[-3, 2, 0]

    # Northwest
    with horizontal(region[i_start - 1, j_end + 3]):
        q = nw_mult * q_corner[3, -2, 0]

    # Northeast
    with horizontal(region[i_end + 1, j_end + 3]):
        q = ne_mult * q_corner[-3, -2, 0]

    return q


def fill_corners_cells(q: FloatField, direction: str, num_fill: int = 2):
    """
    Fill corners of q from Python.

    Corresponds to fill4corners in Fortran.

    Args:
        q (inout): Cell field
        direction: Direction to fill. Either "x" or "y".
        num_fill: Number of indices to fill
    """

    def definition(q: FloatField):
        from __externals__ import func

        with computation(PARALLEL), interval(...):
            q = func(q, q, 1.0, 1.0, 1.0, 1.0)

    if num_fill not in (2, 3):
        raise ValueError("Only supports 2 <= num_fill <= 3")

    if direction == "x":
        func = (
            fill_corners_2cells_mult_x if num_fill == 2 else fill_corners_3cells_mult_x
        )
        stencil = gtstencil(
            definition=definition,
            externals={"func": func},
        )
    elif direction == "y":
        func = (
            fill_corners_2cells_mult_y if num_fill == 2 else fill_corners_3cells_mult_y
        )
        stencil = gtstencil(
            definition=definition,
            externals={"func": func},
        )
    else:
        raise ValueError("Direction not recognized. Specify either x or y")

    extent = 3
    origin = (spec.grid.is_ - extent, spec.grid.js - extent, 0)
    domain = (spec.grid.nic + 2 * extent, spec.grid.njc + 2 * extent, q.shape[2])
    stencil(q, origin=origin, domain=domain)


def copy_sw_corner(q, direction, grid, kslice):
    for j in range(grid.js - grid.halo, grid.js):
        for i in range(grid.is_ - grid.halo, grid.is_):
            if direction == "x":
                q[i, j, kslice] = q[j, grid.is_ - i + 2, kslice]
            if direction == "y":
                q[i, j, kslice] = q[grid.js - j + 2, i, kslice]


def copy_se_corner(q, direction, grid, kslice):
    for j in range(grid.js - grid.halo, grid.js):
        for i in range(grid.ie + 1, grid.ie + grid.halo + 1):
            if direction == "x":
                q[i, j, kslice] = q[grid.je + 1 - j + 2, i - grid.ie + 2, kslice]
            if direction == "y":
                q[i, j, kslice] = q[grid.je + j - 2, grid.ie + 1 - i + 2, kslice]


def copy_ne_corner(q, direction, grid, kslice):
    for j in range(grid.je + 1, grid.je + grid.halo + 1):
        for i in range(grid.ie + 1, grid.ie + grid.halo + 1):
            if direction == "x":
                q[i, j, kslice] = q[j, 2 * (grid.ie + 1) - 1 - i, kslice]
            if direction == "y":
                q[i, j, kslice] = q[2 * (grid.je + 1) - 1 - j, i, kslice]


def copy_nw_corner(q, direction, grid, kslice):
    for j in range(grid.je + 1, grid.je + grid.halo + 1):
        for i in range(grid.is_ - grid.halo, grid.is_):
            if direction == "x":
                q[i, j, kslice] = q[grid.je + 1 - j + 2, i - 2 + grid.ie, kslice]
            if direction == "y":
                q[i, j, kslice] = q[j + 2 - grid.ie, grid.je + 1 - i + 2, kslice]


# can't actually be a stencil because offsets are variable
def copy_corners(q, direction, grid, kslice=slice(0, None)):
    if grid.sw_corner:
        copy_sw_corner(q, direction, grid, kslice)
    if grid.se_corner:
        copy_se_corner(q, direction, grid, kslice)
    if grid.ne_corner:
        copy_ne_corner(q, direction, grid, kslice)
    if grid.nw_corner:
        copy_nw_corner(q, direction, grid, kslice)


@gtscript.function
def fill_corners_bgrid_x(q: FloatField, q_corner: FloatField):
    from __externals__ import i_end, i_start, j_end, j_start

    # sw and se corner
    with horizontal(region[i_start - 1, j_start - 1], region[i_end + 2, j_start - 1]):
        q = q_corner[0, 2, 0]
    with horizontal(region[i_start - 1, j_start - 2], region[i_end + 3, j_start - 1]):
        q = q_corner[-1, 3, 0]
    with horizontal(region[i_start - 1, j_start - 3], region[i_end + 4, j_start - 1]):
        q = q_corner[-2, 4, 0]
    with horizontal(region[i_start - 2, j_start - 1], region[i_end + 2, j_start - 2]):
        q = q_corner[1, 3, 0]
    with horizontal(region[i_start - 2, j_start - 2], region[i_end + 3, j_start - 2]):
        q = q_corner[0, 4, 0]
    with horizontal(region[i_start - 2, j_start - 3], region[i_end + 4, j_start - 2]):
        q = q_corner[-1, 5, 0]
    with horizontal(region[i_start - 3, j_start - 1], region[i_end + 2, j_start - 3]):
        q = q_corner[2, 4, 0]
    with horizontal(region[i_start - 3, j_start - 2], region[i_end + 3, j_start - 3]):
        q = q_corner[1, 5, 0]
    with horizontal(region[i_start - 3, j_start - 3], region[i_end + 4, j_start - 3]):
        q = q_corner[0, 6, 0]
    # nw and ne corner
    with horizontal(region[i_start - 1, j_end + 2], region[i_end + 2, j_end + 2]):
        q = q_corner[0, -2, 0]
    with horizontal(region[i_start - 1, j_end + 3], region[i_end + 3, j_end + 2]):
        q = q_corner[-1, -3, 0]
    with horizontal(region[i_start - 1, j_end + 4], region[i_end + 4, j_end + 2]):
        q = q_corner[-2, -4, 0]
    with horizontal(region[i_start - 2, j_end + 2], region[i_end + 2, j_end + 3]):
        q = q_corner[1, -3, 0]
    with horizontal(region[i_start - 2, j_end + 3], region[i_end + 3, j_end + 3]):
        q = q_corner[0, -4, 0]
    with horizontal(region[i_start - 2, j_end + 4], region[i_end + 4, j_end + 3]):
        q = q_corner[-1, -5, 0]
    with horizontal(region[i_start - 3, j_end + 2], region[i_end + 2, j_end + 4]):
        q = q_corner[2, -4, 0]
    with horizontal(region[i_start - 3, j_end + 3], region[i_end + 3, j_end + 4]):
        q = q_corner[1, -5, 0]
    with horizontal(region[i_start - 3, j_end + 4], region[i_end + 4, j_end + 4]):
        q = q_corner[0, -6, 0]
    return q


@gtscript.function
def fill_corners_bgrid_y(q: FloatField, q_corner: FloatField):
    from __externals__ import i_end, i_start, j_end, j_start

    # sw and nw corners
    with horizontal(region[i_start - 1, j_start - 1], region[i_start - 1, j_end + 2]):
        q = q_corner[2, 0, 0]
    with horizontal(region[i_start - 1, j_start - 2], region[i_start - 2, j_end + 2]):
        q = q_corner[3, 1, 0]
    with horizontal(region[i_start - 1, j_start - 3], region[i_start - 3, j_end + 2]):
        q = q_corner[4, 2, 0]
    with horizontal(region[i_start - 2, j_start - 1], region[i_start - 1, j_end + 3]):
        q = q_corner[3, -1, 0]
    with horizontal(region[i_start - 2, j_start - 2], region[i_start - 2, j_end + 3]):
        q = q_corner[4, 0, 0]
    with horizontal(region[i_start - 2, j_start - 3], region[i_start - 3, j_end + 3]):
        q = q_corner[5, 1, 0]
    with horizontal(region[i_start - 3, j_start - 1], region[i_start - 1, j_end + 4]):
        q = q_corner[4, -2, 0]
    with horizontal(region[i_start - 3, j_start - 2], region[i_start - 2, j_end + 4]):
        q = q_corner[5, -1, 0]
    with horizontal(region[i_start - 3, j_start - 3], region[i_start - 3, j_end + 4]):
        q = q_corner[6, 0, 0]
    # se and ne corners
    with horizontal(region[i_end + 2, j_start - 1], region[i_end + 2, j_end + 2]):
        q = q_corner[-2, 0, 0]
    with horizontal(region[i_end + 2, j_start - 2], region[i_end + 3, j_end + 2]):
        q = q_corner[-3, 1, 0]
    with horizontal(region[i_end + 2, j_start - 3], region[i_end + 4, j_end + 2]):
        q = q_corner[-4, 2, 0]
    with horizontal(region[i_end + 3, j_start - 1], region[i_end + 2, j_end + 3]):
        q = q_corner[-3, -1, 0]
    with horizontal(region[i_end + 3, j_start - 2], region[i_end + 3, j_end + 3]):
        q = q_corner[-4, 0, 0]
    with horizontal(region[i_end + 3, j_start - 3], region[i_end + 4, j_end + 3]):
        q = q_corner[-5, 1, 0]
    with horizontal(region[i_end + 4, j_start - 1], region[i_end + 2, j_end + 4]):
        q = q_corner[-4, -2, 0]
    with horizontal(region[i_end + 4, j_start - 2], region[i_end + 3, j_end + 4]):
        q = q_corner[-5, -1, 0]
    with horizontal(region[i_end + 4, j_start - 3], region[i_end + 4, j_end + 4]):
        q = q_corner[-6, 0, 0]

    return q


# Filling the AGrid corners happens in init_grid, not in the dycore
# may be pragmatic to leave in python initially
def fill_sw_corner_2d_agrid(q, i, j, direction, grid):
    if direction == "x":
        q[grid.is_ - i, grid.js - j, :] = q[grid.is_ - j, i, :]
    if direction == "y":
        q[grid.is_ - j, grid.js - i, :] = q[i, grid.js - j, :]


def fill_nw_corner_2d_agrid(q, i, j, direction, grid):
    if direction == "x":
        q[grid.is_ - i, grid.je + j, :] = q[grid.is_ - j, grid.je - i + 1, :]
    if direction == "y":
        q[grid.is_ - j, grid.je + i, :] = q[i, grid.je + j, :]


def fill_se_corner_2d_agrid(q, i, j, direction, grid):
    if direction == "x":
        q[grid.ie + i, grid.js - j, :] = q[grid.ie + j, i, :]
    if direction == "y":
        q[grid.ie + j, grid.js - i, :] = q[grid.ie - i + 1, grid.js - j, :]


def fill_ne_corner_2d_agrid(q, i, j, direction, grid, mysign=1.0):
    if direction == "x":
        q[grid.ie + i, grid.je + j, :] = q[grid.ie + j, grid.je - i + 1, :]
    if direction == "y":
        q[grid.ie + j, grid.je + i, :] = q[grid.ie - i + 1, grid.je + j, :]


def fill_corners_2d_agrid(q, grid, gridtype, direction="x"):
    for i in range(1, 1 + grid.halo):
        for j in range(1, 1 + grid.halo):
            if grid.sw_corner:
                fill_sw_corner_2d_agrid(q, i, j, direction, grid)
            if grid.nw_corner:
                fill_nw_corner_2d_agrid(q, i, j, direction, grid)
            if grid.se_corner:
                fill_se_corner_2d_agrid(q, i, j, direction, grid)
            if grid.ne_corner:
                fill_ne_corner_2d_agrid(q, i, j, direction, grid)


@gtscript.function
def fill_corners_dgrid_fn(x: FloatField, y: FloatField, mysign: float):
    from __externals__ import i_end, i_start, j_end, j_start

    # sw corner
    with horizontal(region[i_start - 1, j_start - 1]):
        x = mysign * y[0, 1, 0]
    with horizontal(region[i_start - 1, j_start - 1]):
        y = mysign * x[1, 0, 0]
    with horizontal(region[i_start - 1, j_start - 2]):
        x = mysign * y[-1, 2, 0]
    with horizontal(region[i_start - 1, j_start - 2]):
        y = mysign * x[2, 1, 0]
    with horizontal(region[i_start - 1, j_start - 3]):
        x = mysign * y[-2, 3, 0]
    with horizontal(region[i_start - 1, j_start - 3]):
        y = mysign * x[3, 2, 0]
    with horizontal(region[i_start - 2, j_start - 1]):
        x = mysign * y[1, 2, 0]
    with horizontal(region[i_start - 2, j_start - 1]):
        y = mysign * x[2, -1, 0]
    with horizontal(region[i_start - 2, j_start - 2]):
        x = mysign * y[0, 3, 0]
    with horizontal(region[i_start - 2, j_start - 2]):
        y = mysign * x[3, 0, 0]
    with horizontal(region[i_start - 2, j_start - 3]):
        x = mysign * y[-1, 4, 0]
    with horizontal(region[i_start - 2, j_start - 3]):
        y = mysign * x[4, 1, 0]
    with horizontal(region[i_start - 3, j_start - 1]):
        x = mysign * y[2, 3, 0]
    with horizontal(region[i_start - 3, j_start - 1]):
        y = mysign * x[3, -2, 0]
    with horizontal(region[i_start - 3, j_start - 2]):
        x = mysign * y[1, 4, 0]
    with horizontal(region[i_start - 3, j_start - 2]):
        y = mysign * x[4, -1, 0]
    with horizontal(region[i_start - 3, j_start - 3]):
        x = mysign * y[0, 5, 0]
    with horizontal(region[i_start - 3, j_start - 3]):
        y = mysign * x[5, 0, 0]
    # ne corner
    with horizontal(region[i_end + 1, j_end + 2]):
        x = mysign * y[1, -2, 0]
    with horizontal(region[i_end + 2, j_end + 1]):
        y = mysign * x[-2, 1, 0]
    with horizontal(region[i_end + 1, j_end + 3]):
        x = mysign * y[2, -3, 0]
    with horizontal(region[i_end + 2, j_end + 2]):
        y = mysign * x[-3, 0, 0]
    with horizontal(region[i_end + 1, j_end + 4]):
        x = mysign * y[3, -4, 0]
    with horizontal(region[i_end + 2, j_end + 3]):
        y = mysign * x[-4, -1, 0]
    with horizontal(region[i_end + 2, j_end + 2]):
        x = mysign * y[0, -3, 0]
    with horizontal(region[i_end + 3, j_end + 1]):
        y = mysign * x[-3, 2, 0]
    with horizontal(region[i_end + 2, j_end + 3]):
        x = mysign * y[1, -4, 0]
    with horizontal(region[i_end + 3, j_end + 2]):
        y = mysign * x[-4, 1, 0]
    with horizontal(region[i_end + 2, j_end + 4]):
        x = mysign * y[2, -5, 0]
    with horizontal(region[i_end + 3, j_end + 3]):
        y = mysign * x[-5, 0, 0]
    with horizontal(region[i_end + 3, j_end + 2]):
        x = mysign * y[-1, -4, 0]
    with horizontal(region[i_end + 4, j_end + 1]):
        y = mysign * x[-4, 3, 0]
    with horizontal(region[i_end + 3, j_end + 3]):
        x = mysign * y[0, -5, 0]
    with horizontal(region[i_end + 4, j_end + 2]):
        y = mysign * x[-5, 2, 0]
    with horizontal(region[i_end + 3, j_end + 4]):
        x = mysign * y[1, -6, 0]
    with horizontal(region[i_end + 4, j_end + 3]):
        y = mysign * x[-6, 1, 0]
    # nw corner
    with horizontal(region[i_start - 1, j_end + 2]):
        x = y[0, -2, 0]
    with horizontal(region[i_start - 1, j_end + 1]):
        y = x[1, 1, 0]
    with horizontal(region[i_start - 1, j_end + 3]):
        x = y[-1, -3, 0]
    with horizontal(region[i_start - 1, j_end + 2]):
        y = x[2, 0, 0]
    with horizontal(region[i_start - 1, j_end + 4]):
        x = y[-2, -4, 0]
    with horizontal(region[i_start - 1, j_end + 3]):
        y = x[3, -1, 0]
    with horizontal(region[i_start - 2, j_end + 2]):
        x = y[1, -3, 0]
    with horizontal(region[i_start - 2, j_end + 1]):
        y = x[2, 2, 0]
    with horizontal(region[i_start - 2, j_end + 3]):
        x = y[0, -4, 0]
    with horizontal(region[i_start - 2, j_end + 2]):
        y = x[3, 1, 0]
    with horizontal(region[i_start - 2, j_end + 4]):
        x = y[-1, -5, 0]
    with horizontal(region[i_start - 2, j_end + 3]):
        y = x[4, 0, 0]
    with horizontal(region[i_start - 3, j_end + 2]):
        x = y[2, -4, 0]
    with horizontal(region[i_start - 3, j_end + 1]):
        y = x[3, 3, 0]
    with horizontal(region[i_start - 3, j_end + 3]):
        x = y[1, -5, 0]
    with horizontal(region[i_start - 3, j_end + 2]):
        y = x[4, 2, 0]
    with horizontal(region[i_start - 3, j_end + 4]):
        x = y[0, -6, 0]
    with horizontal(region[i_start - 3, j_end + 3]):
        y = x[5, 1, 0]
    # se corner
    with horizontal(region[i_end + 1, j_start - 1]):
        x = y[1, 1, 0]
    with horizontal(region[i_end + 2, j_start - 1]):
        y = x[-2, 0, 0]
    with horizontal(region[i_end + 1, j_start - 2]):
        x = y[2, 2, 0]
    with horizontal(region[i_end + 2, j_start - 2]):
        y = x[-3, 1, 0]
    with horizontal(region[i_end + 1, j_start - 3]):
        x = y[3, 3, 0]
    with horizontal(region[i_end + 2, j_start - 3]):
        y = x[-4, 2, 0]
    with horizontal(region[i_end + 2, j_start - 1]):
        x = y[0, 2, 0]
    with horizontal(region[i_end + 3, j_start - 1]):
        y = x[-3, -1, 0]
    with horizontal(region[i_end + 2, j_start - 2]):
        x = y[1, 3, 0]
    with horizontal(region[i_end + 3, j_start - 2]):
        y = x[-4, 0, 0]
    with horizontal(region[i_end + 2, j_start - 3]):
        x = y[2, 4, 0]
    with horizontal(region[i_end + 3, j_start - 3]):
        y = x[-5, 1, 0]
    with horizontal(region[i_end + 3, j_start - 1]):
        x = y[-1, 3, 0]
    with horizontal(region[i_end + 4, j_start - 1]):
        y = x[-4, -2, 0]
    with horizontal(region[i_end + 3, j_start - 2]):
        x = y[0, 4, 0]
    with horizontal(region[i_end + 4, j_start - 2]):
        y = x[-5, -1, 0]
    with horizontal(region[i_end + 3, j_start - 3]):
        x = y[1, 5, 0]
    with horizontal(region[i_end + 4, j_start - 3]):
        y = x[-6, 0, 0]
    return x, y


@gtstencil
def fill_corners_dgrid_stencil(u: FloatField, v: FloatField, mysign: float):
    with computation(PARALLEL), interval(...):
        (
            u,
            v,
        ) = fill_corners_dgrid_fn(u, v, mysign)


def corner_ke(ke, u, v, ut, vt, i, j, dt, offsets, vsign):
    dt6 = dt / 6.0
    ke[i, j, :] = dt6 * (
        (ut[i, j, :] + ut[i, j - 1, :]) * u[i + offsets["io1"], j, :]
        + (vt[i, j, :] + vt[i - 1, j, :]) * v[i, j + offsets["jo1"], :]
        + (ut[i, j + offsets["jo1"], :] + vsign * vt[i + offsets["io1"], j, :])
        * u[i + offsets["io2"], j, :]
    )


def fix_corner_ke(ke, u, v, ut, vt, dt, grid):
    if grid.sw_corner:
        offsets = {"io1": 0, "jo1": 0, "io2": -1}
        corner_ke(ke, u, v, ut, vt, grid.is_, grid.js, dt, offsets, 1)
    if grid.se_corner:
        offsets = {"io1": -1, "jo1": 0, "io2": 0}
        corner_ke(ke, u, v, ut, vt, grid.ie + 1, grid.js, dt, offsets, -1)
    if grid.ne_corner:
        offsets = {"io1": -1, "jo1": -1, "io2": 0}
        corner_ke(ke, u, v, ut, vt, grid.ie + 1, grid.je + 1, dt, offsets, 1)
    if grid.nw_corner:
        offsets = {"io1": 0, "jo1": -1, "io2": -1}
        corner_ke(ke, u, v, ut, vt, grid.is_, grid.je + 1, dt, offsets, -1)
