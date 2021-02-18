from typing import Optional

import gt4py.gtscript as gtscript
from gt4py.gtscript import PARALLEL, computation, interval

import fv3core._config as spec
import fv3core.stencils.a2b_ord4 as a2b_ord4
import fv3core.stencils.basic_operations as basic
import fv3core.utils.corners as corners
from fv3core.decorators import gtstencil
from fv3core.stencils.basic_operations import copy_stencil
from fv3core.utils.typing import FloatField, FloatFieldIJ


@gtstencil()
def ptc_main(
    u: FloatField,
    va: FloatField,
    cosa_v: FloatFieldIJ,
    sina_v: FloatFieldIJ,
    dyc: FloatFieldIJ,
    ptc: FloatField,
):
    with computation(PARALLEL), interval(...):
        ptc = (u - 0.5 * (va[0, -1, 0] + va) * cosa_v) * dyc * sina_v


@gtstencil()
def ptc_y_edge(
    u: FloatField,
    vc: FloatField,
    dyc: FloatFieldIJ,
    sin_sg4: FloatFieldIJ,
    sin_sg2: FloatFieldIJ,
    ptc: FloatField,
):
    with computation(PARALLEL), interval(...):
        ptc = u * dyc * sin_sg4[0, -1] if vc > 0 else u * dyc * sin_sg2


@gtstencil()
def vorticity_main(
    v: FloatField,
    ua: FloatField,
    cosa_u: FloatFieldIJ,
    sina_u: FloatFieldIJ,
    dxc: FloatFieldIJ,
    vort: FloatField,
):
    with computation(PARALLEL), interval(...):
        vort = (v - 0.5 * (ua[-1, 0, 0] + ua) * cosa_u) * dxc * sina_u


@gtstencil()
def vorticity_x_edge(
    v: FloatField,
    uc: FloatField,
    dxc: FloatFieldIJ,
    sin_sg3: FloatFieldIJ,
    sin_sg1: FloatFieldIJ,
    vort: FloatField,
):
    with computation(PARALLEL), interval(...):
        vort = v * dxc * sin_sg3[-1, 0] if uc > 0 else v * dxc * sin_sg1


@gtstencil()
def delpc_main(vort: FloatField, ptc: FloatField, delpc: FloatField):
    with computation(PARALLEL), interval(...):
        delpc = vort[0, -1, 0] - vort + ptc[-1, 0, 0] - ptc


@gtstencil()
def corner_south_remove_extra_term(vort: FloatField, delpc: FloatField):
    with computation(PARALLEL), interval(...):
        delpc -= vort[0, -1, 0]


@gtstencil()
def corner_north_remove_extra_term(vort: FloatField, delpc: FloatField):
    with computation(PARALLEL), interval(...):
        delpc += vort


@gtscript.function
def damp_tmp(q, da_min_c, d2_bg, dddmp):
    tmpddd = dddmp * q
    mintmp = 0.2 if 0.2 < tmpddd else tmpddd
    maxd2 = d2_bg if d2_bg > mintmp else mintmp
    damp = da_min_c * maxd2
    return damp


@gtstencil()
def damping_nord0_stencil(
    rarea_c: FloatFieldIJ,
    delpc: FloatField,
    vort: FloatField,
    ke: FloatField,
    da_min_c: float,
    d2_bg: float,
    dddmp: float,
    dt: float,
):
    with computation(PARALLEL), interval(...):
        delpc = rarea_c * delpc
        delpcdt = delpc * dt
        absdelpcdt = delpcdt if delpcdt >= 0 else -delpcdt
        damp = damp_tmp(absdelpcdt, da_min_c, d2_bg, dddmp)
        vort = damp * delpc
        ke += vort


@gtstencil()
def damping_nord_highorder_stencil(
    vort: FloatField,
    ke: FloatField,
    delpc: FloatField,
    divg_d: FloatField,
    da_min_c: float,
    d2_bg: float,
    dddmp: float,
    dd8: float,
):
    with computation(PARALLEL), interval(...):
        damp = damp_tmp(vort, da_min_c, d2_bg, dddmp)
        vort = damp * delpc + dd8 * divg_d
        ke = ke + vort


@gtstencil()
def vc_from_divg(divg_d: FloatField, divg_u: FloatFieldIJ, vc: FloatField):
    with computation(PARALLEL), interval(...):
        vc[0, 0, 0] = (divg_d[1, 0, 0] - divg_d) * divg_u


@gtstencil()
def uc_from_divg(divg_d: FloatField, divg_v: FloatFieldIJ, uc: FloatField):
    with computation(PARALLEL), interval(...):
        uc[0, 0, 0] = (divg_d[0, 1, 0] - divg_d) * divg_v


@gtstencil()
def redo_divg_d(uc: FloatField, vc: FloatField, divg_d: FloatField):
    with computation(PARALLEL), interval(...):
        divg_d[0, 0, 0] = uc[0, -1, 0] - uc + vc[-1, 0, 0] - vc


@gtstencil()
def smagorinksy_diffusion_approx(delpc: FloatField, vort: FloatField, absdt: float):
    with computation(PARALLEL), interval(...):
        vort = absdt * (delpc ** 2.0 + vort ** 2.0) ** 0.5


def vorticity_calc(wk, vort, delpc, dt, nord, kstart, nk):
    if nord != 0:
        if spec.namelist.dddmp < 1e-5:
            vort[:, :, kstart : kstart + nk] = 0
        else:
            if spec.namelist.grid_type < 3:
                a2b_ord4.compute(wk, vort, kstart, nk, False)
                smagorinksy_diffusion_approx(
                    delpc,
                    vort,
                    abs(dt),
                    origin=(spec.grid.is_, spec.grid.js, kstart),
                    domain=(spec.grid.nic + 1, spec.grid.njc + 1, nk),
                )
            else:
                raise Exception("Not implemented, smag_corner")


def compute(
    u: FloatField,
    v: FloatField,
    va: FloatField,
    ptc: FloatField,
    vort: FloatField,
    ua: FloatField,
    divg_d: FloatField,
    vc: FloatField,
    uc: FloatField,
    delpc: FloatField,
    ke: FloatField,
    wk: FloatField,
    d2_bg: float,
    dt: float,
    nord: float,
    kstart: int = 0,
    nk: Optional[int] = None,
) -> None:
    grid = spec.grid
    if nk is None:
        nk = grid.npz - kstart
    # Avoid running center-domain computation on tile edges, since they'll be
    # overwritten.
    is2 = grid.is_ + 1 if grid.west_edge else grid.is_
    ie1 = grid.ie if grid.east_edge else grid.ie + 1
    nord = int(nord)
    if nord == 0:
        damping_zero_order(
            u, v, va, ptc, vort, ua, vc, uc, delpc, ke, d2_bg, dt, is2, ie1, kstart, nk
        )
    else:
        copy_stencil(
            divg_d,
            delpc,
            origin=(grid.is_, grid.js, kstart),
            domain=(grid.nic + 1, grid.njc + 1, nk),
        )
        for n in range(1, nord + 1):
            nt = nord - n
            nint = grid.nic + 2 * nt + 1
            njnt = grid.njc + 2 * nt + 1
            js = grid.js - nt
            is_ = grid.is_ - nt
            fillc = (
                (n != nord)
                and spec.namelist.grid_type < 3
                and not grid.nested
                and (
                    grid.sw_corner or grid.se_corner or grid.ne_corner or grid.nw_corner
                )
            )
            if fillc:
                corners.fill_corners_2d(divg_d, grid, "B", "x")
            vc_from_divg(
                divg_d,
                grid.divg_u,
                vc,
                origin=(is_ - 1, js, kstart),
                domain=(nint + 1, njnt, nk),
            )
            if fillc:
                corners.fill_corners_2d(divg_d, grid, "B", "y")
            uc_from_divg(
                divg_d,
                grid.divg_v,
                uc,
                origin=(is_, js - 1, kstart),
                domain=(nint, njnt + 1, nk),
            )
            if fillc:
                corners.fill_corners_dgrid(vc, uc, grid, True)

            redo_divg_d(
                uc, vc, divg_d, origin=(is_, js, kstart), domain=(nint, njnt, nk)
            )
            corner_domain = (1, 1, nk)
            if grid.sw_corner:
                corner_south_remove_extra_term(
                    uc, divg_d, origin=(grid.is_, grid.js, kstart), domain=corner_domain
                )
            if grid.se_corner:
                corner_south_remove_extra_term(
                    uc,
                    divg_d,
                    origin=(grid.ie + 1, grid.js, kstart),
                    domain=corner_domain,
                )
            if grid.ne_corner:
                corner_north_remove_extra_term(
                    uc,
                    divg_d,
                    origin=(grid.ie + 1, grid.je + 1, kstart),
                    domain=corner_domain,
                )
            if grid.nw_corner:
                corner_north_remove_extra_term(
                    uc,
                    divg_d,
                    origin=(grid.is_, grid.je + 1, kstart),
                    domain=corner_domain,
                )
            if not grid.stretched_grid:
                basic.adjustmentfactor_stencil(
                    grid.rarea_c,
                    divg_d,
                    origin=(is_, js, kstart),
                    domain=(nint, njnt, nk),
                )

        vorticity_calc(wk, vort, delpc, dt, nord, kstart, nk)
        if grid.stretched_grid:
            dd8 = grid.da_min * spec.namelist.d4_bg ** (nord + 1)
        else:
            dd8 = (grid.da_min_c * spec.namelist.d4_bg) ** (nord + 1)
        damping_nord_highorder_stencil(
            vort,
            ke,
            delpc,
            divg_d,
            grid.da_min_c,
            d2_bg,
            spec.namelist.dddmp,
            dd8,
            origin=(grid.is_, grid.js, kstart),
            domain=(grid.nic + 1, grid.njc + 1, nk),
        )

    return vort, ke, delpc


def damping_zero_order(
    u: FloatField,
    v: FloatField,
    va: FloatField,
    ptc: FloatField,
    vort: FloatField,
    ua: FloatField,
    vc: FloatField,
    uc: FloatField,
    delpc: FloatField,
    ke: FloatField,
    d2_bg: float,
    dt: float,
    is2: int,
    ie1: int,
    kstart: int,
    nk: int,
) -> None:
    grid = spec.grid
    if not grid.nested:
        # TODO: ptc and vort are equivalent, but x vs y, consolidate if possible.
        ptc_main(
            u,
            va,
            grid.cosa_v,
            grid.sina_v,
            grid.dyc,
            ptc,
            origin=(grid.is_ - 1, grid.js, kstart),
            domain=(grid.nic + 2, grid.njc + 1, nk),
        )
        y_edge_domain = (grid.nic + 2, 1, nk)
        if grid.south_edge:
            ptc_y_edge(
                u,
                vc,
                grid.dyc,
                grid.sin_sg4,
                grid.sin_sg2,
                ptc,
                origin=(grid.is_ - 1, grid.js, kstart),
                domain=y_edge_domain,
            )
        if grid.north_edge:
            ptc_y_edge(
                u,
                vc,
                grid.dyc,
                grid.sin_sg4,
                grid.sin_sg2,
                ptc,
                origin=(grid.is_ - 1, grid.je + 1, kstart),
                domain=y_edge_domain,
            )

        vorticity_main(
            v,
            ua,
            grid.cosa_u,
            grid.sina_u,
            grid.dxc,
            vort,
            origin=(is2, grid.js - 1, kstart),
            domain=(ie1 - is2 + 1, grid.njc + 2, nk),
        )
        x_edge_domain = (1, grid.njc + 2, nk)
        if grid.west_edge:
            vorticity_x_edge(
                v,
                uc,
                grid.dxc,
                grid.sin_sg3,
                grid.sin_sg1,
                vort,
                origin=(grid.is_, grid.js - 1, kstart),
                domain=x_edge_domain,
            )
        if grid.east_edge:
            vorticity_x_edge(
                v,
                uc,
                grid.dxc,
                grid.sin_sg3,
                grid.sin_sg1,
                vort,
                origin=(grid.ie + 1, grid.js - 1, kstart),
                domain=x_edge_domain,
            )
    else:
        raise Exception("nested not implemented")
    compute_origin = (grid.is_, grid.js, kstart)
    compute_domain = (grid.nic + 1, grid.njc + 1, nk)
    delpc_main(vort, ptc, delpc, origin=compute_origin, domain=compute_domain)
    corner_domain = (1, 1, nk)
    if grid.sw_corner:
        corner_south_remove_extra_term(
            vort, delpc, origin=(grid.is_, grid.js, kstart), domain=corner_domain
        )
    if grid.se_corner:
        corner_south_remove_extra_term(
            vort, delpc, origin=(grid.ie + 1, grid.js, kstart), domain=corner_domain
        )
    if grid.ne_corner:
        corner_north_remove_extra_term(
            vort, delpc, origin=(grid.ie + 1, grid.je + 1, kstart), domain=corner_domain
        )
    if grid.nw_corner:
        corner_north_remove_extra_term(
            vort, delpc, origin=(grid.is_, grid.je + 1, kstart), domain=corner_domain
        )

    damping_nord0_stencil(
        grid.rarea_c,
        delpc,
        vort,
        ke,
        grid.da_min_c,
        d2_bg,
        spec.namelist.dddmp,
        dt,
        origin=compute_origin,
        domain=compute_domain,
    )
