import gt4py.gtscript as gtscript
from gt4py.gtscript import PARALLEL, computation, interval

import fv3core.utils.gt4py_utils as utils
from fv3core.decorators import FrozenStencil, computepath_method
from fv3core.stencils.d2a2c_vect import DGrid2AGrid2CGridVectors
from fv3core.utils.grid import axis_offsets
from fv3core.utils.typing import FloatField, FloatFieldIJ


def geoadjust_ut(
    ut: FloatField,
    dy: FloatFieldIJ,
    sin_sg3: FloatFieldIJ,
    sin_sg1: FloatFieldIJ,
    dt2: float,
):
    with computation(PARALLEL), interval(...):
        ut[0, 0, 0] = (
            dt2 * ut * dy * sin_sg3[-1, 0] if ut > 0 else dt2 * ut * dy * sin_sg1
        )


def geoadjust_vt(
    vt: FloatField,
    dx: FloatFieldIJ,
    sin_sg4: FloatFieldIJ,
    sin_sg2: FloatFieldIJ,
    dt2: float,
):
    with computation(PARALLEL), interval(...):
        vt[0, 0, 0] = (
            dt2 * vt * dx * sin_sg4[0, -1] if vt > 0 else dt2 * vt * dx * sin_sg2
        )


def absolute_vorticity(vort: FloatField, fC: FloatFieldIJ, rarea_c: FloatFieldIJ):
    with computation(PARALLEL), interval(...):
        vort[0, 0, 0] = fC + rarea_c * vort


@gtscript.function
def nonhydro_x_fluxes(delp: FloatField, pt: FloatField, w: FloatField, utc: FloatField):
    fx1 = delp[-1, 0, 0] if utc > 0.0 else delp
    fx = pt[-1, 0, 0] if utc > 0.0 else pt
    fx2 = w[-1, 0, 0] if utc > 0.0 else w
    fx1 = utc * fx1
    fx = fx1 * fx
    fx2 = fx1 * fx2
    return fx, fx1, fx2


@gtscript.function
def nonhydro_y_fluxes(delp: FloatField, pt: FloatField, w: FloatField, vtc: FloatField):
    fy1 = delp[0, -1, 0] if vtc > 0.0 else delp
    fy = pt[0, -1, 0] if vtc > 0.0 else pt
    fy2 = w[0, -1, 0] if vtc > 0.0 else w
    fy1 = vtc * fy1
    fy = fy1 * fy
    fy2 = fy1 * fy2
    return fy, fy1, fy2


def compute_nonhydro_x_fluxes(
    delp: FloatField,
    pt: FloatField,
    utc: FloatField,
    w: FloatField,
    fx: FloatField,
    fx1: FloatField,
    fx2: FloatField,
):

    with computation(PARALLEL), interval(...):
        fx, fx1, fx2 = nonhydro_x_fluxes(delp, pt, w, utc)


def transportdelp_update_vorticity_and_kineticenergy(
    delp: FloatField,
    pt: FloatField,
    utc: FloatField,
    vtc: FloatField,
    w: FloatField,
    rarea: FloatFieldIJ,
    delpc: FloatField,
    ptc: FloatField,
    wc: FloatField,
    ke: FloatField,
    vort: FloatField,
    ua: FloatField,
    va: FloatField,
    uc: FloatField,
    vc: FloatField,
    u: FloatField,
    v: FloatField,
    fx: FloatField,
    fx1: FloatField,
    fx2: FloatField,
):

    with computation(PARALLEL), interval(...):
        fy, fy1, fy2 = nonhydro_y_fluxes(delp, pt, w, vtc)
        delpc = delp + (fx1 - fx1[1, 0, 0] + fy1 - fy1[0, 1, 0]) * rarea
        ptc = (pt * delp + (fx - fx[1, 0, 0] + fy - fy[0, 1, 0]) * rarea) / delpc
        wc = (w * delp + (fx2 - fx2[1, 0, 0] + fy2 - fy2[0, 1, 0]) * rarea) / delpc

        # update vorticity and kinetic energy
        ke = uc if ua > 0.0 else uc[1, 0, 0]
        vort = vc if va > 0.0 else vc[0, 1, 0]


def final_kineticenergy(
    ua: FloatField, va: FloatField, vort: FloatField, ke: FloatField, dt2: float
):
    with computation(PARALLEL), interval(...):
        ke = 0.5 * dt2 * (ua * ke + va * vort)


def uf_main(
    u: FloatField,
    va: FloatField,
    dyc: FloatFieldIJ,
    sin_sg2: FloatFieldIJ,
    sin_sg4: FloatFieldIJ,
    cos_sg2: FloatFieldIJ,
    cos_sg4: FloatFieldIJ,
    uf: FloatField,
):
    with computation(PARALLEL), interval(...):
        uf = (
            (u - 0.25 * (va[0, -1, 0] + va) * (cos_sg4[0, -1] + cos_sg2))
            * dyc
            * 0.5
            * (sin_sg4[0, -1] + sin_sg2)
        )


def vf_main(
    v: FloatField,
    ua: FloatField,
    dxc: FloatFieldIJ,
    sin_sg1: FloatFieldIJ,
    sin_sg3: FloatFieldIJ,
    cos_sg1: FloatFieldIJ,
    cos_sg3: FloatFieldIJ,
    vf: FloatField,
):
    with computation(PARALLEL), interval(...):
        vf = (
            (v - 0.25 * (ua[-1, 0, 0] + ua) * (cos_sg3[-1, 0] + cos_sg1))
            * dxc
            * 0.5
            * (sin_sg3[-1, 0] + sin_sg1)
        )


def divergence_main(uf: FloatField, vf: FloatField, divg_d: FloatField):
    with computation(PARALLEL), interval(...):
        divg_d = vf[0, -1, 0] - vf + uf[-1, 0, 0] - uf


def divergence_main_final(rarea_c: FloatFieldIJ, divg_d: FloatField):
    with computation(PARALLEL), interval(...):
        divg_d *= rarea_c

def update_vorticity(
    uc: FloatField,
    vc: FloatField,
    dxc: FloatFieldIJ,
    dyc: FloatFieldIJ,
    vort_c: FloatField,
):
    """Update vort_c.

    Args:
        uc: x-velocity on C-grid (input)
        vc: y-velocity on C-grid (input)
        dxc: grid spacing in x-dir (input)
        dyc: grid spacing in y-dir (input)
        vort_c: C-grid vorticity (output)
    """

    with computation(PARALLEL), interval(...):
        fx = dxc * uc
        fy = dyc * vc
    with computation(PARALLEL), interval(...):
        vort_c = fx[0, -1, 0] - fx - fy[-1, 0, 0] + fy


def update_x_velocity(
    vorticity: FloatField,
    ke: FloatField,
    velocity: FloatField,
    velocity_c: FloatField,
    cosa: FloatFieldIJ,
    sina: FloatFieldIJ,
    rdxc: FloatFieldIJ,
    dt2: float,
):
    with computation(PARALLEL), interval(...):
        tmp_flux = dt2 * (velocity - velocity_c * cosa) / sina
        flux = vorticity[0, 0, 0] if tmp_flux > 0.0 else vorticity[0, 1, 0]
        velocity_c = velocity_c + tmp_flux * flux + rdxc * (ke[-1, 0, 0] - ke)


def update_y_velocity(
    vorticity: FloatField,
    ke: FloatField,
    velocity: FloatField,
    velocity_c: FloatField,
    cosa: FloatFieldIJ,
    sina: FloatFieldIJ,
    rdyc: FloatFieldIJ,
    dt2: float,
):
    with computation(PARALLEL), interval(...):
        tmp_flux = dt2 * (velocity - velocity_c * cosa) / sina
        flux = vorticity[0, 0, 0] if tmp_flux > 0.0 else vorticity[1, 0, 0]
        velocity_c = velocity_c - tmp_flux * flux + rdyc * (ke[0, -1, 0] - ke)


def initialize_delpc_ptc(delpc: FloatField, ptc: FloatField):
    with computation(PARALLEL), interval(...):
        delpc = 0.0
        ptc = 0.0


class CGridShallowWaterDynamics:
    """
    Fortran name is c_sw
    """

    def __init__(self, grid, namelist):
        self.grid = grid
        self.namelist = namelist
        self._dord4 = True

        self._D2A2CGrid_Vectors = DGrid2AGrid2CGridVectors(
            self.grid, self.namelist, self._dord4
        )
        grid_type = self.namelist.grid_type
        shape = self.grid.domain_shape_full(add=(1, 1, 1))
        self.delpc = utils.make_storage_from_shape(shape)
        self.ptc = utils.make_storage_from_shape(shape)
        self._tmp_uf = utils.make_storage_from_shape(shape)
        self._tmp_vf = utils.make_storage_from_shape(shape)
        self._tmp_fx = utils.make_storage_from_shape(shape)
        self._tmp_fx1 = utils.make_storage_from_shape(shape)
        self._tmp_fx2 = utils.make_storage_from_shape(shape)

        self._initialize_delpc_ptc = FrozenStencil(
            func=initialize_delpc_ptc,
            origin=self.grid.full_origin(),
            domain=self.grid.domain_shape_full(),
        )

        self._ke = utils.make_storage_from_shape(shape)
        self._vort = utils.make_storage_from_shape(shape)
        origin = self.grid.compute_origin()
        domain = self.grid.domain_shape_compute(add=(1, 1, 0))
        ax_offsets = axis_offsets(self.grid, origin, domain)

        if self.namelist.nord > 0:
            self._uf_main = FrozenStencil(
                uf_main,
                origin=self.grid.compute_origin(add=(-1, 0, 0)),
                domain=self.grid.domain_shape_compute(add=(2, 1, 0)),
            )
            self._vf_main = FrozenStencil(
                vf_main,
                origin=self.grid.compute_origin(add=(0, -1, 0)),
                domain=self.grid.domain_shape_compute(add=(1, 2, 0)),
            )

            self._divergence_main = FrozenStencil(
                divergence_main,
                origin=origin,
                domain=domain,
            )
            self._divergence_main_final = FrozenStencil(
                divergence_main_final,
                origin=origin,
                domain=domain,
            )
        geo_origin = (self.grid.is_ - 1, self.grid.js - 1, 0)
        self._geoadjust_ut = FrozenStencil(
            func=geoadjust_ut,
            origin=geo_origin,
            domain=(self.grid.nic + 3, self.grid.njc + 2, self.grid.npz),
        )
        self._geoadjust_vt = FrozenStencil(
            func=geoadjust_vt,
            origin=geo_origin,
            domain=(self.grid.nic + 2, self.grid.njc + 3, self.grid.npz),
        )

        self._compute_nonhydro_x_fluxes = FrozenStencil(
            compute_nonhydro_x_fluxes,
            origin=geo_origin,
            domain=self.grid.domain_shape_compute(add=(3, 2, 0)),
        )

        self._transportdelp_updatevorticity_and_ke = FrozenStencil(
            transportdelp_update_vorticity_and_kineticenergy,
            externals={
                "grid_type": grid_type,
            },
            origin=geo_origin,
            domain=self.grid.domain_shape_compute(add=(2, 2, 0)),
        )

        self._final_ke = FrozenStencil(
            final_kineticenergy,
            origin=geo_origin,
            domain=self.grid.domain_shape_compute(add=(2, 2, 0)),
        )
        self._update_vorticity = FrozenStencil(
            update_vorticity,
            origin=origin,
            domain=domain,
        )

        self._absolute_vorticity = FrozenStencil(
            func=absolute_vorticity,
            origin=origin,
            domain=(self.grid.nic + 1, self.grid.njc + 1, self.grid.npz),
        )

        js = self.grid.js
        je = self.grid.je+1
        self._update_y_velocity = FrozenStencil(
            func=update_y_velocity,
            origin=(self.grid.is_, js, 0),
            domain=(self.grid.nic, je - js + 1, self.grid.npz),
        )
        is_ = self.grid.is_
        ie = self.grid.ie+1
        self._update_x_velocity = FrozenStencil(
            func=update_x_velocity,
            origin=(is_, self.grid.js, 0),
            domain=(ie - is_ + 1, self.grid.njc, self.grid.npz),
        )


    @computepath_method
    def _vorticitytransport_cgrid(
        self,
        uc,
        vc,
        vort_c,
        ke_c,
        v,
        u,
        dt2: float,
    ):
        """Update the C-Grid x and y velocity fields.

        Args:
            uc: x-velocity on C-grid (input, output)
            vc: y-velocity on C-grid (input, output)
            vort_c: Vorticity on C-grid (input)
            ke_c: kinetic energy on C-grid (input)
            v: y-velocity on D-grid (input)
            u: x-velocity on D-grid (input)
            dt2: timestep (input)
        """
        self._update_y_velocity(
            vort_c,
            ke_c,
            u,
            vc,
            self.grid.cosa_v,
            self.grid.sina_v,
            self.grid.rdyc,
            dt2,
        )

        self._update_x_velocity(
            vort_c,
            ke_c,
            v,
            uc,
            self.grid.cosa_u,
            self.grid.sina_u,
            self.grid.rdxc,
            dt2,
        )

    def _circulation_cgrid(self, uc, vc):
        self._update_vorticity(
            uc, vc, self.grid.dxc, self.grid.dyc, self._vort
        )

    def _divergence_corner(self, u, v, ua, va, divg_d):
        self._uf_main(
            u,
            va,
            self.grid.dyc,
            self.grid.sin_sg2,
            self.grid.sin_sg4,
            self.grid.cos_sg2,
            self.grid.cos_sg4,
            self._tmp_uf,
        )
        self._vf_main(
            v,
            ua,
            self.grid.dxc,
            self.grid.sin_sg1,
            self.grid.sin_sg3,
            self.grid.cos_sg1,
            self.grid.cos_sg3,
            self._tmp_vf,
        )

        self._divergence_main(self._tmp_uf, self._tmp_vf, divg_d)
        self._divergence_main_final(self.grid.rarea_c, divg_d)

    @computepath_method
    def __call__(
        self,
        delp,
        pt,
        u,
        v,
        w,
        uc,
        vc,
        ua,
        va,
        ut,
        vt,
        divgd,
        omga,
        dt2: float,
    ):
        """
        C-grid shallow water routine.

        Advances C-grid winds by half a time step.
        Args:
            delp: D-grid vertical delta in pressure (in)
            pt: D-grid potential temperature (in)
            u: D-grid x-velocity (in)
            v: D-grid y-velocity (in)
            w: vertical velocity (in)
            uc: C-grid x-velocity (inout)
            vc: C-grid y-velocity (inout)
            ua: A-grid x-velocity (in)
            va: A-grid y-velocity (in)
            ut: u * dx (inout)
            vt: v * dy (inout)
            divgd: D-grid horizontal divergence (inout)
            omga: Vertical pressure velocity (inout)
            dt2: Half a model timestep in seconds (in)
        """
        self._initialize_delpc_ptc(
            self.delpc,
            self.ptc,
        )
        self._D2A2CGrid_Vectors(uc, vc, u, v, ua, va, ut, vt)
        if self.namelist.nord > 0:
            self._divergence_corner(u, v, ua, va, divgd)
        self._geoadjust_ut(
            ut,
            self.grid.dy,
            self.grid.sin_sg3,
            self.grid.sin_sg1,
            dt2,
        )
        self._geoadjust_vt(
            vt,
            self.grid.dx,
            self.grid.sin_sg4,
            self.grid.sin_sg2,
            dt2,
        )

        self._compute_nonhydro_x_fluxes(
            delp,
            pt,
            ut,
            w,
            self._tmp_fx,
            self._tmp_fx1,
            self._tmp_fx2,
        )


        self._transportdelp_updatevorticity_and_ke(
            delp,
            pt,
            ut,
            vt,
            w,
            self.grid.rarea,
            self.delpc,
            self.ptc,
            omga,
            self._ke,
            self._vort,
            ua,
            va,
            uc,
            vc,
            u,
            v,
            self._tmp_fx,
            self._tmp_fx1,
            self._tmp_fx2,
        )

        self._final_ke(ua, va, self._vort, self._ke, dt2)
        self._circulation_cgrid(uc, vc)
        self._absolute_vorticity(
            self._vort,
            self.grid.fC,
            self.grid.rarea_c,
        )
        self._vorticitytransport_cgrid(uc, vc, self._vort, self._ke, v, u, dt2)
        return self.delpc, self.ptc
