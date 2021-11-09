import math
import numpy as np
import fv3core.utils.global_constants as constants
from fv3core.grid import lon_lat_midpoint, great_circle_distance_lon_lat
import fv3gfs.util as fv3util 
nhalo = 3
u0 = 35.0
pcen = [math.pi / 9.0, 2.0 * math.pi / 9.0]
u1 = 1.0
pt0 = 0.0
eta_0 = 0.252
eta_s = 1.0 # surface level
eta_t = 0.2 # tropopause
t_0 = 288.0
delta_t = 480000.0
lapse_rate = 0.005
ptop_min = 1e-8
nhalo = 3
# RADIUS = 6.3712e6 vs Jabowski paper  6.371229 1e6
r0 = constants.RADIUS / 10.0 # specifically for test case == 13, otherwise r0 = 1
# Equation (1) Jablonowski & Williamson Baroclinic test case Perturbation. DCMIP 2016
def compute_horizontal_shape(full_array):
    full_nx, full_ny, _ = full_array.shape
    nx = full_nx - 2*  nhalo - 1
    ny = full_ny - 2 * nhalo - 1
    return nx, ny

def compute_eta(eta, eta_v, ak, bk):
    eta[:-1] = 0.5 * ((ak[:-1] + ak[1:])/1.e5 + bk[:-1] + bk[1:])
    eta_v[:-1] = (eta[:-1] - eta_0) * math.pi * 0.5

def setup_pressure_fields(eta, eta_v, delp, ps, pe, peln, pk, pkz, qvapor, ak, bk, ptop, latitude_agrid, adiabatic):
    nx, ny = compute_horizontal_shape(delp)
    ps[:] = 100000.0
    cmps_x = slice(nhalo,  nhalo + nx)
    cmps_y = slice(nhalo,  nhalo + ny)

    delp[cmps_x, cmps_y, :-1] = ak[None, None, 1:] - ak[None, None, :-1] + ps[cmps_x, cmps_y, None] * (bk[None, None, 1:] - bk[None, None, :-1])

    pe[cmps_x, cmps_y, 0] = ptop
    peln[cmps_x, cmps_y, 0] = math.log(ptop)
    pk[cmps_x, cmps_y, 0] = ptop**constants.KAPPA
    for k in range(1, pe.shape[2]):
        pe[cmps_x, cmps_y, k] = pe[cmps_x, cmps_y, k - 1] + delp[cmps_x, cmps_y, k - 1]
    pk[cmps_x, cmps_y, 1:] = np.exp(constants.KAPPA * np.log(pe[cmps_x, cmps_y, 1:]))
    peln[cmps_x, cmps_y, 1:]  = np.log(pe[cmps_x, cmps_y, 1:])
    pkz[cmps_x, cmps_y, :-1] = (pk[cmps_x, cmps_y, 1:] - pk[cmps_x, cmps_y, :-1]) / (constants.KAPPA * (peln[cmps_x, cmps_y, 1:] - peln[cmps_x, cmps_y, :-1]))
    
    compute_eta(eta, eta_v, ak, bk)

    if not adiabatic:
        
        ptmp = delp[cmps_x, cmps_y, :-1]/(peln[cmps_x, cmps_y, 1:]-peln[cmps_x, cmps_y, :-1]) - 100000.
        qvapor[cmps_x, cmps_y, :-1] = 0.021*np.exp(-(latitude_agrid[cmps_x, cmps_y, None] / pcen[1])**4.) * np.exp(-(ptmp/34000.)**2.)

# Equation (2) Jablonowski & Williamson Baroclinic test case Perturbation. DCMIP 2016 
def zonal_wind(utmp, eta_v, latitude_dgrid, islice, islice_grid, jslice, jslice_grid):
    utmp[islice, jslice,:] =  u0 * np.cos(eta_v[:])**(3.0/2.0) * np.sin(2.0 * latitude_dgrid[islice_grid,jslice_grid, None])**2.0

def near_center_adjustment(utmp, u1, longitude, latitude, islice, islice_grid, jslice, jslice_grid):
    r = np.zeros((utmp.shape[0], utmp.shape[1]))
    r[islice, jslice] = great_circle_distance_lon_lat(pcen[0], longitude[islice_grid,jslice_grid],  pcen[1], latitude[islice_grid,jslice_grid], constants.RADIUS, np)
    r3d = np.repeat(r[:,:, None], utmp.shape[2], axis=2)
    adjust_bool = ((r3d/r0)**2.0 < 40.0)
    utmp[islice, jslice,:][adjust_bool[islice, jslice,:]] = utmp[islice, jslice,:][adjust_bool[islice, jslice,:]] + u1 * np.exp(-(r3d[islice, jslice,:][adjust_bool[islice, jslice,:]]/r0)**2.0)
    
def wind_component_calc(utmp, eta_v, longitude_dgrid, latitude_dgrid, ee, islice, islice_grid, jslice, jslice_grid):
  
    vv = np.zeros(utmp.shape)
    zonal_wind(utmp, eta_v, latitude_dgrid, islice, islice_grid, jslice, jslice_grid)
    near_center_adjustment(utmp, u1, longitude_dgrid, latitude_dgrid, islice, islice_grid, jslice, jslice_grid)
    vv[islice, jslice, :] = utmp[islice, jslice, :]*(ee[islice_grid, jslice_grid, 1]*np.cos(longitude_dgrid[islice_grid, jslice_grid]) - ee[islice_grid, jslice_grid, 0]*np.sin(longitude_dgrid[islice_grid, jslice_grid]))[:,:,None]
 
    return vv


def compute_temperature_component(eta, eta_v, t_mean, latitude, islice, jslice):
    return t_mean + 0.75*(eta[:] * math.pi * u0 / constants. RDGAS) * np.sin(eta_v[:])*np.sqrt(np.cos(eta_v[:])) * (( -2.0 * (np.sin(latitude[islice, jslice, None])**6.0) *(np.cos(latitude[islice, jslice, None])**2.0 + 1.0/3.0) + 10.0/63.0 ) *2.0*u0*np.cos(eta_v[:])**(3.0/2.0) + ((8.0/5.0)*(np.cos(latitude[islice, jslice, None])**3.0)*(np.sin(latitude[islice, jslice, None])**2.0 + 2.0/3.0) - math.pi/4.0 ) * constants.RADIUS * constants.OMEGA )

def compute_surface_geopotential_component(latitude, islice, jslice):
    u_comp = u0 * (np.cos((eta_s-eta_0)*math.pi/2.0))**(3.0/2.0)
    return  u_comp * (( -2.0*(np.sin(latitude[islice, jslice])**6.0) * (np.cos(latitude[islice, jslice])**2.0 + 1.0/3.0) + 10.0/63.0 ) * u_comp + ((8.0/5.0)*(np.cos(latitude[islice, jslice])**3.0)*(np.sin(latitude[islice, jslice])**2.0 + 2.0/3.0) - math.pi/4.0 )*constants.RADIUS * constants.OMEGA)


def baroclinic_initialization(peln, qvapor, delp, u, v, pt, phis, delz, w, eta, eta_v, longitude, latitude, longitude_agrid, latitude_agrid, ee1, ee2, es1, ew2, ptop):
  
    nx, ny = compute_horizontal_shape(delp)
    utmp = np.zeros(u.shape)
    vv1 = np.zeros(u.shape)
    # Equation (2) for j+1

    islice = slice(nhalo, nhalo + nx + 1)
    jslice = slice(nhalo, nhalo + ny)
    jslice_p1 = slice(nhalo + 1, nhalo + ny + 1)
          
    vv1 = wind_component_calc(utmp, eta_v, longitude, latitude, ee2, islice, islice, jslice, jslice_p1)
    vv3 = wind_component_calc(utmp, eta_v, longitude, latitude, ee2,  islice, islice, jslice, jslice)
    pa1, pa2 = lon_lat_midpoint(longitude[:, 0:-1], longitude[:, 1:], latitude[:, 0:-1], latitude[:, 1:], np)
    vv2 = wind_component_calc(utmp, eta_v, pa1, pa2, ew2[:,:-1,:],islice, islice, jslice, slice(nhalo, -nhalo))
    v[islice, jslice,:] = 0.25 * (vv1 + 2.0 * vv2 + vv3)[islice, jslice,:]
   

    # u
    islice = slice(nhalo, nhalo + nx)
    jslice = slice(nhalo, nhalo + ny + 1)
    islice_p1 = slice(nhalo + 1, nhalo + nx + 1)

    uu1 = wind_component_calc(utmp, eta_v, longitude, latitude, ee1, islice, islice, jslice, jslice)
    uu3 = wind_component_calc(utmp, eta_v, longitude, latitude, ee1,  islice, islice_p1, jslice, jslice)
    pa1, pa2 = lon_lat_midpoint(longitude[0:-1, :], longitude[1:, :], latitude[0:-1, :], latitude[1:, :], np)
    uu2 = wind_component_calc(utmp, eta_v, pa1, pa2, es1[:-1, :,:],islice,  slice(nhalo, -nhalo), jslice, jslice)
    u[islice, jslice,:] = 0.25 * (uu1 + 2.0 * uu2 + uu3)[islice, jslice,:]
    
    # Temperature
    islice = slice(nhalo, nhalo + nx)
    jslice = slice(nhalo, nhalo + ny)
    
    t_mean = t_0 * eta[:] ** (constants.RDGAS * lapse_rate / constants.GRAV)
    t_mean[eta_t > eta] = t_mean[eta_t > eta] + delta_t*(eta_t - eta[eta_t > eta])**5.0
    # A-grid cell center temperature
  
   
    pt1 = compute_temperature_component(eta, eta_v, t_mean, latitude_agrid, islice, jslice)
    p1, p2_ij_i1j = lon_lat_midpoint(longitude[0:-1, :], longitude[1:, :], latitude[0:-1, :], latitude[1:, :], np)
    pt2 = compute_temperature_component(eta, eta_v, t_mean, p2_ij_i1j, islice, jslice)
    p1, p2_i1j_i1j1 = lon_lat_midpoint(longitude[1:, 0:-1], longitude[1:, 1:], latitude[1:, 0:-1], latitude[1:, 1:], np)
    pt3 = compute_temperature_component(eta, eta_v, t_mean, p2_i1j_i1j1, islice, jslice)
    p1, p2_ij1_i1j1 = lon_lat_midpoint(longitude[0:-1, 1:], longitude[1:, 1:], latitude[0:-1, 1:], latitude[1:, 1:], np)
    pt4 = compute_temperature_component(eta, eta_v, t_mean, p2_ij1_i1j1, islice, jslice)
    p1, p2_ij_ij1  = lon_lat_midpoint(longitude[:, 0:-1], longitude[:, 1:], latitude[:, 0:-1], latitude[:, 1:], np)
    pt5 = compute_temperature_component(eta, eta_v, t_mean, p2_ij_ij1, islice, jslice)
    pt6 = compute_temperature_component(eta, eta_v, t_mean, latitude, islice, jslice)
    pt7 = compute_temperature_component(eta, eta_v, t_mean, latitude[1:,:], islice, jslice)
    pt8 = compute_temperature_component(eta, eta_v, t_mean, latitude[1:,1:], islice, jslice)
    pt9 = compute_temperature_component(eta, eta_v, t_mean, latitude[:,1:], islice, jslice)
    pt[:] = 1.0
    pt[islice, jslice, :] =  0.25 * pt1 + 0.125 * (pt2 + pt3 + pt4 + pt5) + 0.0625 * (pt6 + pt7 + pt8 + pt9)
    # WARNING untested
    near_center_adjustment(pt, pt0, longitude_agrid, latitude_agrid, islice, islice, jslice, jslice)


     # phis
   
    pt1 = compute_surface_geopotential_component(latitude_agrid, islice, jslice)
    pt2 = compute_surface_geopotential_component(p2_ij_i1j, islice, jslice)
    pt3 = compute_surface_geopotential_component(p2_i1j_i1j1, islice, jslice)
    pt4 = compute_surface_geopotential_component(p2_ij1_i1j1, islice, jslice)
    pt5 = compute_surface_geopotential_component(p2_ij_ij1, islice, jslice)
    pt6 = compute_surface_geopotential_component(latitude, islice, jslice)
    pt7 = compute_surface_geopotential_component(latitude[1:,:], islice, jslice)
    pt8 = compute_surface_geopotential_component(latitude[1:,1:], islice, jslice)
    pt9 = compute_surface_geopotential_component(latitude[:,1:], islice, jslice)
    phis[:] = 1.e25
    phis[islice, jslice] =  0.25 * pt1 + 0.125 * (pt2 + pt3 + pt4 + pt5) + 0.0625 * (pt6 + pt7 + pt8 + pt9)

    # if not hydrostatic:
    w[:] = 1e30
    w[islice, jslice, :] = 0.0
    delz[:] = 1e+30
    delz[islice, jslice, 0:-1] = constants.RDGAS/constants.GRAV * pt[islice, jslice, 0:-1]*(peln[islice, jslice, 0:-1]-peln[islice, jslice, 1:])
    
    # if not adiabatic:
    pt[islice, jslice, :] = pt[islice, jslice, :]/(1. + constants.ZVIR * qvapor[islice, jslice, :])


def p_var(delp, delz, pt, ps, qvapor, pe, peln, pk, pkz, ptop, moist_phys, make_nh, hydrostatic=False,  adjust_dry_mass=False):
    """
    Computes auxiliary pressure variables for a hydrostatic state.
    The variables are: surfce, interface, layer-mean pressure, exener function
    Given (ptop, delp) computes (ps, pk, pe, peln, pkz)
    """
    assert(not adjust_dry_mass)
    assert(not hydrostatic)

    pek = ptop ** constants.KAPPA
    
    nx, ny = compute_horizontal_shape(delp)
    islice = slice(nhalo, nhalo + nx)
    jslice = slice(nhalo, nhalo + ny)
    pe[islice, jslice, 0] = ptop
    pk[islice, jslice, 0] = pek


    for k in range(1, delp.shape[2]):
        pe[islice, jslice, k] = pe[islice, jslice, k - 1] + delp[islice, jslice, k - 1]
    peln[islice, jslice, 1:] = np.log(pe[islice, jslice, 1:])
    pk[islice, jslice, 1:] = np.exp(constants.KAPPA * peln[islice, jslice, 1:])
    ps[islice, jslice] = pe[islice, jslice, -1]
    if ptop < ptop_min:
        ak1 = (constants.KAPPA + 1.0) / constants.KAPPA
        peln[islice, jslice, 0] =  peln[islice, jslice, 1] - ak1
    else:
         peln[islice, jslice, 0] = np.log(ptop)

    if not hydrostatic:
        if make_nh:
            delz[:]= 1.e25
            delz[islice, jslice, :-1] = constants.RDG * pt[islice, jslice, :-1] * (peln[islice, jslice, 1:] - peln[islice, jslice, :-1])
        if moist_phys:
            pkz[islice, jslice, :-1] = np.exp(constants.KAPPA * np.log(constants.RDG * delp[islice, jslice, :-1] * pt[islice, jslice, :-1] * (1. + constants.ZVIR * qvapor[islice, jslice, :-1]) / delz[islice, jslice, :-1]))
        else:
            pkz[islice, jslice, :-1] = np.exp(constants.KAPPA * np.log(constants.RDG * delp[islice, jslice, :-1] * pt[islice, jslice, :-1] / delz[islice, jslice, :-1]))
            
def init_case(eta, eta_v, delp, fC, f0, ps, pe, peln, pk, pkz, qvapor, ak, bk, ptop, u, v, pt, phis, delz, w,  longitude, latitude, longitude_agrid, latitude_agrid, ee1, ee2, es1, ew2, adiabatic, hydrostatic, moist_phys): 
    nx, ny = compute_horizontal_shape(delp)
    delp[:] = 1e30
    delp[:nhalo, :nhalo] = 0.0
    delp[:nhalo, nhalo + ny:] = 0.0
    delp[nhalo + nx:, :nhalo] = 0.0
    delp[nhalo + nx:,  nhalo + ny:] = 0.0
    alpha = 0.0
    fC[:, :] = 2. * constants.OMEGA * (-1.*np.cos(longitude) * np.cos(latitude) * np.sin(alpha) + np.sin(latitude) * np.cos(alpha) )	
    f0[:-1, :-1] = 2. * constants.OMEGA * (-1. * np.cos(longitude_agrid[:-1, :-1]) * np.cos(latitude_agrid[:-1, :-1]) * np.sin(alpha) + np.sin(latitude_agrid[:-1, :-1])*np.cos(alpha) )
    # halo update f0
    # fill_corners(f0, ydir)
    pe[:] = 0.0
    pt[:] = 1.0
    setup_pressure_fields(eta, eta_v, delp, ps, pe, peln, pk, pkz, qvapor, ak, bk, ptop, latitude_agrid=latitude_agrid[:-1, :-1], adiabatic=adiabatic)
    baroclinic_initialization(peln, qvapor, delp, u, v, pt, phis, delz, w, eta, eta_v,  longitude, latitude, longitude_agrid, latitude_agrid, ee1, ee2, es1, ew2, ptop)
    p_var( delp, delz, pt, ps, qvapor, pe, peln, pk, pkz, ptop, moist_phys, make_nh=(not hydrostatic), hydrostatic=hydrostatic)
  
    # halo update phis
    # halo update u and v
