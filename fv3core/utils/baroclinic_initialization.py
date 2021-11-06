import math
import numpy as np
import fv3core.utils.global_constants as constants
from fv3core.grid import lon_lat_midpoint, great_circle_distance_lon_lat
nhalo = 3
u0 = 35.0
pcen = [math.pi / 9.0, 2.0 * math.pi / 9.0]
u1 = 1.0
pt0 = 0.0
# RADIUS = 6.3712e6 vs Jabowski paper  6.371229 1e6
r0 = constants.RADIUS / 10.0 # specifically for test case == 13, otherwise r0 = 1
# Equation (1) Jablonowski & Williamson Baroclinic test case Perturbation. DCMIP 2016 
def compute_eta(eta, eta_v, ak, bk):
    eta_0 = 0.252
    eta[:] = 0.5 * ((ak[:-1] + ak[1:])/1.e5 + bk[:-1] + bk[1:])
    eta_v[:] = (eta[:] - eta_0) * math.pi * 0.5

def setup_pressure_fields(eta, eta_v, delp, ps, pe, peln, pk, pkz, qvapor, ak, bk, ptop, latitude_agrid, adiabatic):  
    ps[:] = 100000.0
    cmps = slice(nhalo, -nhalo)
    large_init = 1e30 
    delp[:nhalo,nhalo:-nhalo, :] = large_init
    delp[nhalo:-nhalo, :nhalo,:] = large_init
    delp[-nhalo:,nhalo:-nhalo, :] = large_init
    delp[nhalo:-nhalo,-nhalo:, :] = large_init
    
    delp[cmps, cmps, :-1] = ak[None, None, 1:] - ak[None, None, :-1] + ps[cmps, cmps, None] * (bk[None, None, 1:] - bk[None, None, :-1])

    pe[cmps, cmps, 0] = ptop
    peln[cmps, cmps, 0] = math.log(ptop)
    pk[cmps, cmps, 0] = ptop**constants.KAPPA
    for k in range(1, pe.shape[2]):
        pe[cmps, cmps, k] = pe[cmps, cmps, k - 1] + delp[cmps, cmps, k - 1]
    pk[cmps, cmps, 1:] = np.exp(constants.KAPPA * np.log(pe[cmps, cmps, 1:]))
    peln[cmps, cmps, 1:]  = np.log(pe[cmps, cmps, 1:])
    pkz[cmps, cmps, :-1] = (pk[cmps, cmps, 1:] - pk[cmps, cmps, :-1]) / (constants.KAPPA * (peln[cmps, cmps, 1:] - peln[cmps, cmps, :-1]))
    
    compute_eta(eta, eta_v, ak, bk)

    if not adiabatic:
        
        ptmp = delp[cmps, cmps, :-1]/(peln[cmps, cmps, 1:]-peln[cmps, cmps, :-1]) - 100000.
        qvapor[cmps, cmps, :-1] = 0.021*np.exp(-(latitude_agrid[cmps, cmps, None] / pcen[1])**4.) * np.exp(-(ptmp/34000.)**2.)

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
    return t_mean + 0.75*(eta[:] * math.pi * u0 / constants. RDGAS) * np.sin(eta_v[:])*np.sqrt(np.cos(eta_v[:])) * (( -2.0 * (np.sin(latitude[islice, jslice, None])**6.0) *(np.cos(latitude[islice, jslice, None])**2.0 + 1.0/3.0) + 10.0/63.0 ) *2.0*u0*np.cos(eta_v[:])**(3.0/2.0) + ((8.0/5.0)*(np.cos(latitude[islice, jslice, None])**3.0)*(np.sin(latitude[islice, jslice, None])**2.0 + 2.0/3.0) - math.pi/4.0 )*constants.RADIUS * constants.OMEGA )

"""
tobias -- daint
nvidia florian
v37 vs for acoustic dace
microphysics with dace -- florian -- for loops
simple acoustics branch simple_acoustics.py
what for fv_dynamics
remapping -- while loop 
lagrangian contributions numpy version. 
regions --tobias. needs correct extents pointwise. analysis not working right. 
dev 37 not quite right. for-acoustics-gtc gt4py, then add to dace? or wait for v37? 
serialize data for self standing app. 
gt4py feature 
indirect addressing -- eddie
higher dimensional storages florian

tobias -- halo updates -- dace orchestrated, and consolidating 
"""

def baroclinic_initialization(qvapor, delp, u, v, pt, eta, eta_v, grid, ptop):
    nx, ny, nz = grid.domain_shape_compute()
    shape = (nx, ny, nz)
  
  
    utmp = np.zeros(u.shape)
    vv1 = np.zeros(u.shape)
    # Equation (2) for j+1

    nhalo = 3
    islice = slice(nhalo, -nhalo)
    jslice = slice(nhalo, -nhalo - 1)
    jslice_p1 = slice(nhalo+1, -nhalo)
       
    vv1 = wind_component_calc(utmp, eta_v, grid.bgrid1.data, grid.bgrid2.data, grid.ee2.data, islice, islice, jslice, jslice_p1)
    vv3 = wind_component_calc(utmp, eta_v, grid.bgrid1.data, grid.bgrid2.data, grid.ee2.data,  islice, islice, jslice, jslice)
    pa1, pa2 = lon_lat_midpoint(grid.bgrid1.data[:, 0:-1], grid.bgrid1.data[:, 1:], grid.bgrid2.data[:, 0:-1], grid.bgrid2.data[:, 1:], np)
    vv2 = wind_component_calc(utmp, eta_v, pa1, pa2, grid.ew2.data[:,:-1,:],islice, islice, jslice, slice(nhalo, -nhalo))
    v[islice, jslice,:] = 0.25 * (vv1 + 2.0 * vv2 + vv3)[islice, jslice,:]
   

    # u

    islice = slice(nhalo, -nhalo - 1)
    jslice = slice(nhalo, -nhalo)
    islice_p1 = slice(nhalo+1, -nhalo)
    uu1 = wind_component_calc(utmp, eta_v, grid.bgrid1.data, grid.bgrid2.data, grid.ee1.data, islice, islice, jslice, jslice)
    uu3 = wind_component_calc(utmp, eta_v, grid.bgrid1.data, grid.bgrid2.data, grid.ee1.data,  islice, islice_p1, jslice, jslice)
    pa1, pa2 = lon_lat_midpoint(grid.bgrid1.data[0:-1, :], grid.bgrid1.data[1:, :], grid.bgrid2.data[0:-1, :], grid.bgrid2.data[1:, :], np)
    uu2 = wind_component_calc(utmp, eta_v, pa1, pa2, grid.es1.data[:-1, :,:],islice,  slice(nhalo, -nhalo), jslice, jslice)
    u[islice, jslice,:] = 0.25 * (uu1 + 2.0 * uu2 + uu3)[islice, jslice,:]
    
    # Temperature
    full_nx, full_ny = grid.agrid2.data.shape
    nx = full_nx - 2* nhalo - 1
    ny = full_ny - 2 * nhalo - 1
    islice = slice(nhalo, nhalo + nx)
    jslice = slice(nhalo, nhalo + ny)
    eta_s = 1.0 # surface level
    eta_t = 0.2 # tropopause
    t_0 = 288.0
    delta_t = 480000.0
    lapse_rate = 0.005
    ii = 0
    jj = 0
    kk = 0
    t_mean = t_0 * eta[:] ** (constants.RDGAS * lapse_rate / constants.GRAV)
    t_mean[eta_t > eta] = t_mean[eta_t > eta] + delta_t*(eta_t - eta[eta_t > eta])**5.0
    # A-grid cell center temperature
  
   
    pt1 = compute_temperature_component(eta, eta_v, t_mean, grid.agrid2.data, islice, jslice)
    p1, p2 = lon_lat_midpoint(grid.bgrid1.data[0:-1, :], grid.bgrid1.data[1:, :], grid.bgrid2.data[0:-1, :], grid.bgrid2.data[1:, :], np)
    pt2 = compute_temperature_component(eta, eta_v, t_mean, p2, islice, jslice)
    p1, p2 = lon_lat_midpoint(grid.bgrid1.data[1:, 0:-1], grid.bgrid1.data[1:, 1:], grid.bgrid2.data[1:, 0:-1], grid.bgrid2.data[1:, 1:], np)
    pt3 = compute_temperature_component(eta, eta_v, t_mean, p2, islice, jslice)
    p1, p2 = lon_lat_midpoint(grid.bgrid1.data[0:-1, 1:], grid.bgrid1.data[1:, 1:], grid.bgrid2.data[0:-1, 1:], grid.bgrid2.data[1:, 1:], np)
    pt4 = compute_temperature_component(eta, eta_v, t_mean, p2, islice, jslice)
    p1, p2 = lon_lat_midpoint(grid.bgrid1.data[:, 0:-1], grid.bgrid1.data[:, 1:], grid.bgrid2.data[:, 0:-1], grid.bgrid2.data[:, 1:], np)
    pt5 = compute_temperature_component(eta, eta_v, t_mean, p2, islice, jslice)
    pt6 = compute_temperature_component(eta, eta_v, t_mean, grid.bgrid2, islice, jslice)
    pt7 = compute_temperature_component(eta, eta_v, t_mean, grid.bgrid2.data[1:,:], islice, jslice)
    pt8 = compute_temperature_component(eta, eta_v, t_mean, grid.bgrid2.data[1:,1:], islice, jslice)
    pt9 = compute_temperature_component(eta, eta_v, t_mean, grid.bgrid2.data[:,1:], islice, jslice)
    pt[:] = 1.0
    pt[islice, jslice, :] =  0.25 * pt1 + 0.125 * (pt2 + pt3 + pt4 + pt5) + 0.0625 * (pt6 + pt7 + pt8 + pt9)
    # WARNING untested
    near_center_adjustment(pt, pt0, grid.agrid1.data, grid.agrid2.data, islice, islice, jslice, jslice)


    # TODO adjust delz!
    # if not adiabatic:
    pt[islice, jslice, :] = pt[islice, jslice, :]/(1. + constants.ZVIR * qvapor[islice, jslice, :])
