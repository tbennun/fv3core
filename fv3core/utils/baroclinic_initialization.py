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
def zonal_wind(utmp, eta_v, latitude_dgrid, islice, jslice, jslice_grid):
    utmp[islice, jslice,:] =  u0 * np.cos(eta_v[:])**(3.0/2.0) * np.sin(2.0 * latitude_dgrid[islice,jslice_grid, None])**2.0

"""
call mid_pt_sphere(agrid(i-1,j,1:2), agrid(i,j,  1:2), py(1,j)) --> 
     py0, py1 = lon_lat_midpoint(
        agrid[nhalo-1, nhalo-2:-nhalo+2, 0], 
        agrid[nhalo, nhalo-2:-nhalo+2, 0], 
        agrid[nhalo-1, nhalo-2:-nhalo+2, 1], 
        agrid[nhalo, nhalo-2:-nhalo+2, 1], np
    )
"""
def wind_component_calc(utmp, eta_v, longitude_dgrid, latitude_dgrid, ee2, islice, jslice, jslice_grid):
    ii=3
    jj=3
    kk=0
    vv = np.zeros(utmp.shape)
    r = np.zeros((utmp.shape[0], utmp.shape[1]))
    zonal_wind(utmp, eta_v, latitude_dgrid, islice, jslice, jslice_grid)
    #print('99999999999gaaaa', islice, jslice,  r[islice, jslice].shape,  great_circle_distance_lon_lat(pcen[1], latitude_dgrid[islice,jslice_grid],  pcen[0], latitude_dgrid[islice,jslice_grid], constants.RADIUS, np).shape)
    r[islice, jslice] = great_circle_distance_lon_lat(pcen[0], longitude_dgrid[islice,jslice_grid],  pcen[1], latitude_dgrid[islice,jslice_grid], constants.RADIUS, np)
   
    # [0]  BAROCLINIC SAMPLE PRE R parts   10105164.047478760       0.34906585039886590       0.69813170079773179        5.6862228671883459      -0.62824771989411210        6371200.0000000000
    #                                       6998544.04526923 [0.3490658503988659,       0.6981317007977318]        5.686222867188346       -0.6282477198941121         6371200.0
    #print("radius is so wrong", r[ii,jj], pcen,  longitude_dgrid[ii,jj+1],  latitude_dgrid[ii,jj+1],  constants.RADIUS, r.shape, utmp.shape)
    r3d = np.repeat(r[:,:, None], utmp.shape[2], axis=2)
    adjust_bool = (-(r3d/r0)**2.0 > -40.0)
    #print('PRE baroclinic sample vv', r[ii,jj], utmp[ii,jj,kk], adjust_bool[ii,jj,kk])
    utmp[islice, jslice,:][adjust_bool[islice, jslice,:]] = utmp[islice, jslice,:][adjust_bool[islice, jslice,:]] + u1 * np.exp(-(r3d[islice, jslice,:][adjust_bool[islice, jslice,:]]/r0)**2.0) 
    vv[islice, jslice, :] = utmp[islice, jslice, :]*(ee2[islice, jslice_grid, 1]*np.cos(longitude_dgrid[islice, jslice_grid]) - ee2[islice, jslice_grid, 0]*np.sin(longitude_dgrid[islice, jslice_grid]))[:,:,None]
    #print('baroclinic sample vv', vv[ii,jj,kk], r[ii,jj], utmp[ii,jj,kk], adjust_bool[ii,jj,kk], 'eeeee', ee2[ii,jj,1], ee2[ii,jj,0], 'lon', longitude_dgrid[ii,jj])
    return vv

#def compute_temperature()
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

def baroclinic_initialization(delp, u, v, pt, eta, eta_v, grid, ptop):
    nx, ny, nz = grid.domain_shape_compute()
    shape = (nx, ny, nz)
  
  
    utmp = np.zeros(u.shape)
    vv1 = np.zeros(u.shape)
    # Equation (2) for j+1
    #utmp[nhalo:-nhalo,nhalo:-nhalo-1,:] =  u0 * np.cos(eta_v[:])**(3.0/2.0) * np.sin(2.0 * grid[nhalo:-nhalo,nhalo+1:-nhalo,1])**2.0
    nhalo = 3
    islice = slice(nhalo, -nhalo)
    jslice = slice(nhalo, -nhalo - 1)
    jslice_p1 = slice(nhalo+1, -nhalo)
    #zonal_wind(utmp, eta_v, grid.bgrid2.data, islice, jslice, jslice_grid)
  
    # r = great_circle_distance_lon_lat(pcen[0], grid.bgrid1.data[islice,jslice_grid], pcen[1], grid.bgrid2.data[islice,jslice_grid], constants.RADIUS, np)
    #r3d = np.repeat(r[:,:, None], utmp.shape[2], axis=2)
    #adjust_bool = (-(r3d/r0)**2.0 > -40.0)
    #utmp[islice, jslice,:][adjust_bool] = utmp[islice, jslice,:][adjust_bool] + u1 * np.exp(-(r3d[adjust_bool]/r0)**2.0) 
    #vv1[islice, jslice, :] = utmp[islice, jslice, :]*(grid.ee2.data[islice, jslice_grid, 1]*np.cos(grid.bgrid1.data[islice, jslice_grid]) - grid.ee2.data[islice, jslice_grid, 0]*np.sin(grid.bgrid1.data[islice, jslice_grid]))[:,:,None]
  
    vv1 = wind_component_calc(utmp, eta_v, grid.bgrid1.data, grid.bgrid2.data, grid.ee2.data, islice, jslice, jslice_p1)
 
    vv3 = wind_component_calc(utmp, eta_v, grid.bgrid1.data, grid.bgrid2.data, grid.ee2.data,  islice, jslice, jslice)
    pa1, pa2 = lon_lat_midpoint(
        grid.bgrid1.data[:, 0:-1], 
        grid.bgrid1.data[:, 1:], 
        grid.bgrid2.data[:, 0:-1], 
        grid.bgrid2.data[:, 1:],
        np
    )
    #print('pa', pa1[6,3], pa2[6,3])
  
    vv2 = wind_component_calc(utmp, eta_v, pa1, pa2, grid.ew2.data[:,:-1,:],islice, jslice, slice(nhalo, -nhalo))
   
    v[islice, jslice,:] = 0.25 * (vv1 + 2.0 * vv2 + vv3)[islice, jslice,:]
    """

[0]  BAROCLINIC SAMPLE vv2   1.8254847867352852E-014   10382616.335223846        29.895464019314691        1.4980357642426241E-014   5.6862228671883459      -0.68385270837775802    
  -0.35514661284459426       0.52251511566129349  
  -0.35514661284459426       0.5225151156612935
  5.6862228671883459 -0.68385270837775802  
pa 5.686222867188345 -0.683852708377758
baroclinic sample vv 3.48501641104009e-14 7391621.343616784 29.89546401931469 False eeeee  lon 5.686222867188345
   vv[islice, jslice, :] = utmp[islice, jslice, :]*(ee2[islice, jslice_grid, 1]*np.cos(longitude_dgrid[islice, jslice_grid]) - ee2[islice, jslice_grid, 0]*np.sin(longitude_dgrid[islice, jslice_grid]))[:,:,None]

      BAROCLINIC SAMPLE vv2   1.8254847867352852E-014   10382616.335223846        29.895464019314691        1.4980357642426241E-014   5.6862228671883459      -0.68385270837775802 
                              3.48501641104009e-14         7391621.343616784      29.89546401931469         2.327801576395027e-14     5.686222867188345       -0.683852708377758
[0]  BAROCLINIC SAMPLE PRE vv1   10105164.047478760        28.184825888175713      F
PRE baroclinic sample vv       6998544.04526923           28.184825888175713      False
[0]  BAROCLINIC SAMPLE vv1   6.2582885292207707E-015   10105164.047478760        28.184825888175713   2926815.52048365 
                                                       10116411.976445448
                             6.258288529220771e-15     6998544.04526923       28.184825888175713 
    [0]  BAROCLINIC SAMPLE vv1   5.0599487872110027E-015   10919915.125866998        22.787983472597265 
                                                           10919915.125866998     
            baroclinic sample vv 5.059948787211003e-15      8291707.009921413         22.787983472597265
                                  5.059948787211003e-15    9238752.88132445 
    [0]  BAROCLINIC SAMPLE vv3  -3.0753913669697378E-015   11350822.487028997        27.700662828606703
                                -3.0753913669697378e-15    8914648.283784742         27.700662828606703   
                                -3.0753913669697378e-15   
    [0]  BAROCLINIC SAMPLE vv2  -1.4128312958558706E-015   11136216.880731197        25.451305990215495       -2.1027629286761906E-016
                                 -2.5299743936055014e-15   8291707.009921413         22.787983472597265
[0]  BAROCLINIC SAMPLE vv2  -1.4128312958558706E-015   11136216.880731197        25.451305990215495   5.3232542185827052      -0.56418973294785524  
                            -1.4128312958558706e-15 
                            -2.825662591711741e-15       8602010.32545767        25.451305990215495   5.323254218582705       -0.5641897329478552                
pa 5.323254218582705   -0.5128997572253229
pa 5.3232542185827052  -0.56418973294785524   
 pa 5.323254218582705  -0.5641897329478552 

 
    """

    # u
    """
    islice = slice(nhalo, -nhalo - 1)
    jslice = slice(nhalo, -nhalo)
    islice_p1 = slice(nhalo+1, -nhalo)
    uu1 = wind_component_calc(utmp, eta_v, grid.bgrid1.data, grid.bgrid2.data, grid.ee1.data, islice, jslice, jslice_p1)
    
    uu3 = wind_component_calc(utmp, eta_v, grid.bgrid1.data, grid.bgrid2.data, grid.ee1.data,  islice, jslice, jslice)
    pa1, pa2 = lon_lat_midpoint(
        grid.bgrid1.data[:, 0:-1], 
        grid.bgrid1.data[:, 1:], 
        grid.bgrid2.data[:, 0:-1], 
        grid.bgrid2.data[:, 1:],
        np
    )
    
    uu2 = wind_component_calc(utmp, eta_v, pa1, pa2, grid.ew2.data[:,:-1,:],islice, jslice, slice(nhalo, -nhalo))
   
    u[islice, jslice,:] = 0.25 * (uu1 + 2.0 * uu2 + uu3)[islice, jslice,:]
    """
    # Temperature
    eta_s = 1.0 # surface level
    eta_t = 0.2 # tropopause
    t_0 = 288.0
    delta_t = 480000.0
    lapse_rate = 0.005

    t_mean = t_0 * eta[:] ** (constants.RDGAS * lapse_rate / constants.GRAV)
    t_mean[eta_t > eta] = t_mean[eta_t > eta] + delta_t*(eta_t - eta[eta_t > eta])**5.0
    # A-grid cell center temperature
    pt1 = t_mean + 0.75*(eta[:] * math.pi * u0 / constants. RDGAS) * np.sin(eta_v[:])*np.sqrt(np.cos(eta_v[:])) * (( -2.0 * (np.sin(grid.agrid2[islice, jslice, None])**6.0) *(np.cos(grid.agrid2[islice, jslice, None])**2.0 + 1.0/3.0) + 10.0/63.0 ) *2.0*u0*np.cos(eta_v[:])**(3.0/2.0) + ((8.0/5.0)*(np.cos(grid.agrid2[islice, jslice, None])**3.0)*(np.sin(grid.agrid2[islice, jslice, None])**2.0 + 2.0/3.0) - math.pi/4.0 )*constants.RADIUS * constants.OMEGA )
