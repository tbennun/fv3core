import math
import numpy as np
import fv3core.utils.global_constants as constants
from fv3core.grid import lon_lat_midpoint, great_circle_distance_lon_lat
import fv3gfs.util as fv3util 
nhalo = 3
# maximum amplitude - close to wind speed of zonal-mean time-mean jet stream in troposphere
u0 = 35.0
# indicates perturbation location 20E, 40N
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
R = constants.RADIUS / 10.0 # specifically for test case == 13, otherwise R = 1


def horizontal_compute_shape(full_array):
    full_nx, full_ny, _ = full_array.shape
    nx = full_nx - 2*  nhalo - 1
    ny = full_ny - 2 * nhalo - 1
    return nx, ny


def compute_eta(eta, eta_v, ak, bk):
    """
    Equation (1) Jablonowski & Williamson Baroclinic test case Perturbation. JRMS 2006
    eta is the vertical coordinate and eta_v is an auxiliary vertical coordinate
    """
    eta[:-1] = 0.5 * ((ak[:-1] + ak[1:])/1.e5 + bk[:-1] + bk[1:])
    eta_v[:-1] = (eta[:-1] - eta_0) * math.pi * 0.5

    
def setup_pressure_fields(eta, eta_v, delp, ps, pe, peln, pk, pkz, qvapor, ak, bk, ptop, latitude_agrid, adiabatic):
    nx, ny = horizontal_compute_shape(delp)
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


def zonal_wind(eta_v, latitude):
    """
    Equation (2) Jablonowski & Williamson Baroclinic test case Perturbation. JRMS 2006
    Returns the zonal wind u
    """
    return u0 * np.cos(eta_v[:])**(3.0/2.0) * np.sin(2.0 * latitude[:, :, None])**2.0


def apply_perturbation(u_component, up, longitude, latitude):
    """
    Apply a Gaussian perturbation to intiate a baroclinic wave in Jablonowski & Williamson 2006
    up is the maximum amplitude of the perturbation
    modifies u_component to include the perturbation of radius R
    """
    r = np.zeros((u_component.shape[0], u_component.shape[1], 1))
    # Equation (11), distance from perturbation at 20E, 40N
    r = great_circle_distance_lon_lat(pcen[0], longitude,  pcen[1], latitude, constants.RADIUS, np)[:, :, None]
    r3d = np.repeat(r, u_component.shape[2], axis=2)
    near_perturbation = ((r3d/R)**2.0 < 40.0)
    # Equation(10) perturbation applied to u_component
    u_component[near_perturbation] = u_component[near_perturbation] + up * np.exp(-(r3d[near_perturbation]/R)**2.0)
   

def local_coordinate_transformation(u_component, longitude, grid_vector_component):
    """
    Transform the zonal wind component to the cubed sphere grid using a grid vector
    """
    return u_component * (grid_vector_component[:, :, 1]*np.cos(longitude) - grid_vector_component[:, :, 0]*np.sin(longitude))[:,:,None]

def wind_component_calc(shape, eta_v, longitude, latitude, grid_vector_component, islice, islice_grid, jslice, jslice_grid):
    u_component = np.zeros(shape)
    grid_slice = (islice_grid, jslice_grid)
    slice_3d = (islice, jslice, slice(None))
    u_component[slice_3d] = zonal_wind(eta_v, latitude[grid_slice])
    apply_perturbation(u_component[slice_3d], u1, longitude[grid_slice], latitude[grid_slice])
    u_component[slice_3d] = local_coordinate_transformation(u_component[slice_3d], longitude[grid_slice], grid_vector_component[islice_grid, jslice_grid, :])
    return u_component

def initialize_zonal_wind(u, eta, eta_v, longitude, latitude, east_grid_vector_component, center_grid_vector_component, islice, islice_grid, jslice, jslice_grid, axis):
    shape = u.shape
    uu1 = wind_component_calc(shape, eta_v, longitude, latitude,  east_grid_vector_component, islice, islice, jslice, jslice_grid)
    uu3 = wind_component_calc(shape, eta_v, longitude, latitude,  east_grid_vector_component, islice, islice_grid, jslice, jslice)
    lower_slice = (slice(None),) * axis + (slice(0, -1),) 
    upper_slice = (slice(None),) * axis + (slice(1, None),)
    pa1, pa2 = lon_lat_midpoint(longitude[lower_slice], longitude[upper_slice], latitude[lower_slice], latitude[upper_slice], np)
    uu2 = wind_component_calc(shape, eta_v, pa1, pa2, center_grid_vector_component,islice, islice, jslice, jslice)
    u[islice, jslice,:] = 0.25 * (uu1 + 2.0 * uu2 + uu3)[islice, jslice,:]
    
def horizontally_averaged_temperature(eta):
    """
    Equations (4) and (5) Jablonowski & Williamson Baroclinic test case Perturbation. JRMS 2006
    """
    # for troposphere:
    t_mean = t_0 * eta[:] ** (constants.RDGAS * lapse_rate / constants.GRAV)
    # above troposphere
    t_mean[eta_t > eta] = t_mean[eta_t > eta] + delta_t*(eta_t - eta[eta_t > eta])**5.0
    return t_mean

def compute_temperature_component(eta, eta_v, t_mean, latitude):
    """
    Equation (6) Jablonowski & Williamson Baroclinic test case Perturbation. JRMS 2006
    The total temperature distribution from the horizontal-mean temperature and a horizontal variation at each level
    """
    lat = latitude[:, :, None]
    return t_mean + 0.75*(eta[:] * math.pi * u0 / constants. RDGAS) * np.sin(eta_v[:])*np.sqrt(np.cos(eta_v[:])) * (( -2.0 * (np.sin(lat)**6.0) *(np.cos(lat)**2.0 + 1.0/3.0) + 10.0/63.0 ) *2.0*u0*np.cos(eta_v[:])**(3.0/2.0) + ((8.0/5.0)*(np.cos(lat)**3.0)*(np.sin(lat)**2.0 + 2.0/3.0) - math.pi/4.0 ) * constants.RADIUS * constants.OMEGA )


def compute_surface_geopotential_component(latitude):
    """
    Equation (7) at the surface level. Jablonowski & Williamson Baroclinic test case Perturbation. JRMS 2006
    The surface geopotential distribution
    """
    u_comp = u0 * (np.cos((eta_s-eta_0)*math.pi/2.0))**(3.0/2.0)
    return  u_comp * (( -2.0*(np.sin(latitude)**6.0) * (np.cos(latitude)**2.0 + 1.0/3.0) + 10.0/63.0 ) * u_comp + ((8.0/5.0)*(np.cos(latitude)**3.0)*(np.sin(latitude)**2.0 + 2.0/3.0) - math.pi/4.0 )*constants.RADIUS * constants.OMEGA)




def compute_grid_edge_midpoint_latitude_components(longitude, latitude):
    lon, lat_avg_x_lower = lon_lat_midpoint(longitude[0:-1, :], longitude[1:, :], latitude[0:-1, :], latitude[1:, :], np)
    lon, lat_avg_y_right = lon_lat_midpoint(longitude[1:, 0:-1], longitude[1:, 1:], latitude[1:, 0:-1], latitude[1:, 1:], np)
    lon, lat_avg_x_upper = lon_lat_midpoint(longitude[0:-1, 1:], longitude[1:, 1:], latitude[0:-1, 1:], latitude[1:, 1:], np)
    lon, lat_avg_y_left  = lon_lat_midpoint(longitude[:, 0:-1], longitude[:, 1:], latitude[:, 0:-1], latitude[:, 1:], np)
    return  lat_avg_x_lower, lat_avg_y_right, lat_avg_x_upper,  lat_avg_y_left

def cell_average_nine_point(pt1, pt2, pt3, pt4, pt5, pt6, pt7, pt8, pt9):
    """
    9-point average: should be 2nd order accurate for a rectangular cell
    9  4  8
    5  1  3          
    6  2  7
    """
    return 0.25 * pt1 + 0.125 * (pt2 + pt3 + pt4 + pt5) + 0.0625 * (pt6 + pt7 + pt8 + pt9)

def cell_average_nine_components(component_function, component_args, latitude, latitude_agrid, lat2, lat3, lat4, lat5, grid_slice):
    """
    Componet 9 cell components of a variable, calling a component_function definiting the equation with component_args
    (arguments unique to that component_function), and precomputed latitude arrays 
    """
    pt1 = component_function(*component_args, latitude=latitude_agrid[grid_slice])
    pt2 = component_function(*component_args, latitude=lat2[grid_slice])
    pt3 = component_function(*component_args, latitude=lat3[grid_slice])
    pt4 = component_function(*component_args, latitude=lat4[grid_slice])
    pt5 = component_function(*component_args, latitude=lat5[grid_slice])
    pt6 = component_function(*component_args, latitude=latitude[grid_slice])
    pt7 = component_function(*component_args, latitude=latitude[1:,:][grid_slice])
    pt8 = component_function(*component_args, latitude=latitude[1:,1:][grid_slice])
    pt9 = component_function(*component_args, latitude=latitude[:,1:][grid_slice])
    return cell_average_nine_point(pt1, pt2, pt3, pt4, pt5, pt6, pt7, pt8, pt9)


def initialize_nonhydrostatic_delz(delz, pt, peln, islice, jslice):
    """
    For the FV3 model, geopotential is computed each timestep but is not part of the intial state
    Here we compute nonhydrostatic delz describing the thickness of each vertical layer
    Thus equations 8 and 9 and 7 above the surface in Jablonowski & Williamson Baroclinic
    are not computed.
    Here delz 
    """
    upper_slice = (islice, jslice, slice(0,-1))
    lower_slice = (islice, jslice, slice(1, None))
    delz[:] = 1.e30
    delz[upper_slice] = constants.RDGAS/constants.GRAV * pt[upper_slice]*(peln[upper_slice]-peln[lower_slice])


def nonadiabatic_moisture_adjusted_temperature(pt, qvapor, slice_3d):
    """
    Update initial temperature to include water vapor contribution
    """
    pt[slice_3d] = pt[slice_3d]/(1. + constants.ZVIR * qvapor[slice_3d])
    
def baroclinic_initialization(peln, qvapor, delp, u, v, pt, phis, delz, w, eta, eta_v, longitude, latitude, longitude_agrid, latitude_agrid, ee1, ee2, es1, ew2, ptop, adiabatic, hydrostatic):
  
    nx, ny = horizontal_compute_shape(delp)
  
    # Equation (2) for v
    # Although meridional wind is 0 in this scheme
    # on the cubed sphere grid, v is not 0 on every tile
    initialize_zonal_wind(v, eta, eta_v, longitude, latitude,
                          east_grid_vector_component=ee2,
                          center_grid_vector_component=ew2,
                          islice=slice(nhalo, nhalo + nx + 1),
                          islice_grid=slice(nhalo, nhalo + nx + 1),
                          jslice=slice(nhalo, nhalo + ny),
                          jslice_grid=slice(nhalo + 1, nhalo + ny + 1),
                          axis=1)

    initialize_zonal_wind(u, eta, eta_v, longitude, latitude,
                          east_grid_vector_component=ee1,
                          center_grid_vector_component=es1,
                          islice=slice(nhalo, nhalo + nx),
                          islice_grid=slice(nhalo + 1, nhalo + nx + 1),
                          jslice=slice(nhalo, nhalo + ny + 1),
                          jslice_grid=slice(nhalo, nhalo + ny + 1),
                          axis=0)
    # slice the standard compute domain for the rest of the variables
    islice = slice(nhalo, nhalo + nx)
    jslice = slice(nhalo, nhalo + ny)
    slice_3d = (islice, jslice, slice(None))
    grid_slice = slice_3d[0:2]
    # We don't compute Equation 3, relative vorticity as the model is not in vorticity-divergence form

   
    # Compute cell latitudes in the midpoint of each cell edge
    lat2, lat3, lat4, lat5 =  compute_grid_edge_midpoint_latitude_components(longitude, latitude)
   
    # initialize temperature
    pt[:] = 1.0
    t_mean = horizontally_averaged_temperature(eta)
    pt[slice_3d] = cell_average_nine_components(compute_temperature_component, [eta, eta_v, t_mean], latitude, latitude_agrid, lat2, lat3, lat4, lat5, grid_slice)

    # initialize surface seopotential
    phis[:] =  1.e25
    phis[grid_slice] = cell_average_nine_components(compute_surface_geopotential_component, [], latitude, latitude_agrid, lat2, lat3, lat4, lat5, grid_slice)
    
    if not hydrostatic:
        w[:] = 1.e30
        # vertical velocity is set to 0 for nonhydrostatic setups
        w[slice_3d] = 0.0
        initialize_nonhydrostatic_delz(delz, pt, peln, islice, jslice)
        
    if not adiabatic:
        nonadiabatic_moisture_adjusted_temperature(pt, qvapor, slice_3d)


def p_var(delp, delz, pt, ps, qvapor, pe, peln, pk, pkz, ptop, moist_phys, make_nh, hydrostatic=False,  adjust_dry_mass=False):
    """
    Computes auxiliary pressure variables for a hydrostatic state.
    The variables are: surfce, interface, layer-mean pressure, exener function
    Given (ptop, delp) computes (ps, pk, pe, peln, pkz)
    """
    assert(not adjust_dry_mass)
    assert(not hydrostatic)

    pek = ptop ** constants.KAPPA
    
    nx, ny = horizontal_compute_shape(delp)
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
    nx, ny = horizontal_compute_shape(delp)
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
    baroclinic_initialization(peln, qvapor, delp, u, v, pt, phis, delz, w, eta, eta_v,  longitude, latitude, longitude_agrid, latitude_agrid, ee1, ee2, es1, ew2, ptop, adiabatic, hydrostatic)
    p_var( delp, delz, pt, ps, qvapor, pe, peln, pk, pkz, ptop, moist_phys, make_nh=(not hydrostatic), hydrostatic=hydrostatic)
  
    # halo update phis
    # halo update u and v
