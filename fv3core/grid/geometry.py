from math import sin
import typing
from fv3core.utils.global_constants import PI
from .gnomonic import lon_lat_to_xyz, xyz_midpoint, normalize_xyz, spherical_cos, get_unit_vector_direction, lon_lat_midpoint, get_lonlat_vect, _vect_cross, great_circle_distance_lon_lat

def get_center_vector(xyz_gridpoints, grid_type, nhalo, tile_partitioner, rank, np):
    '''
    Calculates the unit vector pointing to the center of each grid cell.
    vector1 comes from using the halfway points of the left and top cell edges, while
    vector2 comes from using the halfway points of the bottom and right cell edges
    '''
    if grid_type < 3:
        if False: #ifdef OLD_VECT
            vector1 = xyz_gridpoints[1:, :-1, :] + xyz_gridpoints[1:, 1:, :] - xyz_gridpoints[:-1, :-1, :] - xyz_gridpoints[:-1, 1:, :]
            vector2 = xyz_gridpoints[:-1, 1:, :] + xyz_gridpoints[1:, 1:, :] - xyz_gridpoints[:-1, :-1, :] - xyz_gridpoints[1:, :-1, :]
        else:
            center_points = xyz_midpoint(
                xyz_gridpoints[:-1, :-1, :],
                xyz_gridpoints[1:, :-1, :],
                xyz_gridpoints[:-1, 1:, :],
                xyz_gridpoints[1:, 1:, :])
            
            p1 = xyz_midpoint(xyz_gridpoints[:-1, :-1, :], xyz_gridpoints[:-1, 1:, :])
            p2 = xyz_midpoint(xyz_gridpoints[1:, :-1, :], xyz_gridpoints[1:, 1:, :])
            p3 = np.cross(p1, p2)
            vector1 = normalize_xyz(np.cross( center_points, p3))

            p1 = xyz_midpoint(xyz_gridpoints[:-1, :-1, :], xyz_gridpoints[1:, :-1, :])
            p2 = xyz_midpoint(xyz_gridpoints[:-1, 1:, :], xyz_gridpoints[1:, 1:, :])
            p3 = np.cross(p1, p2)
            vector2 = normalize_xyz(np.cross( center_points, p3))

        #fill ghost on ec1 and ec2:
        if tile_partitioner.on_tile_left(rank):
            if tile_partitioner.on_tile_bottom(rank):
                _fill_ghost(vector1, 0., nhalo, "sw")
                _fill_ghost(vector2, 0., nhalo, "sw")
            if tile_partitioner.on_tile_top(rank):
                _fill_ghost(vector1, 0., nhalo, "nw")
                _fill_ghost(vector2, 0., nhalo, "nw")
        if tile_partitioner.on_tile_right(rank):
            if tile_partitioner.on_tile_bottom(rank):
                _fill_ghost(vector1, 0., nhalo, "se")
                _fill_ghost(vector2, 0., nhalo, "se")
            if tile_partitioner.on_tile_top(rank):
                _fill_ghost(vector1, 0., nhalo, "ne")
                _fill_ghost(vector2, 0., nhalo, "ne")
    else:
        shape_dgrid = xyz_gridpoints.shape
        vector1 = np.zeros((shape_dgrid[0]-1, shape_dgrid[1]-1, 3))
        vector2 = np.zeros((shape_dgrid[0]-1, shape_dgrid[1]-1, 3))
        vector1[:,:,0] = 1
        vector2[:,:,1] = 1

    return vector1, vector2

def calc_unit_vector_west(xyz_dgrid, xyz_agrid, grid_type, nhalo, tile_partitioner, rank, np):
    """
    Calculates the cartesian unit vector pointing west from every grid cell.
    The first set of values is the horizontal component, the second is the vertical component as 
    defined by the cell edges -- in a non-spherical grid these will be x and y unit vectors.
    """
    if grid_type < 3:
        pp = xyz_midpoint(xyz_dgrid[1:-1,:-1,:3], xyz_dgrid[1:-1, 1:, :3])
        p2 = np.cross(xyz_agrid[:-1,:,:3], xyz_agrid[1:,:,:3])
        if tile_partitioner.on_tile_left(rank):
            p2[nhalo] = np.cross(pp[nhalo], xyz_agrid[nhalo,:,:3])
        if tile_partitioner.on_tile_right(rank):
            p2[-nhalo] = np.cross(pp[nhalo], xyz_agrid[-nhalo-1,:,:3])

        
        ew1 = normalize_xyz(np.cross(p2, pp))
        
        p1 = np.cross(xyz_dgrid[1:-1, :-1, :], xyz_dgrid[1:-1, 1:, :])
        ew2 = normalize_xyz(np.cross(p1, pp))

        ew = np.stack((ew1, ew2), axis=-1)

        #fill ghost on ew:
        if tile_partitioner.on_tile_left(rank):
            if tile_partitioner.on_tile_bottom(rank):
                _fill_ghost(ew, 0., nhalo, "sw")
            if tile_partitioner.on_tile_top(rank):
                _fill_ghost(ew, 0., nhalo, "nw")
        if tile_partitioner.on_tile_right(rank):
            if tile_partitioner.on_tile_bottom(rank):
                _fill_ghost(ew, 0., nhalo, "se")
            if tile_partitioner.on_tile_top(rank):
                _fill_ghost(ew, 0., nhalo, "ne")
    
    else:
        ew1 = np.zeros(xyz_dgrid.shape[0], xyz_agrid.shape[1], 3)
        ew2 = np.zeros(xyz_dgrid.shape[0], xyz_agrid.shape[1], 3)
        ew1[:,:,1] = 1.
        ew2[:,:,2] = 1.
        ew = np.stack((ew1, ew2), axis=-1)
    
    return ew

def calc_unit_vector_south(xyz_dgrid, xyz_agrid, grid_type, nhalo, tile_partitioner, rank, np):
    """
    Calculates the cartesian unit vector pointing south from every grid cell.
    The first set of values is the horizontal component, the second is the vertical component as 
    defined by the cell edges -- in a non-spherical grid these will be x and y unit vectors.
    """
    if grid_type < 3:
        pp = xyz_midpoint(xyz_dgrid[:-1, 1:-1, :3], xyz_dgrid[1:, 1:-1, :3])
        p2 = np.cross(xyz_agrid[:,:-1,:3], xyz_agrid[:, 1:, :3])
        if tile_partitioner.on_tile_bottom(rank):
            p2[:,nhalo] = np.cross(pp[:,nhalo], xyz_agrid[:, nhalo, :3])
        if tile_partitioner.on_tile_top(rank):
            p2[:,-nhalo] = np.cross(pp[:,-nhalo], xyz_agrid[:, -nhalo-1, :3])
        
        es2 = normalize_xyz(np.cross(p2, pp))
        
        p1 = np.cross(xyz_dgrid[:-1, 1:-1, :], xyz_dgrid[1:, 1:-1, :])
        es1 = normalize_xyz(np.cross(p1, pp))

        es = np.stack((es1, es2), axis=-1)

        #fill ghost on es:
        if tile_partitioner.on_tile_left(rank):
            if tile_partitioner.on_tile_bottom(rank):
                _fill_ghost(es, 0., nhalo, "sw")
            if tile_partitioner.on_tile_top(rank):
                _fill_ghost(es, 0., nhalo, "nw")
        if tile_partitioner.on_tile_right(rank):
            if tile_partitioner.on_tile_bottom(rank):
                _fill_ghost(es, 0., nhalo, "se")
            if tile_partitioner.on_tile_top(rank):
                _fill_ghost(es, 0., nhalo, "ne")
    else:
        es1 = np.zeros(xyz_agrid.shape[0], xyz_dgrid.shape[1], 3)
        es2 = np.zeros(xyz_agrid.shape[0], xyz_dgrid.shape[1], 3)
        es1[:,:,1] = 1.
        es2[:,:,2] = 1.
        es = np.stack((es1, es2), axis=-1)
    
    return es

def calculate_supergrid_cos_sin(xyz_dgrid, xyz_agrid, ec1, ec2, grid_type, nhalo, tile_partitioner, rank, np):
    """
    Calculates the cosine and sine of the grid angles at each of the following points in a supergrid cell:
    9---4---8
    |       |
    1   5   3
    |       |
    6---2---7
    """
    big_number = 1.e8
    tiny_number = 1.e-8

    shape_a = xyz_agrid.shape
    cos_sg = np.zeros((shape_a[0], shape_a[1], 9))+big_number
    sin_sg = np.zeros((shape_a[0], shape_a[1], 9))+tiny_number

    if grid_type < 3:
        cos_sg[:, :, 5] = spherical_cos(xyz_dgrid[:-1, :-1, :], xyz_dgrid[1:, :-1, :], xyz_dgrid[:-1, 1:, :], np)
        cos_sg[:, :, 6] = -1 * spherical_cos(xyz_dgrid[1:, :-1, :], xyz_dgrid[:-1, :-1, :], xyz_dgrid[1:, 1:, :], np)
        cos_sg[:, :, 7] = spherical_cos(xyz_dgrid[1:, 1:, :], xyz_dgrid[1:, :-1, :], xyz_dgrid[:-1, 1:, :], np)
        cos_sg[:, :, 8] = -1 * spherical_cos(xyz_dgrid[:-1, 1:, :], xyz_dgrid[:-1, :-1, :], xyz_dgrid[1:, 1:, :], np)

        midpoint = xyz_midpoint(xyz_dgrid[:-1, :-1, :], xyz_dgrid[:-1, 1:, :])
        cos_sg[:, :, 0] = spherical_cos(midpoint, xyz_agrid[:, :, :], xyz_dgrid[:-1, 1:, :], np)
        midpoint = xyz_midpoint(xyz_dgrid[:-1, :-1, :], xyz_dgrid[1:, :-1, :])
        cos_sg[:, :, 1] = spherical_cos(midpoint, xyz_dgrid[1:, :-1, :], xyz_agrid[:, :, :], np)
        midpoint = xyz_midpoint(xyz_dgrid[1:, :-1, :], xyz_dgrid[1:, 1:, :])
        cos_sg[:, :, 2] = spherical_cos(midpoint, xyz_agrid[:, :, :], xyz_dgrid[1:, :-1, :], np)
        midpoint = xyz_midpoint(xyz_dgrid[:-1, 1:, :], xyz_dgrid[1:, 1:, :])
        cos_sg[:, :, 3] = spherical_cos(midpoint, xyz_dgrid[:-1, 1:, :], xyz_agrid[:, :, :], np)

        cos_sg[:, :, 4] = np.sum(ec1*ec2, axis=-1)

        sin_sg_tmp = 1.-cos_sg**2
        sin_sg_tmp[sin_sg_tmp < 0.] = 0.
        sin_sg = np.sqrt(sin_sg_tmp)
        sin_sg[sin_sg > 1.] = 1.

        #Adjust for corners:
        if tile_partitioner.on_tile_left(rank):
            if tile_partitioner.on_tile_bottom(rank): #southwest corner
                sin_sg[nhalo-1,:nhalo,2] = sin_sg[:nhalo, nhalo, 1]
                sin_sg[:nhalo, nhalo-1, 3] = sin_sg[nhalo, :nhalo, 0]
            if tile_partitioner.on_tile_top(rank): #northwest corner
                sin_sg[nhalo -1, -nhalo:, 2] = sin_sg[:nhalo:-1, -nhalo-1, 3]
                sin_sg[:nhalo, -nhalo, 1] = sin_sg[nhalo, -nhalo-2:-nhalo+1:-1, 0]
        if tile_partitioner.on_tile_right(rank):
            if tile_partitioner.on_tile_bottom(rank): #southeast corner
                sin_sg[-nhalo, :nhalo, 0]  = sin_sg[-nhalo::-1, nhalo, 1]
                sin_sg[-nhalo:, nhalo-1, 3] = sin_sg[-nhalo-1, :nhalo:-1, 2]
            if tile_partitioner.on_tile_top(rank): #northeast corner
                sin_sg[-nhalo, -nhalo:, 0] = sin_sg[-nhalo:, -nhalo-1, 3]
                sin_sg[-nhalo:, -nhalo, 1] = sin_sg[-nhalo-1, -nhalo:, 2]

    else:
        cos_sg[:] = 0.
        sin_sg[:] = 1.

    return cos_sg, sin_sg

def calculate_l2c_vu(dgrid, xyz_dgrid, nhalo, np):
    #AAM correction
        midpoint_y = np.array(lon_lat_midpoint(dgrid[nhalo:-nhalo, nhalo:-nhalo-1, 0], dgrid[nhalo:-nhalo, nhalo+1:-nhalo, 0], dgrid[nhalo:-nhalo, nhalo:-nhalo-1, 1], dgrid[nhalo:-nhalo, nhalo+1:-nhalo, 1], np))
        unit_dir_y = get_unit_vector_direction(xyz_dgrid[nhalo:-nhalo+1, nhalo:-nhalo, :], xyz_dgrid[nhalo:-nhalo+1, nhalo+1:-nhalo+1, :], np)
        ex, _ = get_lonlat_vect(midpoint_y)
        l2c_v = np.cos(midpoint_y[1] * np.sum(unit_dir_y * ex, axis=0))

        midpoint_x = np.array(lon_lat_midpoint(dgrid[nhalo:-nhalo-1, nhalo:-nhalo, 0], dgrid[nhalo+1:-nhalo, nhalo:-nhalo, 0], dgrid[nhalo:-nhalo-1, nhalo:-nhalo, 1], dgrid[nhalo+1:-nhalo, nhalo:-nhalo, 1], np))
        unit_dir_x = get_unit_vector_direction(xyz_dgrid[nhalo:-nhalo, nhalo:-nhalo+1, :], xyz_dgrid[nhalo+1:-nhalo+1, nhalo:-nhalo+1, :], np)
        ex, _ = get_lonlat_vect(midpoint_x)
        l2c_u = np.cos(midpoint_x[1] * np.sum(unit_dir_x * ex, axis=0))

        return l2c_v, l2c_u

def calculate_trig_uv(xyz_dgrid, cos_sg, sin_sg, nhalo, tile_partitioner, rank, np):
    '''
    Calculates more trig quantities
    '''

    big_number = 1.e8
    tiny_number = 1.e-8

    cosa = sina = rsina = np.zeros(xyz_dgrid[:,:,0].shape)+big_number
    cosa_u = sina_u = np.zeros(xyz_dgrid[:,:,0].shape)+big_number
    cosa_v = sina_v = np.zeros(xyz_dgrid[:,:,0].shape)+big_number

    cross_vect_x = _vect_cross(xyz_dgrid[nhalo-1:-nhalo-1, nhalo:-nhalo, :], xyz_dgrid[nhalo+1:-nhalo+1, nhalo:-nhalo, :])
    if tile_partitioner.on_tile_left(rank):
        cross_vect_x[0,:] = _vect_cross(xyz_dgrid[nhalo, nhalo:-nhalo, :], xyz_dgrid[nhalo+1, nhalo:-nhalo, :])
    if tile_partitioner.on_tile_right(rank):
        cross_vect_x[-1, :] = _vect_cross(xyz_dgrid[-2, nhalo:-nhalo, :], xyz_dgrid[-1, nhalo:-nhalo, :])
    unit_x_vector = normalize_xyz(_vect_cross(cross_vect_x, xyz_dgrid[nhalo:-nhalo, nhalo:-nhalo]))

    cross_vect_y = _vect_cross(xyz_dgrid[nhalo:-nhalo, nhalo-1:-nhalo-1, :], xyz_dgrid[nhalo:-nhalo, nhalo+1:-nhalo+1, :])
    if tile_partitioner.on_tile_bottom(rank):
        cross_vect_y[:,0] = _vect_cross(xyz_dgrid[nhalo:-nhalo, nhalo, :], xyz_dgrid[nhalo:-nhalo, nhalo+1, :])
    if tile_partitioner.on_tile_top(rank):
        cross_vect_y[:, -1] = _vect_cross(xyz_dgrid[nhalo:-nhalo, -2, :], xyz_dgrid[nhalo:-nhalo, -1, :])
    unit_y_vector = normalize_xyz(_vect_cross(cross_vect_y, xyz_dgrid[nhalo:-nhalo, nhalo:-nhalo]))

    if False: #TEST_FP
        tmp1 = np.sum(unit_x_vector*unit_y_vector, axis=0)
        cosa[nhalo:-nhalo, nhalo:-nhalo] = np.clip(np.abs(tmp1), None, 1.)
        cosa[tmp1 < 0]*=-1
        sina[nhalo:-nhalo, nhalo:-nhalo] = np.sqrt(np.clip(1.-cosa**2, 0., None))
    else:
        cosa[nhalo:-nhalo, nhalo:-nhalo] = 0.5*(cos_sg[nhalo-1:-nhalo-1, nhalo-1:-nhalo-1, 7] + cos_sg[nhalo:-nhalo, nhalo:-nhalo, 5])
        sina[nhalo:-nhalo, nhalo:-nhalo] = 0.5*(sin_sg[nhalo-1:-nhalo-1, nhalo-1:-nhalo-1, 7] + sin_sg[nhalo:-nhalo, nhalo:-nhalo, 5])

    cosa_u[1:,:] = 0.5*(cos_sg[:-1,:,2] + cos_sg[1:,:,0])
    sina_u[1:,:] = 0.5*(sin_sg[:-1,:,2] + sin_sg[1:,:,0])
    rsin_u = 1./sina_u**2

    cosa_v[:,1:] = 0.5*(cos_sg[:,:-1,2] + cos_sg[:,1:,1])
    sina_v[:,1:] = 0.5*(sin_sg[:,:-1,2] + sin_sg[:,1:,1])
    rsin_v = 1./sina_v**2

    cosa_s = cos_sg[:,:,4]
    rsin2 = 1./sin_sg[:,:,4]**2
    rsin2[rsin2 < tiny_number] = tiny_number

    rsina[nhalo:-nhalo+1, nhalo:-nhalo+1] = 1./sina[nhalo:-nhalo+1, nhalo:-nhalo+1]**2

    #fill ghost on cosa_s:
    if tile_partitioner.on_tile_left(rank):
        if tile_partitioner.on_tile_bottom(rank):
            _fill_ghost(cosa_s, big_number, nhalo, "sw")
        if tile_partitioner.on_tile_top(rank):
            _fill_ghost(cosa_s, big_number, nhalo, "nw")
    if tile_partitioner.on_tile_right(rank):
        if tile_partitioner.on_tile_bottom(rank):
            _fill_ghost(cosa_s, big_number, nhalo, "se")
        if tile_partitioner.on_tile_top(rank):
            _fill_ghost(cosa_s, big_number, nhalo, "ne")

    # Set special sin values at edges
    if tile_partitioner.on_tile_left(rank):
        rsina[nhalo, nhalo:-nhalo+1] = big_number
        rsin_u[nhalo,:] = 1./sina_u[nhalo,:]
        rsin_v[nhalo,:] = 1./sina_v[nhalo,:]
    if tile_partitioner.on_tile_right(rank):
        rsina[-nhalo, nhalo:-nhalo+1] = big_number
        rsin_u[-nhalo+1,:] = 1./sina_u[-nhalo+1,:]
        rsin_v[-nhalo,:] = 1./sina_v[-nhalo,:]
    if tile_partitioner.on_tile_bottom(rank):
        rsina[nhalo:-nhalo+1, nhalo] = big_number
        rsin_u[:,nhalo] = 1./sina_u[:,nhalo]
        rsin_v[:,nhalo] = 1./sina_v[:,nhalo]
    if tile_partitioner.on_tile_top(rank):
        rsina[:,-nhalo] = big_number
        rsin_u[:,-nhalo] = 1./sina_u[:,-nhalo]
        rsin_v[:,-nhalo+1] = 1./sina_v[:,-nhalo+1]
    
    rsina[rsina < tiny_number] = tiny_number
    rsin_u[rsin_u < tiny_number] = tiny_number
    rsin_v[rsin_v < tiny_number] = tiny_number

    return cosa, sina, cosa_u, cosa_v, cosa_s, sina_u, sina_v, rsin_u, rsin_v, rsina, rsin2, unit_x_vector, unit_y_vector

def supergrid_corner_fix(cos_sg, sin_sg, nhalo, tile_partitioner, rank):
    """
    _fill_ghost overwrites some of the sin_sg 
    values along the outward-facing edge of a tile in the corners, which is incorrect. 
    This function resolves the issue by filling in the appropriate values after the _fill_ghost call
    """
    big_number = 1.e8
    tiny_number = 1.e-8
    
    if tile_partitioner.on_tile_left(rank):
        if tile_partitioner.on_tile_bottom(rank):
            _fill_ghost(sin_sg, tiny_number, nhalo, "sw")
            _fill_ghost(cos_sg, tiny_number, nhalo, "sw")
            _rotate_trig_sg_sw_counterclockwise(sin_sg[:,:,1], sin_sg[:,:,2], nhalo)
            _rotate_trig_sg_sw_counterclockwise(cos_sg[:,:,1], cos_sg[:,:,2], nhalo)
            _rotate_trig_sg_sw_clockwise(sin_sg[:,:,0], sin_sg[:,:,3], nhalo)
            _rotate_trig_sg_sw_clockwise(cos_sg[:,:,0], cos_sg[:,:,3], nhalo)
        if tile_partitioner.on_tile_top(rank):
            _fill_ghost(sin_sg, tiny_number, nhalo, "nw")
            _fill_ghost(cos_sg, tiny_number, nhalo, "nw")
            _rotate_trig_sg_nw_counterclockwise(sin_sg[:,:,0], sin_sg[:,:,1], nhalo)
            _rotate_trig_sg_nw_counterclockwise(cos_sg[:,:,0], cos_sg[:,:,1], nhalo)
            _rotate_trig_sg_nw_clockwise(sin_sg[:,:,3], sin_sg[:,:,2], nhalo)
            _rotate_trig_sg_nw_clockwise(cos_sg[:,:,3], cos_sg[:,:,2], nhalo)
    if tile_partitioner.on_tile_right(rank):
        if tile_partitioner.on_tile_bottom(rank):
            _fill_ghost(sin_sg, tiny_number, nhalo, "se")
            _fill_ghost(cos_sg, tiny_number, nhalo, "se")
            _rotate_trig_sg_se_clockwise(sin_sg[:,:,1], sin_sg[:,:,0], nhalo)
            _rotate_trig_sg_se_clockwise(cos_sg[:,:,1], cos_sg[:,:,0], nhalo)
            _rotate_trig_sg_se_counterclockwise(sin_sg[:,:,2], sin_sg[:,:,3], nhalo)
            _rotate_trig_sg_se_counterclockwise(cos_sg[:,:,2], cos_sg[:,:,3], nhalo)
        if tile_partitioner.on_tile_top(rank):
            _fill_ghost(sin_sg, tiny_number, nhalo, "ne")
            _fill_ghost(cos_sg, tiny_number, nhalo, "ne")
            _rotate_trig_sg_ne_counterclockwise(sin_sg[:,:,3], sin_sg[:,:,0], nhalo)
            _rotate_trig_sg_ne_counterclockwise(cos_sg[:,:,3], cos_sg[:,:,0], nhalo)
            _rotate_trig_sg_ne_clockwise(sin_sg[:,:,2], sin_sg[:,:,1], nhalo)
            _rotate_trig_sg_ne_clockwise(cos_sg[:,:,2], cos_sg[:,:,1], nhalo)


def _rotate_trig_sg_sw_counterclockwise(sg_field_in, sg_field_out, nhalo):
    sg_field_out[nhalo-1, :nhalo] = sg_field_in[:nhalo,nhalo]

def _rotate_trig_sg_sw_clockwise(sg_field_in, sg_field_out, nhalo):    
    sg_field_out[:nhalo, nhalo-1,3] = sg_field_in[nhalo, :nhalo, 0]

def _rotate_trig_sg_nw_counterclockwise(sg_field_in, sg_field_out, nhalo):
    _rotate_trig_sg_sw_clockwise(sg_field_in[:,::-1], sg_field_out[:,::-1], nhalo)

def _rotate_trig_sg_nw_clockwise(sg_field_in, sg_field_out, nhalo):
    _rotate_trig_sg_sw_counterclockwise(sg_field_in[:,::-1], sg_field_out[:,::-1], nhalo)

def _rotate_trig_sg_se_counterclockwise(sg_field_in, sg_field_out, nhalo):
    _rotate_trig_sg_sw_clockwise(sg_field_in[::-1,:], sg_field_out[::-1,:], nhalo)

def _rotate_trig_sg_se_clockwise(sg_field_in, sg_field_out, nhalo):
    _rotate_trig_sg_sw_counterclockwise(sg_field_in[::-1,:], sg_field_out[::-1,:], nhalo)

def _rotate_trig_sg_ne_counterclockwise(sg_field_in, sg_field_out, nhalo):
    _rotate_trig_sg_sw_counterclockwise(sg_field_in[:,::-1], sg_field_out[:,::-1], nhalo)

def _rotate_trig_sg_ne_clockwise(sg_field_in, sg_field_out, nhalo):
    _rotate_trig_sg_sw_clockwise(sg_field_in[::-1,::-1], sg_field_out[::-1,::-1], nhalo)


def calculate_divg_del6(sin_sg, sina_u, sina_v, dx, dy, dxc, dyc, nhalo, tile_partitioner, rank):
    
    divg_u = sina_v * dyc / dx
    del6_u = sina_v * dx / dyc
    divg_v = sina_u * dxc / dy
    del6_v = sina_u * dy / dxc

    if tile_partitioner.on_tile_bottom(rank):
        divg_u[:, nhalo] = 0.5*(sin_sg[:, nhalo, 1] + sin_sg[:, nhalo-1, 3])*dyc[:, nhalo] / dx[:, nhalo]
        del6_u[:, nhalo] = 0.5*(sin_sg[:, nhalo, 1] + sin_sg[:, nhalo-1, 3])*dx[:, nhalo] / dyc[:, nhalo]
    if tile_partitioner.on_tile_top(rank):
        divg_u[:, -nhalo] = 0.5*(sin_sg[:, -nhalo, 1] + sin_sg[:, -nhalo-1, 3])*dyc[:, -nhalo] / dx[:, -nhalo]
        del6_u[:, -nhalo] = 0.5*(sin_sg[:, -nhalo, 1] + sin_sg[:, -nhalo-1, 3])*dx[:, -nhalo] / dyc[:, -nhalo]
    if tile_partitioner.on_tile_left(rank):
        divg_v[nhalo, :] = 0.5*(sin_sg[nhalo, :, 0] + sin_sg[nhalo, :, 2])*dxc[nhalo, :] / dy[nhalo, :]
        del6_v[nhalo, :] = 0.5*(sin_sg[nhalo, :, 0] + sin_sg[nhalo, :, 2])*dy[nhalo, :] / dxc[nhalo, :]
    if tile_partitioner.on_tile_right(rank):
        divg_v[-nhalo, :] = 0.5*(sin_sg[-nhalo-1, :, 0] + sin_sg[-nhalo-2, :, 2])*dxc[-nhalo, :] / dy[-nhalo, :]
        del6_v[-nhalo, :] = 0.5*(sin_sg[-nhalo-1, :, 0] + sin_sg[-nhalo-2, :, 2])*dy[-nhalo, :] / dxc[-nhalo, :]

    return divg_u, divg_v, del6_u, del6_v

def calculate_grid_z(ec1, ec2, vlon, vlat, np):
    z11 = np.sum(ec1 * vlon, axis=-1)
    z12 = np.sum(ec1 * vlat, axis=-1)
    z21 = np.sum(ec2 * vlon, axis=-1)
    z22 = np.sum(ec2 * vlat, axis=-1)
    return z11, z12, z21, z22

def calculate_grid_a(z11, z12, z21, z22,  sin_sg):
    a11 = 0.5*z22/sin_sg[:,:,4]
    a12 = 0.5*z12/sin_sg[:,:,4]
    a21 = 0.5*z21/sin_sg[:,:,4]
    a22 = 0.5*z11/sin_sg[:,:,4]
    return a11, a12, a21, a22

def _global_mx():
    pass

def _global_mx_c():
    pass

def edge_factors(grid, agrid, grid_type, nhalo, tile_partitioner, rank, radius, np):
    """
    Creates interpolation factors from the A grid to the B grid on face edges
    """
    big_number = 1.e8
    npx = agrid.shape[0]
    npy = agrid.shape[1]
    edge_n, edge_s = np.zeros(npx)+big_number
    edge_e, edge_w = np.zeros(npy)+big_number

    if grid_type < 3:
        if tile_partitioner.on_tile_left(rank):
            py = lon_lat_midpoint(agrid[nhalo-1, nhalo:-nhalo, 0], agrid[nhalo, nhalo:-nhalo, 0], agrid[nhalo-1, nhalo:-nhalo, 1], agrid[nhalo, nhalo:-nhalo, 1], np)
            d1 = great_circle_distance_lon_lat(py[:-1, 0], grid[nhalo,nhalo+1:-nhalo,0], py[:-1,1], grid[nhalo,nhalo+1:-nhalo,1], radius, np)
            d2 = great_circle_distance_lon_lat(py[1:, 0], grid[nhalo,nhalo+1:-nhalo,0], py[1:,1], grid[nhalo,nhalo+1:-nhalo,1], radius, np)
            edge_w = d2/(d1+d2)
        if tile_partitioner.on_tile_right(rank):
            py = lon_lat_midpoint(agrid[-nhalo-1, nhalo:-nhalo, 0], agrid[-nhalo, nhalo:-nhalo, 0], agrid[-nhalo-1, nhalo:-nhalo, 1], agrid[-nhalo, nhalo:-nhalo, 1], np)
            d1 = great_circle_distance_lon_lat(py[:-1, 0], grid[-nhalo,nhalo+1:-nhalo,0], py[:-1,1], grid[-nhalo,nhalo+1:-nhalo,1], radius, np)
            d2 = great_circle_distance_lon_lat(py[1:, 0], grid[-nhalo,nhalo+1:-nhalo,0], py[1:,1], grid[-nhalo,nhalo+1:-nhalo,1], radius, np)
            edge_e = d2/(d1+d2)
        if tile_partitioner.on_tile_bottom(rank):
            px = lon_lat_midpoint(agrid[nhalo:-nhalo, nhalo-1, 0], agrid[nhalo:-nhalo, nhalo, 0], agrid[nhalo:-nhalo, nhalo-1, 1], agrid[nhalo:-nhalo, nhalo, 1], np)
            d1 = great_circle_distance_lon_lat(px[:-1, 0], grid[nhalo+1:-nhalo, nhalo, 0], px[:-1, 1], grid[nhalo+1:-nhalo, nhalo, 1], radius, np)
            d2 = great_circle_distance_lon_lat(px[1:,0], grid[nhalo+1:-nhalo, nhalo, 0], px[1:,1], grid[nhalo+1:-nhalo, nhalo, 1], radius, np)
            edge_s = d2/(d1+d2)
        if tile_partitioner.on_tile_bottom(rank):
            px = lon_lat_midpoint(agrid[nhalo:-nhalo, -nhalo-1, 0], agrid[nhalo:-nhalo, -nhalo, 0], agrid[nhalo:-nhalo, -nhalo-1, 1], agrid[nhalo:-nhalo, -nhalo, 1], np)
            d1 = great_circle_distance_lon_lat(px[:-1, 0], grid[nhalo+1:-nhalo, -nhalo, 0], px[:-1, 1], grid[nhalo+1:-nhalo, -nhalo, 1], radius, np)
            d2 = great_circle_distance_lon_lat(px[1:,0], grid[nhalo+1:-nhalo, -nhalo, 0], px[1:,1], grid[nhalo+1:-nhalo, -nhalo, 1], radius, np)
            edge_n = d2/(d1+d2)

    return edge_w, edge_e, edge_s, edge_n

def efactor_a2c_v(grid, agrid, grid_type, nhalo, tile_partitioner, rank, radius, np):
    '''
    Creates interpolation factors at face edges to interpolate from A to C grids
    '''
    big_number = 1.e8
    npx = agrid.shape[0]
    npy = agrid.shape[1]
    if npx != npy: raise ValueError("npx must equal npy")
    if npx %2 == 0: raise ValueError("npx must be odd")

    im2 = (npx-1)/2
    jm2 = (npy-1)/2

    d2 = d1 = np.zeros(npy+2)

    edge_vect_s = edge_vect_n = np.zeros(npx)+ big_number
    edge_vect_e = edge_vect_w = np.zeros(npy)+ big_number

    if grid_type < 3:
        if tile_partitioner.on_tile_left(rank):
            py = lon_lat_midpoint(agrid[nhalo-1, nhalo-2:-nhalo+2, 0], agrid[nhalo, nhalo-2:-nhalo+2, 0], agrid[nhalo-1, nhalo-2:-nhalo+2, 1], agrid[nhalo, nhalo-2:-nhalo+2, 1], np)
            p2 = lon_lat_midpoint(grid[nhalo, nhalo-2:-nhalo+2, 0], grid[nhalo, nhalo-1:-nhalo+3, 0], grid[nhalo, nhalo-2:-nhalo+2, 1], grid[nhalo, nhalo-1:-nhalo+3, 1], np)
            d1[:jm2+1] = great_circle_distance_lon_lat(py[:jm2+1, 0], p2[:jm2+1, 0], py[:jm2+1,1], p2[:jm2+1,1], radius, np)
            d2[:jm2+1] = great_circle_distance_lon_lat(py[1:jm2+2, 0], p2[:jm2+1, 0], py[1:jm2+2,1], p2[:jm2+1,1], radius, np)
            d1[jm2+1:] = great_circle_distance_lon_lat(py[jm2+1:, 0], p2[jm2+1:, 0], py[jm2+1:,1], p2[jm2+1:,1], radius, np)
            d2[jm2+1:] = great_circle_distance_lon_lat(py[jm2:-1, 0], p2[jm2+1:, 0], py[jm2:-1,1], p2[jm2+1:,1], radius, np)
            edge_vect_w = d1/(d2+d1)
            if tile_partitioner.on_tile_bottom(rank):
                edge_vect_w[nhalo-1] = edge_vect_w[nhalo]
            if tile_partitioner.on_tile_top(rank):
                edge_vect_w[-nhalo+1] = edge_vect_w[-nhalo]
        if tile_partitioner.on_tile_right(rank):
            py = lon_lat_midpoint(agrid[-nhalo-1, nhalo-2:-nhalo+2, 0], agrid[-nhalo, nhalo-2:-nhalo+2, 0], agrid[-nhalo-1, nhalo-2:-nhalo+2, 1], agrid[-nhalo, nhalo-2:-nhalo+2, 1], np)
            p2 = lon_lat_midpoint(grid[-nhalo, nhalo-2:-nhalo+2, 0], grid[-nhalo, nhalo-1:-nhalo+3, 0], grid[-nhalo, nhalo-2:-nhalo+2, 1], grid[-nhalo, nhalo-1:-nhalo+3, 1], np)
            d1[:jm2+1] = great_circle_distance_lon_lat(py[:jm2+1, 0], p2[:jm2+1, 0], py[:jm2+1,1], p2[:jm2+1,1], radius, np)
            d2[:jm2+1] = great_circle_distance_lon_lat(py[1:jm2+2, 0], p2[:jm2+1, 0], py[1:jm2+2,1], p2[:jm2+1,1], radius, np)
            d1[jm2+1:] = great_circle_distance_lon_lat(py[jm2+1:, 0], p2[jm2+1:, 0], py[jm2+1:,1], p2[jm2+1:,1], radius, np)
            d2[jm2+1:] = great_circle_distance_lon_lat(py[jm2:-1, 0], p2[jm2+1:, 0], py[jm2:-1,1], p2[jm2+1:,1], radius, np)
            edge_vect_e = d1/(d2+d1)
            if tile_partitioner.on_tile_bottom(rank):
                edge_vect_e[nhalo-1] = edge_vect_e[nhalo]
            if tile_partitioner.on_tile_top(rank):
                edge_vect_e[-nhalo+1] = edge_vect_e[-nhalo]
        if tile_partitioner.on_tile_bottom(rank):
            px = lon_lat_midpoint(agrid[nhalo-2:-nhalo+2, nhalo-1, 0], agrid[nhalo-2:-nhalo+2, nhalo, 0], agrid[nhalo-2:-nhalo+2, nhalo-1, 1], agrid[nhalo-2:-nhalo+2, nhalo, 1], np)
            p1 = lon_lat_midpoint(grid[nhalo-2:-nhalo+2, nhalo, 0], grid[nhalo-1:-nhalo+3, nhalo, 0], grid[nhalo-2:-nhalo+2, nhalo, 1], grid[nhalo-1:-nhalo+3, nhalo, 1], np)
            d1[:im2+1] = great_circle_distance_lon_lat(px[:im2+1,0], p1[:im2+1,0], px[:im2+1,1], p1[:im2+1,1], radius, np)
            d2[:im2+1] = great_circle_distance_lon_lat(px[1:im2+2,0], p1[:im2+1,0], px[1:im2+2,1], p1[:im2+1,1], radius, np)
            d1[im2+1:] = great_circle_distance_lon_lat(px[im2+1:,0], p1[im2+1:,0], px[im2+1:,1], p1[im2+1:,1], radius, np)
            d2[im2+1:] = great_circle_distance_lon_lat(px[im2:-1,0], p1[im2+1:,0], px[im2-1:-1,1], p1[im2+1:,1], radius, np)
            edge_vect_s = d1/(d2+d1)
            if tile_partitioner.on_tile_left(rank):
                edge_vect_s[nhalo-1] = edge_vect_s[nhalo]
            if tile_partitioner.on_tile_right(rank):
                edge_vect_s[-nhalo+1] = edge_vect_s[-nhalo]
        if tile_partitioner.on_tile_top(rank):
            px = lon_lat_midpoint(agrid[nhalo-2:-nhalo+2, -nhalo-1, 0], agrid[nhalo-2:-nhalo+2, -nhalo, 0], agrid[nhalo-2:-nhalo+2, -nhalo-1, 1], agrid[nhalo-2:-nhalo+2, -nhalo, 1], np)
            p1 = lon_lat_midpoint(grid[nhalo-2:-nhalo+2, -nhalo, 0], grid[nhalo-1:-nhalo+3, -nhalo, 0], grid[nhalo-2:-nhalo+2, -nhalo, 1], grid[nhalo-1:-nhalo+3, -nhalo, 1], np)
            d1[:im2+1] = great_circle_distance_lon_lat(px[:im2+1,0], p1[:im2+1,0], px[:im2+1,1], p1[:im2+1,1], radius, np)
            d2[:im2+1] = great_circle_distance_lon_lat(px[1:im2+2,0], p1[:im2+1,0], px[1:im2+2,1], p1[:im2+1,1], radius, np)
            d1[im2+1:] = great_circle_distance_lon_lat(px[im2+1:,0], p1[im2+1:,0], px[im2+1:,1], p1[im2+1:,1], radius, np)
            d2[im2+1:] = great_circle_distance_lon_lat(px[im2:-1,0], p1[im2+1:,0], px[im2-1:-1,1], p1[im2+1:,1], radius, np)
            edge_vect_n = d1/(d2+d1)
            if tile_partitioner.on_tile_left(rank):
                edge_vect_n[nhalo-1] = edge_vect_n[nhalo]
            if tile_partitioner.on_tile_right(rank):
                edge_vect_n[-nhalo+1] = edge_vect_n[-nhalo]            

    return edge_vect_w, edge_vect_e, edge_vect_s, edge_vect_n
    

def unit_vector_lonlat(grid, np):
    '''
    Calculates the cartesian unit vectors for each point on a lat/lon grid
    '''

    sin_lon = np.sin(grid[:,:,0])
    cos_lon = np.cos(grid[:,:,0])
    sin_lat = np.sin(grid[:,:,1])
    cos_lat = np.cos(grid[:,:,1])

    unit_lon = np.array([-sin_lon, cos_lon, np.zeros(grid[:,:,0].shape)])
    unit_lat = np.array([-sin_lat*cos_lon, -sin_lat*sin_lon, cos_lat])

    return unit_lon, unit_lat

def _fill_ghost(field, value: float, nhalo: int, corner: str):
    """
    Fills a tile halo corner (ghost cells) of a field with a set value along the first 2 axes
    Args:
        field: the field to fill in, assumed to have x and y as the first 2 dimensions
        value: the value to fill
        nhalo: the number of halo points in the field
        corner: which corner to fill
    """
    if (corner == "sw") or (corner == "southwest"):
        field[:nhalo, :nhalo] = value
    elif (corner == "nw") or (corner == "northwest"):
        field[:nhalo, -nhalo:] = value
    elif (corner == "se") or (corner == "southeast"):
        field[-nhalo:, :nhalo] = value
    elif (corner == "ne") or (corner == "northeast"):
        field[-nhalo:, -nhalo:] = value
    else:
        raise ValueError("fill ghost requires a corner to be one of: sw, se, nw, ne")
    