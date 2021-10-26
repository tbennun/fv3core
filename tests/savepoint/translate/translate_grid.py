from typing import Any, Dict

import pytest

import fv3core._config as spec
import fv3core.utils.global_config as global_config
import fv3gfs.util as fv3util
from fv3core.grid import MetricTerms, global_mirror_grid, gnomonic_grid, set_eta
from fv3core.grid.geometry import calculate_divg_del6
from fv3core.testing.parallel_translate import ParallelTranslateGrid
from fv3core.utils.global_constants import CARTESIAN_DIM, LON_OR_LAT_DIM, TILE_DIM


class TranslateGnomonicGrids(ParallelTranslateGrid):

    max_error = 2e-14

    inputs = {
        "lon": {
            "name": "longitude_on_cell_corners",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "radians",
            "n_halo": 0,
        },
        "lat": {
            "name": "latitude_on_cell_corners",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "radians",
            "n_halo": 0,
        },
    }
    outputs = {
        "lon": {
            "name": "longitude_on_cell_corners",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "radians",
            "n_halo": 0,
        },
        "lat": {
            "name": "latitude_on_cell_corners",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "radians",
            "n_halo": 0,
        },
    }

    def compute_sequential(self, inputs_list, communicator_list):
        outputs = []
        for inputs in inputs_list:
            outputs.append(self.compute(inputs))
        return outputs

    def compute(self, inputs):
        state = self.state_from_inputs(inputs)
        gnomonic_grid(
            self.grid.grid_type,
            state["longitude_on_cell_corners"].view[:],
            state["latitude_on_cell_corners"].view[:],
            state["longitude_on_cell_corners"].np,
        )
        outputs = self.outputs_from_state(state)
        return outputs


class TranslateMirrorGrid(ParallelTranslateGrid):

    inputs = {
        "master_grid_global": {
            "name": "grid_global",
            "dims": [
                fv3util.X_INTERFACE_DIM,
                fv3util.Y_INTERFACE_DIM,
                LON_OR_LAT_DIM,
                TILE_DIM,
            ],
            "units": "radians",
            "n_halo": 3,
        },
        "master_ng": {"name": "n_ghost", "dims": []},
        "master_npx": {"name": "npx", "dims": []},
        "master_npy": {"name": "npy", "dims": []},
    }
    outputs = {
        "master_grid_global": {
            "name": "grid_global",
            "dims": [
                fv3util.X_INTERFACE_DIM,
                fv3util.Y_INTERFACE_DIM,
                LON_OR_LAT_DIM,
                TILE_DIM,
            ],
            "units": "radians",
            "n_halo": 3,
        },
    }

    def compute_parallel(self, inputs, communicator):
        pytest.skip(f"{self.__class__} not running in parallel")

    def compute_sequential(self, inputs_list, communicator_list):
        outputs = []
        outputs.append(self.compute(inputs_list[0]))
        for inputs in inputs_list[1:]:
            outputs.append(inputs)
        return outputs

    def compute(self, inputs):
        state = self.state_from_inputs(inputs)
        global_mirror_grid(
            state["grid_global"].data,
            state["n_ghost"],
            state["npx"],
            state["npy"],
            state["grid_global"].np,
        )
        outputs = self.outputs_from_state(state)
        return outputs


class TranslateGridAreas(ParallelTranslateGrid):

    inputs = {
        "grid": {
            "name": "grid",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM, LON_OR_LAT_DIM],
            "units": "radians",
        },
        "agrid": {
            "name": "agrid",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, LON_OR_LAT_DIM],
            "units": "radians",
        },
        "area": {
            "name": "area",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "m^2",
        },
        "area_c": {
            "name": "area_cgrid",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "m^2",
        },
        "dxa": {
            "name": "dx_agrid",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "m",
        },
        "dya": {
            "name": "dy_agrid",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "m",
        },
        "dxc": {
            "name": "dx_cgrid",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM],
            "units": "m",
        },
        "dyc": {
            "name": "dy_cgrid",
            "dims": [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "m",
        },
    }
    outputs = {
        "area": {
            "name": "area",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "m^2",
        },
        "area_c": {
            "name": "area_cgrid",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "m^2",
        },
        "dxa": {
            "name": "dx_agrid",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "m",
        },
        "dya": {
            "name": "dy_agrid",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "m",
        },
        "dxc": {
            "name": "dx_cgrid",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM],
            "units": "m",
        },
        "dyc": {
            "name": "dy_cgrid",
            "dims": [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "m",
        },
    }

    def compute_parallel(self, inputs, communicator):
        namelist = spec.namelist
        grid_generator = MetricTerms.from_tile_sizing(
            npx=namelist.npx,
            npy=namelist.npy,
            npz=1,
            communicator=communicator,
            backend=global_config.get_backend(),
        )

        in_state = self.state_from_inputs(inputs)
        grid_generator._grid.data[:] = in_state["grid"].data[:]
        grid_generator._agrid.data[:] = in_state["agrid"].data[:]
        state = {}
        for metric_term, metadata in self.outputs.items():
            state[metadata["name"]] = getattr(grid_generator, metric_term)
        return self.outputs_from_state(state)


class TranslateGridGrid(ParallelTranslateGrid):

    max_error = 1e-14
    inputs: Dict[str, Any] = {
        "grid_global": {
            "name": "grid",
            "dims": [
                fv3util.X_INTERFACE_DIM,
                fv3util.Y_INTERFACE_DIM,
                LON_OR_LAT_DIM,
                TILE_DIM,
            ],
            "units": "radians",
        }
    }
    outputs = {
        "grid": {
            "name": "grid",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM, LON_OR_LAT_DIM],
            "units": "radians",
        },
    }

    def __init__(self, grids):
        super().__init__(grids)
        self.max_error = 1.0e-13

    def compute_parallel(self, inputs, communicator):
        namelist = spec.namelist
        grid_generator = MetricTerms.from_tile_sizing(
            npx=namelist.npx,
            npy=namelist.npy,
            npz=1,
            communicator=communicator,
            backend=global_config.get_backend(),
        )

        state = {}
        for metric_term, metadata in self.outputs.items():
            state[metadata["name"]] = getattr(grid_generator, metric_term)
        return self.outputs_from_state(state)


class TranslateDxDy(ParallelTranslateGrid):

    inputs = {
        "grid": {
            "name": "grid",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM, LON_OR_LAT_DIM],
            "units": "radians",
        },
    }
    outputs = {
        "dx": {
            "name": "dx",
            "dims": [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "m",
        },
        "dy": {
            "name": "dy",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM],
            "units": "m",
        },
    }

    def compute_parallel(self, inputs, communicator):
        namelist = spec.namelist
        grid_generator = MetricTerms.from_tile_sizing(
            npx=namelist.npx,
            npy=namelist.npy,
            npz=1,
            communicator=communicator,
            backend=global_config.get_backend(),
        )

        in_state = self.state_from_inputs(inputs)
        grid_generator._grid.data[:] = in_state["grid"].data[:]
        state = {}
        for metric_term, metadata in self.outputs.items():
            state[metadata["name"]] = getattr(grid_generator, metric_term)
        return self.outputs_from_state(state)


class TranslateAGrid(ParallelTranslateGrid):

    inputs = {
        "agrid": {
            "name": "agrid",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, LON_OR_LAT_DIM],
            "units": "radians",
        },
        "grid": {
            "name": "grid",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM, LON_OR_LAT_DIM],
            "units": "radians",
        },
    }
    outputs = {
        "agrid": {
            "name": "agrid",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, LON_OR_LAT_DIM],
            "units": "radians",
        },
        "grid": {
            "name": "grid",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM, LON_OR_LAT_DIM],
            "units": "radians",
        },
    }

    def compute_parallel(self, inputs, communicator):
        namelist = spec.namelist
        grid_generator = MetricTerms.from_tile_sizing(
            npx=namelist.npx,
            npy=namelist.npy,
            npz=1,
            communicator=communicator,
            backend=global_config.get_backend(),
        )

        in_state = self.state_from_inputs(inputs)
        grid_generator._grid.data[:] = in_state["grid"].data[:]
        grid_generator._init_agrid()
        state = {}
        for metric_term, metadata in self.outputs.items():
            state[metadata["name"]] = getattr(grid_generator, metric_term)
        return self.outputs_from_state(state)


class TranslateInitGrid(ParallelTranslateGrid):
    inputs = {
        "ndims": {"name": "ndims", "dims": []},
        "nregions": {
            "name": "nregions",
            "dims": [],
        },
        "grid_name": {
            "name": "grid_name",
            "dims": [],
        },
        "sw_corner": {
            "name": "sw_corner",
            "dims": [],
        },
        "se_corner": {
            "name": "se_corner",
            "dims": [],
        },
        "nw_corner": {
            "name": "nw_corner",
            "dims": [],
        },
        "ne_corner": {
            "name": "ne_corner",
            "dims": [],
        },
    }
    outputs: Dict[str, Any] = {
        "gridvar": {
            "name": "grid",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM, LON_OR_LAT_DIM],
            "units": "radians",
        },
        "agrid": {
            "name": "agrid",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, LON_OR_LAT_DIM],
            "units": "radians",
        },
        "area": {
            "name": "area",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "m^2",
        },
        "area_c": {
            "name": "area_cgrid",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "m^2",
        },
        "dx": {
            "name": "dx",
            "dims": [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "m",
        },
        "dy": {
            "name": "dy",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM],
            "units": "m",
        },
        "dxc": {
            "name": "dx_cgrid",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM],
            "units": "m",
        },
        "dyc": {
            "name": "dy_cgrid",
            "dims": [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "m",
        },
        "dxa": {
            "name": "dx_agrid",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "m",
        },
        "dya": {
            "name": "dy_agrid",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "m",
        },
    }

    def __init__(self, grids):
        super().__init__(grids)
        self.ignore_near_zero_errors = {}
        self.ignore_near_zero_errors["grid"] = True

    def compute_parallel(self, inputs, communicator):
        namelist = spec.namelist
        grid_generator = MetricTerms.from_tile_sizing(
            npx=namelist.npx,
            npy=namelist.npy,
            npz=1,
            communicator=communicator,
            backend=global_config.get_backend(),
        )
        state = {}
        for metric_term, metadata in self.outputs.items():
            state[metadata["name"]] = getattr(grid_generator, metric_term)
        return self.outputs_from_state(state)


class TranslateSetEta(ParallelTranslateGrid):
    inputs: Dict[str, Any] = {
        "npz": {
            "name": "npz",
            "dims": [],
            "units": "",
        },
        "ks": {
            "name": "ks",
            "dims": [],
            "units": "",
        },
        "ptop": {
            "name": "ptop",
            "dims": [],
            "units": "mb",
        },
        "ak": {
            "name": "ak",
            "dims": [fv3util.Z_INTERFACE_DIM],
            "units": "mb",
        },
        "bk": {
            "name": "bk",
            "dims": [fv3util.Z_INTERFACE_DIM],
            "units": "",
        },
    }
    outputs: Dict[str, Any] = {
        "ks": {
            "name": "ks",
            "dims": [],
            "units": "",
        },
        "ptop": {
            "name": "ptop",
            "dims": [],
            "units": "mb",
        },
        "ak": {
            "name": "ak",
            "dims": [fv3util.Z_INTERFACE_DIM],
            "units": "mb",
        },
        "bk": {
            "name": "bk",
            "dims": [fv3util.Z_INTERFACE_DIM],
            "units": "",
        },
    }

    def compute_sequential(self, inputs_list, communicator_list):
        state_list = []
        for inputs in inputs_list:
            state_list.append(self._compute_local(inputs))
        return self.outputs_list_from_state_list(state_list)

    def _compute_local(self, inputs):
        state = self.state_from_inputs(inputs)
        state["ks"], state["ptop"], state["ak"].data[:], state["bk"].data[:] = set_eta(
            state["npz"]
        )
        return state


class TranslateUtilVectors(ParallelTranslateGrid):
    def __init__(self, grids):
        super().__init__(grids)
        self._base.in_vars["data_vars"] = {
            "ec1": {
                "kend": 2,
                "kaxis": 0,
            },
            "ec2": {
                "kend": 2,
                "kaxis": 0,
            },
            "ew1": {
                "kend": 2,
                "kaxis": 0,
            },
            "ew2": {
                "kend": 2,
                "kaxis": 0,
            },
            "es1": {
                "kend": 2,
                "kaxis": 0,
            },
            "es2": {
                "kend": 2,
                "kaxis": 0,
            },
        }

    inputs: Dict[str, Any] = {
        "grid": {
            "name": "grid",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM, LON_OR_LAT_DIM],
            "units": "radians",
        },
        "agrid": {
            "name": "agrid",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, LON_OR_LAT_DIM],
            "units": "radians",
        },
        "ec1": {
            "name": "ec1",
            "dims": [CARTESIAN_DIM, fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "ec2": {
            "name": "ec2",
            "dims": [CARTESIAN_DIM, fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "ew1": {
            "name": "ew1",
            "dims": [CARTESIAN_DIM, fv3util.X_INTERFACE_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "ew2": {
            "name": "ew2",
            "dims": [CARTESIAN_DIM, fv3util.X_INTERFACE_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "es1": {
            "name": "es1",
            "dims": [CARTESIAN_DIM, fv3util.X_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "",
        },
        "es2": {
            "name": "es2",
            "dims": [CARTESIAN_DIM, fv3util.X_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "",
        },
    }
    outputs: Dict[str, Any] = {
        "ec1": {
            "name": "ec1",
            "dims": [CARTESIAN_DIM, fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "ec2": {
            "name": "ec2",
            "dims": [CARTESIAN_DIM, fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "ew1": {
            "name": "ew1",
            "dims": [CARTESIAN_DIM, fv3util.X_INTERFACE_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "ew2": {
            "name": "ew2",
            "dims": [CARTESIAN_DIM, fv3util.X_INTERFACE_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "es1": {
            "name": "es1",
            "dims": [CARTESIAN_DIM, fv3util.X_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "",
        },
        "es2": {
            "name": "es2",
            "dims": [CARTESIAN_DIM, fv3util.X_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "",
        },
    }

    def compute_parallel(self, inputs, communicator):
        namelist = spec.namelist
        grid_generator = MetricTerms.from_tile_sizing(
            npx=namelist.npx,
            npy=namelist.npy,
            npz=1,
            communicator=communicator,
            backend=global_config.get_backend(),
        )

        in_state = self.state_from_inputs(inputs)
        grid_generator._grid.data[:] = in_state["grid"].data[:]
        grid_generator._agrid.data[:] = in_state["agrid"].data[:]
        state = {}
        for metric_term, metadata in self.outputs.items():
            state[metadata["name"]] = getattr(grid_generator, metric_term)
        return self.outputs_from_state(state)


class TranslateTrigTerms(ParallelTranslateGrid):
    def __init__(self, grids):
        super().__init__(grids)
        self._base.in_vars["data_vars"] = {
            "ec1": {
                "kend": 2,
                "kaxis": 0,
            },
            "ec2": {
                "kend": 2,
                "kaxis": 0,
            },
            "ee1": {
                "kend": 2,
                "kaxis": 0,
            },
            "ee2": {
                "kend": 2,
                "kaxis": 0,
            },
        }

    inputs: Dict[str, Any] = {
        "grid": {
            "name": "grid",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM, LON_OR_LAT_DIM],
            "units": "",
        },
        "agrid": {
            "name": "agrid",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, LON_OR_LAT_DIM],
            "units": "radians",
        },
        "cos_sg1": {
            "name": "cos_sg1",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "sin_sg1": {
            "name": "sin_sg1",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "cos_sg2": {
            "name": "cos_sg2",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "sin_sg2": {
            "name": "sin_sg2",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "cos_sg3": {
            "name": "cos_sg3",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "sin_sg3": {
            "name": "sin_sg3",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "cos_sg4": {
            "name": "cos_sg4",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "sin_sg4": {
            "name": "sin_sg4",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "cos_sg5": {
            "name": "cos_sg5",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "sin_sg5": {
            "name": "sin_sg5",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "cos_sg6": {
            "name": "cos_sg6",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "sin_sg6": {
            "name": "sin_sg6",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "cos_sg7": {
            "name": "cos_sg7",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "sin_sg7": {
            "name": "sin_sg7",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "cos_sg8": {
            "name": "cos_sg8",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "sin_sg8": {
            "name": "sin_sg8",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "cos_sg9": {
            "name": "cos_sg9",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "sin_sg9": {
            "name": "sin_sg9",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "ec1": {
            "name": "ec1",
            "dims": [CARTESIAN_DIM, fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "ec2": {
            "name": "ec2",
            "dims": [CARTESIAN_DIM, fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "ee1": {
            "name": "ee1",
            "dims": [CARTESIAN_DIM, fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "",
        },
        "ee2": {
            "name": "ee2",
            "dims": [CARTESIAN_DIM, fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "",
        },
        "cosa_u": {
            "name": "cosa_u",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "cosa_v": {
            "name": "cosa_v",
            "dims": [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "",
        },
        "cosa_s": {
            "name": "cosa_s",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "sina_u": {
            "name": "sina_u",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "sina_v": {
            "name": "sina_v",
            "dims": [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "",
        },
        "rsin_u": {
            "name": "rsin_u",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "rsin_v": {
            "name": "rsin_v",
            "dims": [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "",
        },
        "rsina": {
            "name": "rsina",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "",
            "n_halo": 0,
        },
        "rsin2": {"name": "rsin2", "dims": [fv3util.X_DIM, fv3util.Y_DIM], "units": ""},
        "cosa": {
            "name": "cosa",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "",
        },
        "sina": {
            "name": "sina",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "",
        },
    }
    outputs: Dict[str, Any] = {
        "cos_sg1": {
            "name": "cos_sg1",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "sin_sg1": {
            "name": "sin_sg1",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "cos_sg2": {
            "name": "cos_sg2",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "sin_sg2": {
            "name": "sin_sg2",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "cos_sg3": {
            "name": "cos_sg3",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "sin_sg3": {
            "name": "sin_sg3",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "cos_sg4": {
            "name": "cos_sg4",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "sin_sg4": {
            "name": "sin_sg4",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "cos_sg5": {
            "name": "cos_sg5",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "sin_sg5": {
            "name": "sin_sg5",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "cos_sg6": {
            "name": "cos_sg6",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "sin_sg6": {
            "name": "sin_sg6",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "cos_sg7": {
            "name": "cos_sg7",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "sin_sg7": {
            "name": "sin_sg7",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "cos_sg8": {
            "name": "cos_sg8",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "sin_sg8": {
            "name": "sin_sg8",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "cos_sg9": {
            "name": "cos_sg9",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "sin_sg9": {
            "name": "sin_sg9",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "ee1": {
            "name": "ee1",
            "dims": [CARTESIAN_DIM, fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "",
        },
        "ee2": {
            "name": "ee2",
            "dims": [CARTESIAN_DIM, fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "",
        },
        "cosa_u": {
            "name": "cosa_u",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "cosa_v": {
            "name": "cosa_v",
            "dims": [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "",
        },
        "cosa_s": {
            "name": "cosa_s",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "sina_u": {
            "name": "sina_u",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "sina_v": {
            "name": "sina_v",
            "dims": [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "",
        },
        "rsin_u": {
            "name": "rsin_u",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "rsin_v": {
            "name": "rsin_v",
            "dims": [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "",
        },
        "rsina": {
            "name": "rsina",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "",
            "n_halo": 0,
        },
        "rsin2": {"name": "rsin2", "dims": [fv3util.X_DIM, fv3util.Y_DIM], "units": ""},
        "cosa": {
            "name": "cosa",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "",
        },
        "sina": {
            "name": "sina",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "",
        },
    }

    def compute_parallel(self, inputs, communicator):
        namelist = spec.namelist
        grid_generator = MetricTerms.from_tile_sizing(
            npx=namelist.npx,
            npy=namelist.npy,
            npz=1,
            communicator=communicator,
            backend=global_config.get_backend(),
        )

        in_state = self.state_from_inputs(inputs)
        grid_generator._grid.data[:] = in_state["grid"].data[:]
        grid_generator._agrid.data[:] = in_state["agrid"].data[:]
        grid_generator._ec1 = in_state["ec1"]
        grid_generator._ec2 = in_state["ec2"]
        state = {}
        for metric_term, metadata in self.outputs.items():
            state[metadata["name"]] = getattr(grid_generator, metric_term)
        return self.outputs_from_state(state)


class TranslateAAMCorrection(ParallelTranslateGrid):
    inputs: Dict[str, Any] = {
        "grid": {
            "name": "grid",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM, LON_OR_LAT_DIM],
            "units": "radians",
        },
        "l2c_v": {
            "name": "l2c_v",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM],
            "units": "",
            "n_halo": 0,
        },
        "l2c_u": {
            "name": "l2c_u",
            "dims": [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "",
            "n_halo": 0,
        },
    }
    outputs: Dict[str, Any] = {
        "l2c_v": {
            "name": "l2c_v",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM],
            "units": "",
            "n_halo": 0,
        },
        "l2c_u": {
            "name": "l2c_u",
            "dims": [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "",
            "n_halo": 0,
        },
    }

    def compute_parallel(self, inputs, communicator):
        namelist = spec.namelist
        grid_generator = MetricTerms.from_tile_sizing(
            npx=namelist.npx,
            npy=namelist.npy,
            npz=1,
            communicator=communicator,
            backend=global_config.get_backend(),
        )

        in_state = self.state_from_inputs(inputs)
        grid_generator._grid.data[:] = in_state["grid"].data[:]
        state = {}
        for metric_term, metadata in self.outputs.items():
            state[metadata["name"]] = getattr(grid_generator, metric_term)
        return self.outputs_from_state(state)


class TranslateTrigSubset(ParallelTranslateGrid):
    def __init__(self, grids):
        super().__init__(grids)
        self._base.in_vars["data_vars"] = {
            "ee1": {
                "kend": 2,
                "kaxis": 0,
            },
            "ee2": {
                "kend": 2,
                "kaxis": 0,
            },
        }

    inputs: Dict[str, Any] = {
        "grid": {
            "name": "grid",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM, LON_OR_LAT_DIM],
            "units": "radians",
        },
        "cos_sg1": {
            "name": "cos_sg1",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "sin_sg1": {
            "name": "sin_sg1",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "cos_sg2": {
            "name": "cos_sg2",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "sin_sg2": {
            "name": "sin_sg2",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "cos_sg3": {
            "name": "cos_sg3",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "sin_sg3": {
            "name": "sin_sg3",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "cos_sg4": {
            "name": "cos_sg4",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "sin_sg4": {
            "name": "sin_sg4",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "cos_sg5": {
            "name": "cos_sg5",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "sin_sg5": {
            "name": "sin_sg5",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "cos_sg6": {
            "name": "cos_sg6",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "sin_sg6": {
            "name": "sin_sg6",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "cos_sg7": {
            "name": "cos_sg7",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "sin_sg7": {
            "name": "sin_sg7",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "cos_sg8": {
            "name": "cos_sg8",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "sin_sg8": {
            "name": "sin_sg8",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "cos_sg9": {
            "name": "cos_sg9",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "sin_sg9": {
            "name": "sin_sg9",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "ee1": {
            "name": "ee1",
            "dims": [CARTESIAN_DIM, fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "",
        },
        "ee2": {
            "name": "ee2",
            "dims": [CARTESIAN_DIM, fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "",
        },
        "cosa_u": {
            "name": "cosa_u",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "cosa_v": {
            "name": "cosa_v",
            "dims": [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "",
        },
        "cosa_s": {
            "name": "cosa_s",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "sina_u": {
            "name": "sina_u",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "sina_v": {
            "name": "sina_v",
            "dims": [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "",
        },
        "rsin_u": {
            "name": "rsin_u",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "rsin_v": {
            "name": "rsin_v",
            "dims": [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "",
        },
        "rsina": {
            "name": "rsina",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "",
            "n_halo": 0,
        },
        "rsin2": {"name": "rsin2", "dims": [fv3util.X_DIM, fv3util.Y_DIM], "units": ""},
        "cosa": {
            "name": "cosa",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "",
        },
        "sina": {
            "name": "sina",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "",
        },
    }
    outputs: Dict[str, Any] = {
        "ee1": {
            "name": "ee1",
            "dims": [CARTESIAN_DIM, fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "",
        },
        "ee2": {
            "name": "ee2",
            "dims": [CARTESIAN_DIM, fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "",
        },
        "cosa_u": {
            "name": "cosa_u",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "cosa_v": {
            "name": "cosa_v",
            "dims": [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "",
        },
        "cosa_s": {
            "name": "cosa_s",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "sina_u": {
            "name": "sina_u",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "sina_v": {
            "name": "sina_v",
            "dims": [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "",
        },
        "rsin_u": {
            "name": "rsin_u",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "rsin_v": {
            "name": "rsin_v",
            "dims": [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "",
        },
        "rsina": {
            "name": "rsina",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "",
            "n_halo": 0,
        },
        "rsin2": {"name": "rsin2", "dims": [fv3util.X_DIM, fv3util.Y_DIM], "units": ""},
        "cosa": {
            "name": "cosa",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "",
        },
        "sina": {
            "name": "sina",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "",
        },
    }

    def compute_parallel(self, inputs, communicator):
        namelist = spec.namelist
        grid_generator = MetricTerms.from_tile_sizing(
            npx=namelist.npx,
            npy=namelist.npy,
            npz=1,
            communicator=communicator,
            backend=global_config.get_backend(),
        )

        in_state = self.state_from_inputs(inputs)
        grid_generator._grid.data[:] = in_state["grid"].data[:]
        grid_generator._cos_sg1 = in_state["cos_sg1"]
        grid_generator._cos_sg2 = in_state["cos_sg2"]
        grid_generator._cos_sg3 = in_state["cos_sg3"]
        grid_generator._cos_sg4 = in_state["cos_sg4"]
        grid_generator._cos_sg5 = in_state["cos_sg5"]
        grid_generator._cos_sg6 = in_state["cos_sg6"]
        grid_generator._cos_sg7 = in_state["cos_sg7"]
        grid_generator._cos_sg8 = in_state["cos_sg8"]
        grid_generator._cos_sg9 = in_state["cos_sg9"]
        grid_generator._sin_sg1 = in_state["sin_sg1"]
        grid_generator._sin_sg2 = in_state["sin_sg2"]
        grid_generator._sin_sg3 = in_state["sin_sg3"]
        grid_generator._sin_sg4 = in_state["sin_sg4"]
        grid_generator._sin_sg5 = in_state["sin_sg5"]
        grid_generator._sin_sg6 = in_state["sin_sg6"]
        grid_generator._sin_sg7 = in_state["sin_sg7"]
        grid_generator._sin_sg8 = in_state["sin_sg8"]
        grid_generator._sin_sg9 = in_state["sin_sg9"]
        state = {}
        grid_generator._calculate_derived_trig_terms_for_testing()
        for metric_term, metadata in self.outputs.items():
            state[metadata["name"]] = getattr(grid_generator, metric_term)
        return self.outputs_from_state(state)


class TranslateDivgDel6(ParallelTranslateGrid):
    inputs: Dict[str, Any] = {
        "sin_sg1": {
            "name": "sin_sg1",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "sin_sg2": {
            "name": "sin_sg2",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "sin_sg3": {
            "name": "sin_sg3",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "sin_sg4": {
            "name": "sin_sg4",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "sina_u": {
            "name": "sina_u",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "sina_v": {
            "name": "sina_v",
            "dims": [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "",
        },
        "dx": {
            "name": "dx",
            "dims": [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "m",
        },
        "dy": {
            "name": "dy",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM],
            "units": "m",
        },
        "dxc": {
            "name": "dx_cgrid",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM],
            "units": "m",
        },
        "dyc": {
            "name": "dy_cgrid",
            "dims": [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "",
        },
        "divg_u": {
            "name": "divg_u",
            "dims": [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "",
        },
        "divg_v": {
            "name": "divg_v",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM],
            "units": "m",
        },
        "del6_u": {
            "name": "del6_u",
            "dims": [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "",
        },
        "del6_v": {
            "name": "del6_v",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM],
            "units": "",
        },
    }
    outputs: Dict[str, Any] = {
        "divg_u": {
            "name": "divg_u",
            "dims": [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "",
        },
        "divg_v": {
            "name": "divg_v",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "del6_u": {
            "name": "del6_u",
            "dims": [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "",
        },
        "del6_v": {
            "name": "del6_v",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM],
            "units": "",
        },
    }

    def compute_sequential(self, inputs_list, communicator_list):
        state_list = []
        for inputs, communicator in zip(inputs_list, communicator_list):
            state_list.append(self._compute_local(inputs, communicator))
        return self.outputs_list_from_state_list(state_list)

    def _compute_local(self, inputs, communicator):
        state = self.state_from_inputs(inputs)
        sin_sg = []
        for i in range(1, 5):
            sin_sg.append(state[f"sin_sg{i}"].data[:-1, :-1])
        sin_sg = state["sin_sg1"].np.array(sin_sg).transpose(1, 2, 0)
        (
            state["divg_u"].data[:-1, :],
            state["divg_v"].data[:, :-1],
            state["del6_u"].data[:-1, :],
            state["del6_v"].data[:, :-1],
        ) = calculate_divg_del6(
            sin_sg,
            state["sina_u"].data[:, :-1],
            state["sina_v"].data[:-1, :],
            state["dx"].data[:-1, :],
            state["dy"].data[:, :-1],
            state["dx_cgrid"].data[:, :-1],
            state["dy_cgrid"].data[:-1, :],
            self.grid.halo,
            communicator.tile.partitioner,
            communicator.rank,
        )
        return state

    def compute_parallel(self, inputs, communicator):
        namelist = spec.namelist
        grid_generator = MetricTerms.from_tile_sizing(
            npx=namelist.npx,
            npy=namelist.npy,
            npz=1,
            communicator=communicator,
            backend=global_config.get_backend(),
        )

        in_state = self.state_from_inputs(inputs)
        grid_generator._sin_sg1 = in_state["sin_sg1"]
        grid_generator._sin_sg2 = in_state["sin_sg2"]
        grid_generator._sin_sg3 = in_state["sin_sg3"]
        grid_generator._sin_sg4 = in_state["sin_sg4"]
        grid_generator._sina_u = in_state["sina_u"]
        grid_generator._sina_v = in_state["sina_v"]
        grid_generator._dx = in_state["dx"]
        grid_generator._dy = in_state["dy"]
        grid_generator._dx_cgrid = in_state["dx_cgrid"]
        grid_generator._dy_cgrid = in_state["dy_cgrid"]
        state = {}
        for metric_term, metadata in self.outputs.items():
            state[metadata["name"]] = getattr(grid_generator, metric_term)
        return self.outputs_from_state(state)


class TranslateInitCubedtoLatLon(ParallelTranslateGrid):
    def __init__(self, grids):
        super().__init__(grids)
        self._base.in_vars["data_vars"] = {
            "ec1": {
                "kend": 2,
                "kaxis": 0,
            },
            "ec2": {
                "kend": 2,
                "kaxis": 0,
            },
        }

    inputs: Dict[str, Any] = {
        "agrid": {
            "name": "agrid",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, LON_OR_LAT_DIM],
            "units": "radians",
        },
        "ec1": {
            "name": "ec1",
            "dims": [CARTESIAN_DIM, fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "ec2": {
            "name": "ec2",
            "dims": [CARTESIAN_DIM, fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "sin_sg5": {
            "name": "sin_sg5",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
    }
    outputs: Dict[str, Any] = {
        "vlon": {
            "name": "vlon",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, CARTESIAN_DIM],
            "units": "",
            "n_halo": 2,
        },
        "vlat": {
            "name": "vlat",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, CARTESIAN_DIM],
            "units": "",
            "n_halo": 2,
        },
        "z11": {
            "name": "z11",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
            "n_halo": 1,
        },
        "z12": {
            "name": "z12",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
            "n_halo": 1,
        },
        "z21": {
            "name": "z21",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
            "n_halo": 1,
        },
        "z22": {
            "name": "z22",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
            "n_halo": 1,
        },
        "a11": {
            "name": "a11",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
            "n_halo": 1,
        },
        "a12": {
            "name": "a12",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
            "n_halo": 1,
        },
        "a21": {
            "name": "a21",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
            "n_halo": 1,
        },
        "a22": {
            "name": "a22",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
            "n_halo": 1,
        },
    }

    def compute_parallel(self, inputs, communicator):
        namelist = spec.namelist
        grid_generator = MetricTerms.from_tile_sizing(
            npx=namelist.npx,
            npy=namelist.npy,
            npz=1,
            communicator=communicator,
            backend=global_config.get_backend(),
        )

        in_state = self.state_from_inputs(inputs)
        grid_generator._sin_sg5 = in_state["sin_sg5"]
        grid_generator._agrid.data[:] = in_state["agrid"].data[:]
        grid_generator._ec1 = in_state["ec1"]
        grid_generator._ec2 = in_state["ec2"]
        state = {}
        for metric_term, metadata in self.outputs.items():
            state[metadata["name"]] = getattr(grid_generator, metric_term)
        return self.outputs_from_state(state)


class TranslateEdgeFactors(ParallelTranslateGrid):
    inputs: Dict[str, Any] = {
        "grid": {
            "name": "grid",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM, LON_OR_LAT_DIM],
            "units": "radians",
        },
        "agrid": {
            "name": "agrid",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, LON_OR_LAT_DIM],
            "units": "radians",
        },
        "edge_s": {
            "name": "edge_s",
            "dims": [fv3util.X_INTERFACE_DIM],
            "units": "",
            "n_halo": 0,
        },
        "edge_n": {
            "name": "edge_n",
            "dims": [fv3util.X_INTERFACE_DIM],
            "units": "",
            "n_halo": 0,
        },
        "edge_e": {
            "name": "edge_e",
            "dims": [fv3util.Y_INTERFACE_DIM],
            "units": "",
            "n_halo": 0,
        },
        "edge_w": {
            "name": "edge_w",
            "dims": [fv3util.Y_INTERFACE_DIM],
            "units": "",
            "n_halo": 0,
        },
        "edge_vect_s": {
            "name": "edge_vect_s",
            "dims": [fv3util.X_DIM],
            "units": "",
        },
        "edge_vect_n": {
            "name": "edge_vect_n",
            "dims": [fv3util.X_DIM],
            "units": "",
        },
        "edge_vect_e": {
            "name": "edge_vect_e",
            "dims": [fv3util.Y_DIM],
            "units": "",
        },
        "edge_vect_w": {
            "name": "edge_vect_w",
            "dims": [fv3util.Y_DIM],
            "units": "",
        },
    }
    outputs: Dict[str, Any] = {
        "edge_s": {
            "name": "edge_s",
            "dims": [fv3util.X_INTERFACE_DIM],
            "units": "",
            "n_halo": 0,
        },
        "edge_n": {
            "name": "edge_n",
            "dims": [fv3util.X_INTERFACE_DIM],
            "units": "",
            "n_halo": 0,
        },
        "edge_e": {
            "name": "edge_e",
            "dims": [fv3util.Y_INTERFACE_DIM],
            "units": "",
            "n_halo": 0,
        },
        "edge_w": {
            "name": "edge_w",
            "dims": [fv3util.Y_INTERFACE_DIM],
            "units": "",
            "n_halo": 0,
        },
        "edge_vect_s": {
            "name": "edge_vect_s",
            "dims": [fv3util.X_DIM],
            "units": "",
        },
        "edge_vect_n": {
            "name": "edge_vect_n",
            "dims": [fv3util.X_DIM],
            "units": "",
        },
        "edge_vect_e": {
            "name": "edge_vect_e",
            "dims": [fv3util.Y_DIM],
            "units": "",
        },
        "edge_vect_w": {
            "name": "edge_vect_w",
            "dims": [fv3util.Y_DIM],
            "units": "",
        },
    }

    def compute_parallel(self, inputs, communicator):
        namelist = spec.namelist
        grid_generator = MetricTerms.from_tile_sizing(
            npx=namelist.npx,
            npy=namelist.npy,
            npz=1,
            communicator=communicator,
            backend=global_config.get_backend(),
        )

        in_state = self.state_from_inputs(inputs)
        grid_generator._grid.data[:] = in_state["grid"].data[:]
        grid_generator._agrid.data[:] = in_state["agrid"].data[:]
        state = {}
        for metric_term, metadata in self.outputs.items():
            state[metadata["name"]] = getattr(grid_generator, metric_term)
        return self.outputs_from_state(state)


class TranslateInitGridUtils(ParallelTranslateGrid):
    def __init__(self, grids):
        super().__init__(grids)
        self._base.in_vars["data_vars"] = {
            "ec1": {
                "kend": 2,
                "kaxis": 0,
            },
            "ec2": {
                "kend": 2,
                "kaxis": 0,
            },
            "ew1": {
                "kend": 2,
                "kaxis": 0,
            },
            "ew2": {
                "kend": 2,
                "kaxis": 0,
            },
            "es1": {
                "kend": 2,
                "kaxis": 0,
            },
            "es2": {
                "kend": 2,
                "kaxis": 0,
            },
            "ee1": {
                "kend": 2,
                "kaxis": 0,
            },
            "ee2": {
                "kend": 2,
                "kaxis": 0,
            },
        }

    inputs: Dict[str, Any] = {
        "gridvar": {
            "name": "grid",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM, LON_OR_LAT_DIM],
            "units": "radians",
        },
        "agrid": {
            "name": "agrid",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, LON_OR_LAT_DIM],
            "units": "radians",
        },
        "area": {
            "name": "area",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "m^2",
        },
        "area_c": {
            "name": "area_cgrid",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "m^2",
        },
        "dx": {
            "name": "dx",
            "dims": [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "m",
        },
        "dy": {
            "name": "dy",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM],
            "units": "m",
        },
        "dxc": {
            "name": "dx_cgrid",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM],
            "units": "m",
        },
        "dyc": {
            "name": "dy_cgrid",
            "dims": [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "m",
        },
        "dxa": {
            "name": "dx_agrid",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "m",
        },
        "dya": {
            "name": "dy_agrid",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "m",
        },
        "npz": {
            "name": "npz",
            "dims": [],
            "units": "",
        },
    }
    outputs: Dict[str, Any] = {
        "ks": {
            "name": "ks",
            "dims": [],
            "units": "",
        },
        "ptop": {
            "name": "ptop",
            "dims": [],
            "units": "mb",
        },
        "ak": {
            "name": "ak",
            "dims": [fv3util.Z_INTERFACE_DIM],
            "units": "mb",
        },
        "bk": {
            "name": "bk",
            "dims": [fv3util.Z_INTERFACE_DIM],
            "units": "",
        },
        "ec1": {
            "name": "ec1",
            "dims": [CARTESIAN_DIM, fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "ec2": {
            "name": "ec2",
            "dims": [CARTESIAN_DIM, fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "ew1": {
            "name": "ew1",
            "dims": [CARTESIAN_DIM, fv3util.X_INTERFACE_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "ew2": {
            "name": "ew2",
            "dims": [CARTESIAN_DIM, fv3util.X_INTERFACE_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "es1": {
            "name": "es1",
            "dims": [CARTESIAN_DIM, fv3util.X_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "",
        },
        "es2": {
            "name": "es2",
            "dims": [CARTESIAN_DIM, fv3util.X_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "",
        },
        "cos_sg1": {
            "name": "cos_sg1",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "sin_sg1": {
            "name": "sin_sg1",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "cos_sg2": {
            "name": "cos_sg2",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "sin_sg2": {
            "name": "sin_sg2",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "cos_sg3": {
            "name": "cos_sg3",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "sin_sg3": {
            "name": "sin_sg3",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "cos_sg4": {
            "name": "cos_sg4",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "sin_sg4": {
            "name": "sin_sg4",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "cos_sg5": {
            "name": "cos_sg5",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "sin_sg5": {
            "name": "sin_sg5",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "cos_sg6": {
            "name": "cos_sg6",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "sin_sg6": {
            "name": "sin_sg6",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "cos_sg7": {
            "name": "cos_sg7",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "sin_sg7": {
            "name": "sin_sg7",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "cos_sg8": {
            "name": "cos_sg8",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "sin_sg8": {
            "name": "sin_sg8",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "cos_sg9": {
            "name": "cos_sg9",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "sin_sg9": {
            "name": "sin_sg9",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "l2c_v": {
            "name": "l2c_v",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM],
            "units": "",
            "n_halo": 0,
        },
        "l2c_u": {
            "name": "l2c_u",
            "dims": [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "",
            "n_halo": 0,
        },
        "ee1": {
            "name": "ee1",
            "dims": [CARTESIAN_DIM, fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "",
        },
        "ee2": {
            "name": "ee2",
            "dims": [CARTESIAN_DIM, fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "",
        },
        "cosa_u": {
            "name": "cosa_u",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "cosa_v": {
            "name": "cosa_v",
            "dims": [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "",
        },
        "cosa_s": {
            "name": "cosa_s",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "sina_u": {
            "name": "sina_u",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "sina_v": {
            "name": "sina_v",
            "dims": [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "",
        },
        "rsin_u": {
            "name": "rsin_u",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "rsin_v": {
            "name": "rsin_v",
            "dims": [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "",
        },
        "rsina": {
            "name": "rsina",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "",
            "n_halo": 0,
        },
        "rsin2": {"name": "rsin2", "dims": [fv3util.X_DIM, fv3util.Y_DIM], "units": ""},
        "cosa": {
            "name": "cosa",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "",
        },
        "sina": {
            "name": "sina",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "",
        },
        "divg_u": {
            "name": "divg_u",
            "dims": [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "",
        },
        "divg_v": {
            "name": "divg_v",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "del6_u": {
            "name": "del6_u",
            "dims": [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "",
        },
        "del6_v": {
            "name": "del6_v",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "vlon": {
            "name": "vlon",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, CARTESIAN_DIM],
            "units": "",
            "n_halo": 2,
        },
        "vlat": {
            "name": "vlat",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, CARTESIAN_DIM],
            "units": "",
            "n_halo": 2,
        },
        "z11": {
            "name": "z11",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
            "n_halo": 1,
        },
        "z12": {
            "name": "z12",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
            "n_halo": 1,
        },
        "z21": {
            "name": "z21",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
            "n_halo": 1,
        },
        "z22": {
            "name": "z22",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
            "n_halo": 1,
        },
        "a11": {
            "name": "a11",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
            "n_halo": 1,
        },
        "a12": {
            "name": "a12",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
            "n_halo": 1,
        },
        "a21": {
            "name": "a21",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
            "n_halo": 1,
        },
        "a22": {
            "name": "a22",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
            "n_halo": 1,
        },
        "edge_s": {
            "name": "edge_s",
            "dims": [fv3util.X_INTERFACE_DIM],
            "units": "",
            "n_halo": 0,
        },
        "edge_n": {
            "name": "edge_n",
            "dims": [fv3util.X_INTERFACE_DIM],
            "units": "",
            "n_halo": 0,
        },
        "edge_e": {
            "name": "edge_e",
            "dims": [fv3util.Y_INTERFACE_DIM],
            "units": "",
            "n_halo": 0,
        },
        "edge_w": {
            "name": "edge_w",
            "dims": [fv3util.Y_INTERFACE_DIM],
            "units": "",
            "n_halo": 0,
        },
        "edge_vect_s": {
            "name": "edge_vect_s",
            "dims": [fv3util.X_DIM],
            "units": "",
        },
        "edge_vect_n": {
            "name": "edge_vect_n",
            "dims": [fv3util.X_DIM],
            "units": "",
        },
        "edge_vect_e": {
            "name": "edge_vect_e",
            "dims": [fv3util.Y_DIM],
            "units": "",
        },
        "edge_vect_w": {
            "name": "edge_vect_w",
            "dims": [fv3util.Y_DIM],
            "units": "",
        },
        "da_min": {
            "name": "da_min",
            "dims": [],
            "units": "m^2",
        },
        "da_min_c": {
            "name": "da_min_c",
            "dims": [],
            "units": "m^2",
        },
        "da_max": {
            "name": "da_max",
            "dims": [],
            "units": "m^2",
        },
        "da_max_c": {
            "name": "da_max_c",
            "dims": [],
            "units": "m^2",
        },
    }

    def compute_parallel(self, inputs, communicator):
        namelist = spec.namelist
        grid_generator = MetricTerms.from_tile_sizing(
            npx=namelist.npx,
            npy=namelist.npy,
            npz=inputs["npz"],
            communicator=communicator,
            backend=global_config.get_backend(),
        )
        input_state = self.state_from_inputs(inputs)
        grid_generator._grid = input_state["grid"]
        grid_generator._agrid = input_state["agrid"]
        state = {}
        for metric_term, metadata in self.outputs.items():
            state[metadata["name"]] = getattr(grid_generator, metric_term)
        return self.outputs_from_state(state)
