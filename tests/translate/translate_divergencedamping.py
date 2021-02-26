import fv3core.stencils.divergence_damping as dd
from fv3core.testing import TranslateFortranData2Py
import numpy as np

class TranslateDivergenceDamping(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)
        self.compute_func = dd.compute
        self.in_vars["data_vars"] = {
            "u": {},
            "v": {},
            "va": {},
            "ptc": {},
            "vort": {},
            "ua": {},
            "divg_d": {},
            "vc": {},
            "uc": {},
            "delpc": {},
            "ke": {},
            "wk": {},
            "nord_col": {},
            "d2_bg": {},
        }
        self.in_vars["parameters"] = ["dt"]
        self.out_vars = {
            "vort": {},
            "ke": {"iend": grid.ied + 1, "jend": grid.jed + 1},
            "delpc": {},
        }
        self.max_error = 3.0e-11

    def compute(self, inputs):
        inputs['nord_col'] = np.asarray([int(x) for x in inputs['nord_col'][0, 0, :]])
        return self.column_split_compute(inputs, {"nord": "nord_col", "d2_bg": "d2_bg"})
