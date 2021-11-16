import gt4py
import numpy as np
from gt4py import gtscript
from gt4py.gtscript import PARALLEL, Field, computation, interval, stencil

from fv3core.decorators import FrozenStencil, computepath_function, computepath_method


backend = "gtc:numpy"


def test_stencil(
    input_field: Field[np.float64], mid_field: Field[np.float64], output_field: Field[np.float64]
):
    with computation(PARALLEL), interval(...):
        mid_field = input_field
        mid_field += 1
        output_field = mid_field


class dummy_stencil_class:
    def __init__(self, origin, domain) -> None:
        self.mid_field = gt4py.storage.zeros(
            backend, default_origin=origin, shape=(5, 5, 1), dtype=np.float64
        )
        self.mid_field._istransient = True
        self.basic_adjust_divide_stencil = FrozenStencil(
            test_stencil,
            origin=origin,
            domain=domain,
        )

    @computepath_method(use_dace=True)
    def run(self, input_field, output_field):
        self.basic_adjust_divide_stencil(input_field, self.mid_field, output_field)


if __name__ == "__main__":
    domain = (4, 4, 1)
    origin = (0, 0, 0)
    input_field = gt4py.storage.ones(
        backend, default_origin=origin, shape=(5, 5, 1), dtype=np.float64
    )
    output_field = gt4py.storage.zeros(
        backend, default_origin=origin, shape=(5, 5, 1), dtype=np.float64
    )
    myclass = dummy_stencil_class(origin, domain)
    myclass.run(input_field, output_field)
    print(output_field[:, :, 0])
