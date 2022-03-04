# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import dace.optimization as optim
from dace.optimization.data_layout_tuner import TuningGroups
import numpy as np

from dace.transformation import SingleStateTransformation
from dace.transformation.transformation import PatternNode
from dace.sdfg.utils import node_path_graph, get_view_edge, get_view_node


class ViewRemove(SingleStateTransformation):
    a = PatternNode(dace.nodes.AccessNode)
    v = PatternNode(dace.nodes.AccessNode)

    @classmethod
    def expressions(cls):
        return [node_path_graph(cls.a, cls.v), node_path_graph(cls.v, cls.a)]

    def can_be_applied(self,
                       graph: dace.SDFGState,
                       expr_index: int,
                       sdfg: dace.SDFG,
                       permissive: bool = False) -> bool:
        if not isinstance(self.v.desc(sdfg), dace.data.View):
            return False
        if type(self.a.desc(sdfg)) != dace.data.Array:
            return False
        # if graph.in_degree(self.a) > 0:
        #     return False
        if get_view_node(graph, self.v) != self.a:
            return False
        if self.a.desc(sdfg).strides != self.v.desc(sdfg).strides:
            return False
        return True

    def apply(self, graph: dace.SDFGState, sdfg: dace.SDFG):
        # If the strides are the same, offset the range by memlet and reconnect
        view_edge = get_view_edge(graph, self.v)
        if self.expr_index == 0:
            for e in graph.out_edges(self.v):
                for me in graph.memlet_tree(e):
                    me.data.subset.offset(view_edge.data.subset, False)
                    me.data.data = self.a.data
                graph.remove_edge(e)
                graph.add_edge(self.a, e.src_conn, e.dst, e.dst_conn, e.data)
        else:
            for e in graph.in_edges(self.v):
                for me in graph.memlet_tree(e):
                    me.data.subset.offset(view_edge.data.subset, False)
                    me.data.data = self.a.data
                graph.remove_edge(e)
                graph.add_edge(e.src, e.src_conn, self.a, e.dst_conn, e.data)
        graph.remove_node(self.v)


import itertools
import json
import math
import os
from dace import SDFG, SDFGState, dtypes
from dace.optimization import cutout_tuner
from dace.transformation import helpers as xfh
from dace.sdfg.analysis import cutout as cutter
from typing import Dict, Iterator, List, Tuple
try:
    from tqdm import tqdm
except (ImportError, ModuleNotFoundError):
    tqdm = lambda x, **kwargs: x

from gtc.dace.nodes import Map, dcir, StencilComputation, ExpansionItem, make_expansion_order

ExpansionSpecification = List[ExpansionItem]


class StencilTuner(cutout_tuner.CutoutTuner):
    def __init__(
        self,
        sdfg: SDFG,
        measurement: dtypes.InstrumentationType = dtypes.InstrumentationType.
        Timer,
    ) -> None:
        super().__init__(sdfg=sdfg)
        self.instrument = measurement

    def cutouts(self) -> Iterator[Tuple[SDFGState, StencilComputation]]:
        for node, state in self._sdfg.all_nodes_recursive():
            if isinstance(node, StencilComputation):
                yield state, node

    def space(self,
              node: StencilComputation) -> Iterator[ExpansionSpecification]:
        MAX_TOTAL_TILE_SIZE = 4096
        #TILE_SIZES = (1, 8, 16, 32, 64)
        TILE_SIZES = (1, 8)
        
        for order in node.valid_expansion_orders():
            # No I/J loops or tiled K loop
            if 'ILoop' in order or 'JLoop' in order or ('TileK' in order and 'KLoop' in order):
                continue
            # If tiles are not contiguous in order
            seen_nontiles = False
            skip = False
            for elem in order:
                if 'Tile' in elem:
                    if seen_nontiles: # Already seen the end of tiles
                        skip = True
                        break
                if elem[0] in ('I', 'J', 'K'):
                    seen_nontiles = True
            if skip:
                continue
            
            # Now that we have a valid computational layout, we can tune the following:
            # * OpenMP/Sequential maps
            #     * If OpenMP: whether to collapse or not
            # * Tile sizes (if exist in order): [4,] 8, 16, 32, 64[, 128] (also product does not exceed max tile size)
            options = [(False, True), TILE_SIZES, TILE_SIZES, TILE_SIZES]
            for sequential, tile_i, tile_j, tile_k in itertools.product(*options):
                # Check validity of tiles
                if tile_i > 1 and 'TileI' not in order:
                    continue
                if tile_j > 1 and 'TileJ' not in order:
                    continue
                if tile_k > 1 and 'TileK' not in order:
                    continue
                if tile_i * tile_j * tile_k > MAX_TOTAL_TILE_SIZE:
                    continue

                obj = make_expansion_order(node, order)

                # Set tiles
                tile_sizes = {dcir.Axis.I: tile_i, dcir.Axis.J: tile_j, dcir.Axis.K: tile_k}
                for expansion_item in obj:
                    if isinstance(expansion_item, Map):
                        for it in expansion_item.iterations:
                            if it.kind == 'tiling':
                                it.stride = tile_sizes[it.axis]

                # Set sequential / parallel maps
                if sequential:
                    for expansion_item in obj:
                        if isinstance(expansion_item, Map):
                            expansion_item.schedule = dtypes.ScheduleType.Sequential
                else:
                    # TODO: Collapse
                    pass
                        
                yield obj

    def optimize(self, apply: bool = True, measurements: int = 30) -> Dict:
        dreport = self._sdfg.get_instrumented_data()

        tuning_report = {}
        for state, node in tqdm(list(self.cutouts())):
            if os.path.exists(f'{node.label}.stuning'):
                print(f'Using cached {node.label}')
                with open(f'{node.label}.stuning', 'r') as fp:
                    tuning_report[node.label] = json.load(fp)
                continue
            cutout = cutter.cutout_state(state, node, make_copy=False)
            cutout.instrument = self.instrument

            arguments = {}
            for cstate in cutout.nodes():
                for dnode in cstate.data_nodes():
                    if cutout.arrays[dnode.data].transient:
                        continue

                    arguments[dnode.data] = dreport.get_first_version(dnode.data)

            results = {}
            best_choice = None
            best_runtime = math.inf
            for spec in tqdm(list(self.space(node)), desc=node.label):
                node._expansion_specification = spec

                runtime = self.measure(cutout, arguments, measurements)
                results[str(spec)] = runtime

                if runtime < best_runtime:
                    best_choice = spec
                    best_runtime = runtime

            if apply and best_choice is not None:
                node._expansion_specification = spec

            tuning_report[node.label] = results
            with open(f'{node.label}.stuning', 'w') as fp:
                json.dump(results, fp)
        return tuning_report

    def measure(self,
                sdfg: dace.SDFG,
                    arguments: Dict[str, dace.data.ArrayLike],
                repetitions: int = 30) -> float:
        with dace.config.set_temporary('debugprint', value=False):
            sdfg = sdfg.from_json(sdfg.to_json())
            sdfg.expand_library_nodes()
            sdfg.apply_transformations_repeated(ViewRemove)
            sdfg.simplify()
        return super().measure(sdfg, arguments, repetitions)


class SequentialDataLayoutTuner(optim.DataLayoutTuner):
    def measure(self,
                sdfg: dace.SDFG,
                arguments: Dict[str, dace.data.ArrayLike],
                repetitions: int = 30) -> float:
        for node, _ in sdfg.all_nodes_recursive():
            if isinstance(node, dace.nodes.MapEntry):
                node.schedule = dace.ScheduleType.Sequential
        return super().measure(sdfg, arguments, repetitions)


if __name__ == '__main__':
    sdfg = dace.SDFG.from_file('aha-12792.sdfg')
    sdfg.build_folder = '.gt_cache/dacecache/fv3core_stencils_fvtp2d_FiniteVolumeTransport___call__'
    if False:
        sdfg.expand_library_nodes()
        sdfg.apply_transformations_repeated(ViewRemove)
        sdfg.simplify()
        tuner = SequentialDataLayoutTuner(sdfg)
        report = tuner.optimize(group_by=TuningGroups.Inputs_Outputs_Dimension)
    else:
        tuner = StencilTuner(sdfg)
        report = tuner.optimize()


    print(report)
