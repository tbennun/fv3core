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
from dace.transformation.dataflow import MapCollapse
from dace.sdfg.analysis import cutout as cutter
from typing import Any, Dict, Iterator, List, Tuple
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
        super().__init__(sdfg=sdfg, task='stencil')
        self.instrument = measurement

    def cutouts(self) -> Iterator[Tuple[SDFGState, StencilComputation]]:
        for node, state in self._sdfg.all_nodes_recursive():
            if isinstance(node, StencilComputation):
                node_id = state.node_id(node)
                state_id = self._sdfg.node_id(state)
                yield (state_id, node_id), (state, node)

    def space(self,
              node: StencilComputation) -> Iterator[ExpansionSpecification]:
        #yield node._expansion_specification
        #return
        MAX_TOTAL_TILE_SIZE = 4096
        #TILE_SIZES = (1, 8, 16, 32, 64)
        TILE_SIZES = (1, 8, 16, 64)
        #TILE_SIZES = (0,)
        SEQUENTIAL_OPTS = (False,)  # True)
        
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
            options = [SEQUENTIAL_OPTS, TILE_SIZES, TILE_SIZES, TILE_SIZES]
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
                collapse = 0

                # Set tiles
                if tile_i != 0:
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
                    yield obj, collapse
                else:
                    # TODO: Collapse
                    yield obj, 0
                    yield obj, 1
                        


    def evaluate(self, state: dace.SDFGState, node: dace.nodes.Node, dreport, measurements: int) -> Dict:
        cutout = cutter.cutout_state(state, node, make_copy=False)
        cutout.instrument = self.instrument
                
        arguments = {}
        for cstate in cutout.nodes():
            for dnode in cstate.data_nodes():
                if cutout.arrays[dnode.data].transient:
                    continue

                arguments[dnode.data] = dreport.get_first_version(dnode.data)

        results = {}
        label = f'{self.rank + 1}/{self.num_ranks}: {node.label}'   
        for spec, collapse in tqdm(list(self.space(node)), desc=label):
            node._expansion_specification = spec

            runtime = self.measure(cutout, arguments, collapse, measurements)
            results[str(spec) + f", collapse: {collapse}"] = runtime

        return results

    def evaluate_single(self, config: Any, cutout: dace.SDFG, arguments: Dict[str, Any], state: dace.SDFGState, node: dace.nodes.Node, dreport, measurements: int) -> Tuple[Any, Any]:
        spec, collapse = config
        node._expansion_specification = spec
        runtime = self.measure(cutout, arguments, collapse, measurements)
        return (str(spec) + f", collapse: {collapse}"), runtime
    
    def measure(self,
                sdfg: dace.SDFG,
                arguments: Dict[str, dace.data.ArrayLike],
                collapse: int,
                repetitions: int = 30) -> float:
        with dace.config.set_temporary('debugprint', value=False):
            sdfg = sdfg.from_json(sdfg.to_json())
            sdfg.build_folder = '/dev/shm'
            sdfg.expand_library_nodes()
            sdfg.apply_transformations_repeated(ViewRemove)
            sdfg.simplify()
            sdfg.apply_transformations_repeated(MapCollapse)
            if collapse != 0:
                for node, _ in sdfg.all_nodes_recursive():
                    if isinstance(node, dace.nodes.MapEntry):
                        node.collapse = len(node.params)

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
    sdfg = dace.SDFG.from_file('aha-fvtp2d-c128.sdfg')
    sdfg.build_folder = f'.gt_cache/dacecache/{sdfg.name}'
    if False:
        sdfg.expand_library_nodes()
        sdfg.apply_transformations_repeated(ViewRemove)
        sdfg.simplify()
        tuner = SequentialDataLayoutTuner(sdfg)
        report = tuner.optimize(group_by=TuningGroups.Inputs_Outputs_Dimension)
    else:
        tuner = StencilTuner(sdfg)
        dist = optim.DistributedSpaceTuner(tuner)
        report = dist.optimize()


    print(report)
