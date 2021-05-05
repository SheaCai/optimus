import math
import numpy as np
from scipy.optimize import minimize
import copy

from collections import namedtuple

from .. import util
from .layer import ConvLayer, LocalRegionLayer
from .network import Network
from .resource import Resource
from . import schedule_generator
from . import loop_enum as le


class InterLayerReuse(object):
    """
    Inter-layer reuse.
    """

    SchedIndex = namedtuple('SchedIndex', ['sp_idx', 'tm_idx'])
    Scale = namedtuple('Scale', ['s_h', 's_w'])
    MinSize = namedtuple('MinSize', ['h', 'w'])

    def __init__(self, network, fusion_group, resource, loop_lower_bound, topological_order=True,
                 z_fusion=False, d_fusion=False, womincost=False):
        if not isinstance(network, Network):
            raise TypeError('InterLayerReuse: network must be '
                            'a Network instance.')

        if not isinstance(fusion_group, list):
            raise TypeError('InterLayerReuse: fusion_group must be '
                            'a list.')

        if not isinstance(resource, Resource):
            raise TypeError('InterLayerPipeline: resource must be '
                            'a Resource instance.')

        self.network = network
        self.fusion_group = fusion_group
        self.resource = resource
        self.loop_lower_bound = loop_lower_bound
        self.topological_order = topological_order
        self.z_fusion = z_fusion
        self.d_fusion = d_fusion
        self.womincost = womincost

        self.valid = self._prepare()
        if not self.valid:
            return

        # self._calc_sched_dag()
        #
        # self.valid = self._init_alternate_pair()
        # if not self.valid:
        #     return

    def sched(self, mode):
        if self.valid:
            self._calc_sched_dag()
            self.valid = self._init_alternate_pair(mode)

    def _prepare(self):
        self.firsts = []
        self.lasts = []
        self.ext_inputs_dict = dict()
        self.ext_outputs = set()
        self.fused_weight_size = 0
        self.fused_input_size = 0
        self.fused_output_size = 0

        if self.d_fusion:
            for layer in self.fusion_group:
                if len(self.network.nexts(layer)) > 1 or len(self.network.prevs(layer)) > 1:
                    self.valid = False
                    return False

        for layer in self.fusion_group:
            tmp = tuple()
            for nx in self.network.nexts(layer):
                if nx not in self.fusion_group:
                    tmp += (nx, )
                    self.ext_outputs.add(layer)
            if tmp == self.network.nexts(layer):
                self.lasts.append(layer)

            tmp = tuple()
            for pre in self.network.prevs(layer):
                if pre not in self.fusion_group:
                    tmp += (pre, )
                    if pre not in self.ext_inputs_dict:
                        self.ext_inputs_dict[pre] = [layer]
                    else:
                        self.ext_inputs_dict[pre].append(layer)
            if tmp == self.network.prevs(layer):
                if isinstance(self.network[layer], LocalRegionLayer):
                    return False
                self.firsts.append(layer)
            if isinstance(self.network[layer], ConvLayer):
                self.fused_weight_size += self.network[layer].total_filter_size

        for ip in self.ext_inputs_dict:
            if ip is None:
                self.fused_input_size += self.network[self.network.INPUT_LAYER_KEY].total_ofmap_size
            else:
                self.fused_input_size += self.network[ip].total_ofmap_size
        for op in self.ext_outputs:
            self.fused_output_size += self.network[op].total_ofmap_size
        return True

    def _calc_sched_dag(self):

        # The DAG vertex list in the topological order.
        if self.topological_order:
            self.dag_vertex_list = self._topological_order()
        else:
            self.dag_vertex_list = self.fusion_group

        # Make a directory from layer name to DAG vertex index.
        self.dag_vertex_dict = {}

        for vidx, layer_name in enumerate(self.dag_vertex_list):
            assert layer_name not in self.dag_vertex_dict
            self.dag_vertex_dict[layer_name] = vidx

        # The previous and next relationship of the DAG vertices.
        self.dag_prev_dict = dict((vidx, set()) for vidx
                                  in range(len(self.dag_vertex_list)))
        self.dag_next_dict = dict((vidx, set()) for vidx
                                  in range(len(self.dag_vertex_list)))

        for layer_name in self.fusion_group:
            vidx = self.dag_vertex_dict[layer_name]

            # Previous layers.
            for p in self.network.prevs(layer_name):
                if not p or p not in self.fusion_group:
                    continue
                pvidx = self.dag_vertex_dict[p]
                if pvidx != vidx:
                    self.dag_prev_dict[vidx].add(pvidx)

            # Next layers.
            for n in self.network.nexts(layer_name):
                if not n or n not in self.fusion_group:
                    continue
                nvidx = self.dag_vertex_dict[n]
                if nvidx != vidx:
                    self.dag_next_dict[vidx].add(nvidx)

        self.ext_inputs_idx = {}
        for vidx, layer_name in enumerate(self.ext_inputs_dict.keys()):
            assert layer_name not in self.dag_vertex_dict
            self.ext_inputs_idx[layer_name] = vidx + len(self.dag_vertex_list)

    def _topological_order(self):

        # The visited layers in the DFS order.
        visited = []
        # The unseen pending layers.
        unseen = set(self.fusion_group)
        # The layers that have been seen, but not visited due to unvisited
        # previous layers.
        seen = set()

        def _dfs(vertex):
            assert vertex not in seen
            if vertex in visited:
                return

            unseen.discard(vertex)
            seen.add(vertex)

            next_vertices = []

            for n in reversed(self.network.nexts(vertex)):
                if n and n not in next_vertices and n in unseen:
                    next_vertices.append(n)

            for nv in next_vertices:
                _dfs(nv)

            visited.append(vertex)
            seen.remove(vertex)

        # Start from the first layers.
        for v in self.firsts:
            _dfs(v)
        assert not unseen
        assert not seen

        return list(reversed(visited))

    def ordered_layer_list(self):

        return list(sum(self.dag_vertex_list, tuple()))

    def _init_scale(self):
        scale_tmp = [None for _ in self.dag_vertex_list]

        for idx, l in enumerate(self.dag_vertex_list):
            layer = self.network[l]
            if l in self.firsts:
                scale_tmp[idx] = [layer.hstd, layer.wstd]
                continue

            max_hs, max_ws = 0, 0
            for src in self.dag_prev_dict[idx]:
                src_scale = scale_tmp[src]
                assert src_scale
                max_hs = src_scale[0] if src_scale[0] > max_hs else max_hs
                max_ws = src_scale[1] if src_scale[1] > max_ws else max_ws
            scale_tmp[idx] \
                = [max_hs * layer.hstd, max_ws * layer.wstd]

        self.scale = [None for _ in self.dag_vertex_list]

        last_h = []
        last_w = []
        for l in self.lasts:
            idx = self.dag_vertex_dict[l]
            last_h.append(scale_tmp[idx][0])
            last_w.append(scale_tmp[idx][1])
        s_h = util.lcm(*last_h)
        s_w = util.lcm(*last_w)

        for l in reversed(self.dag_vertex_list):
            idx = self.dag_vertex_dict[l]
            if l in self.lasts:
                self.scale[idx] = \
                    InterLayerReuse.Scale(s_h / scale_tmp[idx][0], s_w / scale_tmp[idx][1])
                continue

            s_h_tmp, s_w_tmp = None, None
            for dst_idx in self.dag_next_dict[idx]:
                dst = self.dag_vertex_list[dst_idx]
                dst_layer = self.network[dst]
                dst_scale = self.scale[dst_idx]
                assert dst_scale
                if s_h_tmp is None and s_w_tmp is None:
                    s_h_tmp, s_w_tmp = dst_layer.hstd * dst_scale.s_h, dst_layer.wstd * dst_scale.s_w
                else:
                    assert s_h_tmp == dst_layer.hstd * dst_scale.s_h \
                           and s_w_tmp == dst_layer.wstd * dst_scale.s_w
            self.scale[idx] = \
                InterLayerReuse.Scale(s_h_tmp, s_w_tmp)

        self.minSize = [None for _ in self.dag_vertex_list]
        for l in reversed(self.dag_vertex_list):
            idx = self.dag_vertex_dict[l]
            layer = self.network[l]
            if l in self.lasts:
                self.minSize[idx] = InterLayerReuse.MinSize(self.scale[idx].s_h, self.scale[idx].s_w)
                continue

            h_tmp, w_tmp = None, None
            for dst_idx in self.dag_next_dict[idx]:
                dst = self.dag_vertex_list[dst_idx]
                dst_layer = self.network[dst]
                dst_minsize = self.minSize[dst_idx]
                assert dst_minsize
                if isinstance(dst_layer, LocalRegionLayer):
                    hreg, wreg = dst_layer.hreg, dst_layer.wreg
                else:
                    hreg, wreg = dst_layer.hfil, dst_layer.wfil
                if h_tmp is None and w_tmp is None:
                    h_tmp = (dst_minsize.h - 1) * dst_layer.hstd + hreg
                    w_tmp = (dst_minsize.w - 1) * dst_layer.wstd + wreg
                    h_tmp = layer.hofm if h_tmp > layer.hofm else h_tmp
                    w_tmp = layer.wofm if w_tmp > layer.wofm else w_tmp
                else:
                    if (dst_minsize.h - 1) * dst_layer.hstd + hreg > h_tmp:
                        h_tmp = (dst_minsize.h - 1) * dst_layer.hstd + hreg
                        h_tmp = layer.hofm if h_tmp > layer.hofm else h_tmp
                    if (dst_minsize.w - 1) * dst_layer.wstd + wreg > w_tmp:
                        w_tmp = (dst_minsize.w - 1) * dst_layer.wstd + wreg
                        w_tmp = layer.wofm if w_tmp > layer.wofm else w_tmp
            self.minSize[idx] = InterLayerReuse.MinSize(h_tmp, w_tmp)

    def _init_alternate_pair_optimus(self):
        self._init_scale()
        irrelevant = [le.D, le.R, le.K, le.C]
        self.loop_block = [None for _ in self.dag_vertex_list]
        self.loop_order = [None for _ in self.dag_vertex_list]

        self.min_feature_footprint, self.is_full_buffer, self.add_one_line_footprint = self._alternate()

        if self.is_full_buffer is None:
            return False

        level = self.resource.buffer_levels() - 2
        s = self.resource.buffer(level).capacity * self. resource.paras[level].count

        if s <= self.min_feature_footprint:
            return False

        if s >= self.fused_weight_size + self.min_feature_footprint:
            self.sfil_fit = True
            self.tile_num = 1

            h_m = self.network.input_layer().hofm
            w_m = self.network.input_layer().wofm
            for l in self.lasts:
                if self.network[l].hofm < h_m:
                    h_m = self.network[l].hofm
                if self.network[l].wofm < w_m:
                    w_m = self.network[l].wofm

            s = s - self.fused_weight_size
            line_num = math.floor((s - self.min_feature_footprint) /
                                  self.add_one_line_footprint) + 1
            if line_num > h_m:
                h = h_m
                b = int(max(s // ((h_m - 1) * self.add_one_line_footprint + self.min_feature_footprint),
                            self.network.input_layer().nimg))
            else:
                h = line_num
                b = 1

            H = [None for _ in self.dag_vertex_list]
            for l in reversed(self.dag_vertex_list):
                idx = self.dag_vertex_dict[l]
                layer = self.network[l]
                if l in self.lasts:
                    H[idx] = int(self.scale[idx].s_h * h)
                    continue

                h_tmp = None
                for dst_idx in self.dag_next_dict[idx]:
                    dst = self.dag_vertex_list[dst_idx]
                    dst_layer = self.network[dst]
                    dst_h = H[dst_idx]
                    assert dst_h is not None
                    if isinstance(dst_layer, LocalRegionLayer):
                        hreg = dst_layer.hreg
                    else:
                        hreg = dst_layer.hfil
                    if h_tmp is None:
                        h_tmp = min((dst_h - 1) * dst_layer.hstd + hreg, layer.hofm)
                    else:
                        if (dst_h - 1) * dst_layer.hstd + hreg > h_tmp:
                            h_tmp = min((dst_h - 1) * dst_layer.hstd + hreg, layer.hofm)
                H[idx] = math.floor(h_tmp)

            for l in self.dag_vertex_list:
                idx = self.dag_vertex_dict[l]
                layer = self.network[l]
                if isinstance(layer, LocalRegionLayer):
                    continue
                loop_lower_bound = self.loop_lower_bound(layer)
                kk = min(max(loop_lower_bound.k, loop_lower_bound.c), layer.nofm)
                k = layer.nofm if self.is_full_buffer[idx] else kk
                c = min(max(loop_lower_bound.k, loop_lower_bound.c), layer.nifm)

                for pre in self.dag_prev_dict[idx]:
                    if self.is_full_buffer[pre]:
                        c = layer.nifm

                if k < layer.nofm and c < layer.nifm:
                    c = layer.nifm
                self.loop_block[idx] = [layer.wfil, layer.hfil, c, layer.wofm, H[idx], k, b]
                self.loop_order[idx] = schedule_generator.loop_order_generator(layer, self.loop_block[idx], irrelevant)

        else:
            if self.z_fusion or self.d_fusion:
                return False
            self.sfil_fit = False
            h_m = self.network.input_layer().hofm
            w_m = self.network.input_layer().wofm
            for l in self.lasts:
                if self.network[l].hofm < h_m:
                    h_m = self.network[l].hofm
                if self.network[l].wofm < w_m:
                    w_m = self.network[l].wofm

            line_num = math.floor((s - self.min_feature_footprint) / self.add_one_line_footprint) + 1
            if line_num > h_m:
                h = h_m
                b = int(min(s // ((h_m - 1) * self.add_one_line_footprint + self.min_feature_footprint),
                            self.network.input_layer().nimg))
            else:
                h = line_num
                b = 1

            H = [None for _ in self.dag_vertex_list]
            for l in reversed(self.dag_vertex_list):
                idx = self.dag_vertex_dict[l]
                layer = self.network[l]
                if l in self.lasts:
                    H[idx] = int(self.scale[idx].s_h * h)
                    continue

                h_tmp = None
                for dst_idx in self.dag_next_dict[idx]:
                    dst = self.dag_vertex_list[dst_idx]
                    dst_layer = self.network[dst]
                    dst_h = H[dst_idx]
                    assert dst_h is not None
                    if isinstance(dst_layer, LocalRegionLayer):
                        hreg = dst_layer.hreg
                    else:
                        hreg = dst_layer.hfil
                    if h_tmp is None:
                        h_tmp = min((dst_h - 1) * dst_layer.hstd + hreg, layer.hofm)
                    else:
                        if (dst_h - 1) * dst_layer.hstd + hreg > h_tmp:
                            h_tmp = min((dst_h - 1) * dst_layer.hstd + hreg, layer.hofm)
                H[idx] = math.floor(h_tmp)

            for l in self.dag_vertex_list:
                idx = self.dag_vertex_dict[l]
                layer = self.network[l]
                if isinstance(layer, LocalRegionLayer):
                    continue
                loop_lower_bound = self.loop_lower_bound(layer)

                k = layer.nofm if self.is_full_buffer[idx] \
                    else min(max(loop_lower_bound.k, loop_lower_bound.c), layer.nofm)
                c = min(max(loop_lower_bound.k, loop_lower_bound.c), layer.nifm)

                for pre in self.dag_prev_dict[idx]:
                    if self.is_full_buffer[pre]:
                        c = layer.nifm
                if k < layer.nofm and c < layer.nifm:
                    c = layer.nifm
                self.loop_block[idx] = [layer.wfil, layer.hfil, c, layer.wofm, H[idx], k, b]
                self.loop_order[idx] = \
                    schedule_generator.loop_order_generator(layer, self.loop_block[idx], irrelevant)

                self.tile_num = math.ceil(h_m * self.network.input_layer().nimg / (b * h))
        self.q = self.fused_weight_size * self.tile_num + self.fused_input_size + self.fused_output_size

        if self.network.net_name != "SqueezeNet":
            p2 = self.resource.access_cost[2]
            p1 = self.resource.access_cost[1]

            q0, q1, q2 = p1[0], p1[1], p2[2] + p1[2]

            f_args = (q0, q1, q2, b, h_m)
            fun = self.fun(f_args)
            c_args = (b, h_m, self.idx_dict)
            con = self.con(c_args)
            x0 = [1 for _ in range(len(set(self.idx_dict.values())) + 1)]
            if b > 1:
                x0[0] = b
            else:
                x0[0] = h
            for idx in self.idx_dict:
                if idx < len(self.dag_vertex_list):
                    layer = self.network[self.dag_vertex_list[idx]]
                    if isinstance(layer, LocalRegionLayer):
                        continue
                    loop_lower_bound = self.loop_lower_bound(layer)
                    x0[self.idx_dict[idx]] = loop_lower_bound.k

            x0 = np.asarray(x0)
            res = minimize(fun, x0, method='COBYLA', constraints=con)

            if res.success:
                if b > 1:
                    b = math.ceil(res.x[0])
                    h = h_m
                else:
                    b = 1
                    h = math.ceil(res.x[0])
                H = [None for _ in self.dag_vertex_list]
                for l in reversed(self.dag_vertex_list):
                    idx = self.dag_vertex_dict[l]
                    layer = self.network[l]
                    if l in self.lasts:
                        H[idx] = int(self.scale[idx].s_h * h)
                        continue

                    h_tmp = None
                    for dst_idx in self.dag_next_dict[idx]:
                        dst = self.dag_vertex_list[dst_idx]
                        dst_layer = self.network[dst]
                        dst_h = H[dst_idx]
                        assert dst_h is not None
                        if isinstance(dst_layer, LocalRegionLayer):
                            hreg = dst_layer.hreg
                        else:
                            hreg = dst_layer.hfil
                        if h_tmp is None:
                            h_tmp = min((dst_h - 1) * dst_layer.hstd + hreg, layer.hofm)
                        else:
                            if (dst_h - 1) * dst_layer.hstd + hreg > h_tmp:
                                h_tmp = min((dst_h - 1) * dst_layer.hstd + hreg, layer.hofm)
                    H[idx] = math.floor(h_tmp)

                for l in self.dag_vertex_list:
                    idx = self.dag_vertex_dict[l]
                    layer = self.network[l]
                    if isinstance(layer, LocalRegionLayer):
                        continue

                    if idx in self.idx_dict:
                        k = res.x[self.idx_dict[idx]]
                    else:
                        k = layer.nofm

                    if self.dag_prev_dict[idx] and list(self.dag_prev_dict[idx])[0] in self.idx_dict:
                        c = self.idx_dict[list(self.dag_prev_dict[idx])[0]]
                    else:
                        c = layer.nifm
                    self.loop_block[idx] = \
                        [layer.wfil, layer.hfil, math.ceil(c), layer.wofm, H[idx], math.ceil(k), b]

                    self.loop_order[idx] = \
                        schedule_generator.loop_order_generator(layer, self.loop_block[idx], irrelevant)

            q = self.fused_weight_size * self.network.input_layer().nimg * h_m * q2 / (b * h) \
                + self.fused_input_size * p2[0] + self.fused_output_size * p2[1]
            for l in self.dag_vertex_list:
                idx = self.dag_vertex_dict[l]
                layer = self.network[l]
                if isinstance(layer, LocalRegionLayer):
                    continue
                q += (q1 * layer.nifm * layer.total_ofmap_size / self.loop_block[idx][2])
                q += (q0 * layer.nofm * layer.total_ifmap_size / self.loop_block[idx][5])
            self.q = q
            self.tile_num = math.ceil(h_m * self.network.input_layer().nimg / (b * h))

        return True

    def _init_alternate_pair_others(self, mode):
        self._init_scale()
        irrelevant = [le.D, le.R, le.K, le.C]
        self.loop_block = [None for _ in self.dag_vertex_list]
        self.loop_order = [None for _ in self.dag_vertex_list]

        self.min_feature_footprint, self.is_full_buffer, self.add_one_line_footprint = self._alternate()

        if self.is_full_buffer is None:
            return False

        level = self.resource.buffer_levels() - 2
        s = self.resource.buffer(level).capacity * self. resource.paras[level].count

        if s <= self.min_feature_footprint:
            return False

        if s >= self.fused_weight_size + self.min_feature_footprint:
            self.sfil_fit = True
            self.tile_num = 1

            h_m = self.network.input_layer().hofm
            w_m = self.network.input_layer().wofm
            for l in self.lasts:
                if self.network[l].hofm < h_m:
                    h_m = self.network[l].hofm
                if self.network[l].wofm < w_m:
                    w_m = self.network[l].wofm

            s = s - self.fused_weight_size
            line_num = math.floor((s - self.min_feature_footprint) /
                                  self.add_one_line_footprint) + 1
            if line_num > h_m:
                h = h_m
                b = int(max(s // ((h_m - 1) * self.add_one_line_footprint + self.min_feature_footprint),
                            self.network.input_layer().nimg))
            else:
                h = line_num
                b = 1

            H = [None for _ in self.dag_vertex_list]
            for l in reversed(self.dag_vertex_list):
                idx = self.dag_vertex_dict[l]
                layer = self.network[l]
                if l in self.lasts:
                    H[idx] = int(self.scale[idx].s_h * h)
                    continue

                h_tmp = None
                for dst_idx in self.dag_next_dict[idx]:
                    dst = self.dag_vertex_list[dst_idx]
                    dst_layer = self.network[dst]
                    dst_h = H[dst_idx]
                    assert dst_h is not None
                    if isinstance(dst_layer, LocalRegionLayer):
                        hreg = dst_layer.hreg
                    else:
                        hreg = dst_layer.hfil
                    if h_tmp is None:
                        h_tmp = min((dst_h - 1) * dst_layer.hstd + hreg, layer.hofm)
                    else:
                        if (dst_h - 1) * dst_layer.hstd + hreg > h_tmp:
                            h_tmp = min((dst_h - 1) * dst_layer.hstd + hreg, layer.hofm)
                H[idx] = math.floor(h_tmp)

            for l in self.dag_vertex_list:
                idx = self.dag_vertex_dict[l]
                layer = self.network[l]
                if isinstance(layer, LocalRegionLayer):
                    continue
                loop_lower_bound = self.loop_lower_bound(layer)
                kk = min(max(loop_lower_bound.k, loop_lower_bound.c), layer.nofm)
                k = layer.nofm if self.is_full_buffer[idx] else kk
                c = min(max(loop_lower_bound.k, loop_lower_bound.c), layer.nifm)

                for pre in self.dag_prev_dict[idx]:
                    if self.is_full_buffer[pre]:
                        c = layer.nifm

                if k < layer.nofm and c < layer.nifm:
                    c = layer.nifm
                self.loop_block[idx] = [layer.wfil, layer.hfil, c, layer.wofm, H[idx], k, b]
                self.loop_order[idx] = schedule_generator.loop_order_generator(layer, self.loop_block[idx], irrelevant)

        else:
            if self.z_fusion or self.d_fusion:
                return False
            self.sfil_fit = False
            h_m = self.network.input_layer().hofm
            w_m = self.network.input_layer().wofm
            for l in self.lasts:
                if self.network[l].hofm < h_m:
                    h_m = self.network[l].hofm
                if self.network[l].wofm < w_m:
                    w_m = self.network[l].wofm

            line_num = math.floor((s - self.min_feature_footprint) / self.add_one_line_footprint) + 1
            if line_num > h_m:
                h = h_m
                b = int(min(s // ((h_m - 1) * self.add_one_line_footprint + self.min_feature_footprint),
                            self.network.input_layer().nimg))
            else:
                h = line_num
                b = 1

            H = [None for _ in self.dag_vertex_list]
            for l in reversed(self.dag_vertex_list):
                idx = self.dag_vertex_dict[l]
                layer = self.network[l]
                if l in self.lasts:
                    H[idx] = int(self.scale[idx].s_h * h)
                    continue

                h_tmp = None
                for dst_idx in self.dag_next_dict[idx]:
                    dst = self.dag_vertex_list[dst_idx]
                    dst_layer = self.network[dst]
                    dst_h = H[dst_idx]
                    assert dst_h is not None
                    if isinstance(dst_layer, LocalRegionLayer):
                        hreg = dst_layer.hreg
                    else:
                        hreg = dst_layer.hfil
                    if h_tmp is None:
                        h_tmp = min((dst_h - 1) * dst_layer.hstd + hreg, layer.hofm)
                    else:
                        if (dst_h - 1) * dst_layer.hstd + hreg > h_tmp:
                            h_tmp = min((dst_h - 1) * dst_layer.hstd + hreg, layer.hofm)
                H[idx] = math.floor(h_tmp)

            for l in self.dag_vertex_list:
                idx = self.dag_vertex_dict[l]
                layer = self.network[l]
                if isinstance(layer, LocalRegionLayer):
                    continue
                loop_lower_bound = self.loop_lower_bound(layer)

                k = layer.nofm if self.is_full_buffer[idx] \
                    else min(max(loop_lower_bound.k, loop_lower_bound.c), layer.nofm)
                c = min(max(loop_lower_bound.k, loop_lower_bound.c), layer.nifm)

                for pre in self.dag_prev_dict[idx]:
                    if self.is_full_buffer[pre]:
                        c = layer.nifm
                if k < layer.nofm and c < layer.nifm:
                    c = layer.nifm
                self.loop_block[idx] = [layer.wfil, layer.hfil, c, layer.wofm, H[idx], k, b]
                self.loop_order[idx] = \
                    schedule_generator.loop_order_generator(layer, self.loop_block[idx], irrelevant)

                self.tile_num = math.ceil(h_m * self.network.input_layer().nimg / (b * h))
        self.q = self.fused_weight_size * self.tile_num + self.fused_input_size + self.fused_output_size

        if mode == 1:
            p2 = self.resource.access_cost[2]
            p1 = self.resource.access_cost[1]

            q0, q1, q2 = p1[0], p1[1], p2[2] + p1[2]

            f_args = (q0, q1, q2, b, h_m)
            fun = self.fun(f_args)
            c_args = (b, h_m, self.idx_dict)
            con = self.con(c_args)
            x0 = [1 for _ in range(len(set(self.idx_dict.values())) + 1)]
            if b > 1:
                x0[0] = b
            else:
                x0[0] = h
            for idx in self.idx_dict:
                if idx < len(self.dag_vertex_list):
                    layer = self.network[self.dag_vertex_list[idx]]
                    if isinstance(layer, LocalRegionLayer):
                        continue
                    loop_lower_bound = self.loop_lower_bound(layer)
                    x0[self.idx_dict[idx]] = loop_lower_bound.k

            x0 = np.asarray(x0)
            res = minimize(fun, x0, method='COBYLA', constraints=con)

            if res.success:
                if b > 1:
                    b = math.ceil(res.x[0])
                    h = h_m
                else:
                    b = 1
                    h = math.ceil(res.x[0])
                H = [None for _ in self.dag_vertex_list]
                for l in reversed(self.dag_vertex_list):
                    idx = self.dag_vertex_dict[l]
                    layer = self.network[l]
                    if l in self.lasts:
                        H[idx] = int(self.scale[idx].s_h * h)
                        continue

                    h_tmp = None
                    for dst_idx in self.dag_next_dict[idx]:
                        dst = self.dag_vertex_list[dst_idx]
                        dst_layer = self.network[dst]
                        dst_h = H[dst_idx]
                        assert dst_h is not None
                        if isinstance(dst_layer, LocalRegionLayer):
                            hreg = dst_layer.hreg
                        else:
                            hreg = dst_layer.hfil
                        if h_tmp is None:
                            h_tmp = min((dst_h - 1) * dst_layer.hstd + hreg, layer.hofm)
                        else:
                            if (dst_h - 1) * dst_layer.hstd + hreg > h_tmp:
                                h_tmp = min((dst_h - 1) * dst_layer.hstd + hreg, layer.hofm)
                    H[idx] = math.floor(h_tmp)

                for l in self.dag_vertex_list:
                    idx = self.dag_vertex_dict[l]
                    layer = self.network[l]
                    if isinstance(layer, LocalRegionLayer):
                        continue

                    if idx in self.idx_dict:
                        k = res.x[self.idx_dict[idx]]
                    else:
                        k = layer.nofm

                    if self.dag_prev_dict[idx] and list(self.dag_prev_dict[idx])[0] in self.idx_dict:
                        c = self.idx_dict[list(self.dag_prev_dict[idx])[0]]
                    else:
                        c = layer.nifm
                    self.loop_block[idx] = \
                        [layer.wfil, layer.hfil, math.ceil(c), layer.wofm, H[idx], math.ceil(k), b]

                    self.loop_order[idx] = \
                        schedule_generator.loop_order_generator(layer, self.loop_block[idx], irrelevant)

            q = self.fused_weight_size * self.network.input_layer().nimg * h_m * q2 / (b * h) \
                + self.fused_input_size * p2[0] + self.fused_output_size * p2[1]
            for l in self.dag_vertex_list:
                idx = self.dag_vertex_dict[l]
                layer = self.network[l]
                if isinstance(layer, LocalRegionLayer):
                    continue
                q += (q1 * layer.nifm * layer.total_ofmap_size / self.loop_block[idx][2])
                q += (q0 * layer.nofm * layer.total_ifmap_size / self.loop_block[idx][5])
            self.q = q
            self.tile_num = math.ceil(h_m * self.network.input_layer().nimg / (b * h))

        return True

    def _init_alternate_pair(self, mode):
        if self.d_fusion or self.z_fusion:
            return self._init_alternate_pair_others(mode)
        else:
            return self._init_alternate_pair_optimus()

    def fun(self, args):

        q0, q1, q2, b, h_m = args
        expr = ''

        fidx = 1
        idx_dict = dict()
        if b > 1:
            p2 = q2 * self.fused_weight_size * self.network.input_layer().nimg
            expr += '+ {p2} / x[0] '.format(p2=p2)
        else:
            p2 = q2 * self.fused_weight_size * self.network.input_layer().nimg * h_m
            expr += '+ {p2} / x[0] '.format(p2=p2)
        for l in self.dag_vertex_list:
            idx = self.dag_vertex_dict[l]
            layer = self.network[l]

            if isinstance(layer, ConvLayer):
                assert len(self.dag_prev_dict[idx]) <= 1
                p0 = q0 * layer.total_ifmap_size * layer.nofm
                p1 = q1 * layer.total_ofmap_size * layer.nifm

                k = True if self.is_full_buffer[idx] else False
                c = False
                for pre in self.dag_prev_dict[idx]:
                    if self.is_full_buffer[pre]:
                        c = True
                if not k and not c:
                    c = True

                if not k:
                    if idx in idx_dict:
                        cur_fidx = idx_dict[idx]
                    else:
                        cur_fidx = fidx
                        idx_dict[idx] = fidx
                        fidx += 1
                    expr += '+ {p0} / x[{idx}] '.format(p0=p0, idx=cur_fidx)

                if not c:

                    if len(self.dag_prev_dict[idx]) == 1:
                        pidx = list(self.dag_prev_dict[idx])[0]
                        if pidx in idx_dict:
                            cur_fidx = idx_dict[pidx]
                        else:
                            cy_idx = pidx
                            while len(self.dag_prev_dict[cy_idx]) == 1:
                                if isinstance(self.network[self.dag_vertex_list[cy_idx]], ConvLayer):
                                    break
                                cy_idx = list(self.dag_prev_dict[cy_idx])[0]
                            if len(self.dag_prev_dict[cy_idx]) == 1:
                                if cy_idx in idx_dict:
                                    cur_fidx = idx_dict[cy_idx]
                                    idx_dict[pidx] = cur_fidx
                                else:
                                    cur_fidx = fidx
                                    idx_dict[cy_idx] = cur_fidx
                                    idx_dict[pidx] = cur_fidx
                                    fidx += 1
                            elif len(self.dag_prev_dict[cy_idx]) == 0:
                                continue

                            else:
                                cur_fidx = fidx
                                idx_dict[pidx] = cur_fidx
                                fidx += 1

                    else:
                        continue

                    expr += '+ {p1} / x[{idx}] '.format(p1=p1, idx=cur_fidx)

        self.idx_dict = idx_dict
        expr = expr[1:]
        v = lambda x: eval(expr)
        return v

    def con(self, args):
        b, h_m, idx_dict = args

        ineq_cons = []
        if b > 1:
            ineq_cons.append('x[0] - 1')
            ineq_cons.append('-x[0] + {nimg}'.format(nimg=self.network.input_layer().nimg))
        else:
            ineq_cons.append('x[0] - 1')
            ineq_cons.append('-x[0] + {hh}'.format(hh=h_m))

        ext_inputs = set(self.ext_inputs_dict.keys())
        ss = ''
        for l in self.dag_vertex_list:
            idx = self.dag_vertex_dict[l]
            layer = self.network[l]
            sca = self.scale[idx]
            minsize = self.minSize[idx]
            loop_lower_bound = self.loop_lower_bound(layer)

            if l in self.firsts:
                if not self.is_full_buffer[idx]:
                    for src in self.network.prevs(l):
                        if src in ext_inputs:
                            if src is None:
                                src_layer = self.network.input_layer()
                            else:
                                src_layer = self.network[src]
                            m_h = min((minsize.h - 1) * layer.hstd + layer.hfil, src_layer.hofm)
                            s_h = sca.s_h * layer.hstd
                            if b > 1:
                                ss += '+x[0]*{hh}*{w}*{k}'\
                                    .format(hh=layer.hifm, w=layer.wifm, k=src_layer.nofm)
                            else:
                                ss += '+({m_h}-{s_h}+{s_h}*x[0])*{w}*{k}' \
                                    .format(m_h=m_h, s_h=s_h, w=layer.wifm, k=src_layer.nofm)
                            ext_inputs.remove(src)

                if isinstance(layer, LocalRegionLayer) and self.is_full_buffer[idx]:
                    if b > 1:
                        ss += '+x[0]*{hh}*{w}*{k}'.format(hh=layer.hofm, w=layer.wofm, k=layer.nofm)
                    else:
                        ss += '+({m_h}-{s_h}+{s_h}*x[0])*{w}*{k}'\
                            .format(m_h=minsize.h, s_h=sca.s_h, w=layer.wofm, k=layer.nofm)

            if isinstance(layer, ConvLayer):
                if self.is_full_buffer[idx]:
                    if b > 1:
                        ss += '+x[0]*{hh}*{w}*{k}' \
                            .format(hh=layer.hofm, w=layer.wofm, k=layer.nofm)
                    else:
                        ss += '+({m_h}-{s_h}+{s_h}*x[0])*{w}*{k}' \
                            .format(m_h=minsize.h, s_h=sca.s_h, w=layer.wofm, k=layer.nofm)

                else:
                    cur_fidx = idx_dict[idx]
                    if b > 1:
                        ss += '+x[0]*{hh}*{w}*x[{idx}]' \
                            .format(hh=layer.hofm, w=layer.wofm, idx=cur_fidx)
                    else:
                        ss += '+({m_h}-{s_h}+{s_h}*x[0])*{w}*x[{idx}]' \
                            .format(m_h=minsize.h, s_h=sca.s_h, w=layer.wofm, idx=cur_fidx)

                    ineq_cons.append('x[{idx}] - {k}'.format(idx=cur_fidx, k=loop_lower_bound.k))
                    ineq_cons.append('-x[{idx}] + {nofm}'.format(idx=cur_fidx, nofm=layer.nofm))

        for src in ext_inputs:
            if src is None:
                src_layer = self.network.input_layer()
            else:
                src_layer = self.network[src]
            loop_lower_bound = self.loop_lower_bound(src_layer)
            pidx = self.ext_inputs_idx[src]
            if pidx in idx_dict:
                cur_fidx = idx_dict[pidx]
                ineq_cons.append('x[{pidx}] - {k}'.format(pidx=cur_fidx, k=loop_lower_bound.k))
                ineq_cons.append('-x[{pidx}] + {nofm}'.format(pidx=cur_fidx, nofm=src_layer.nofm))

        s = self.resource.buffer(1).capacity
        ss = '-(' + ss[1:] + ')+{}'.format(s)
        ineq_cons.append(ss)
        cons = ()
        for ineq in ineq_cons:
            cons += ({'type': 'ineq', 'fun': lambda x, ineq=ineq: eval(ineq)}, )

        cons_res = copy.copy(cons)
        return cons_res

    def _alternate(self):

        level = self.resource.buffer_levels() - 2
        s = self.resource.buffer(level).capacity * self.resource.paras[level].count

        min_feature_footprint_t = s - 0.0000001
        add_one_line_footprint_t = float('inf')

        is_full_buffer_t = None
        for start in [True, False]:
            is_full_buffer = [None for _ in range(len(self.dag_vertex_list))]
            min_feature_footprint = 0
            add_one_line_footprint = 0
            ext_inputs = set(self.ext_inputs_dict.keys())
            for l in self.dag_vertex_list:
                idx = self.dag_vertex_dict[l]
                layer = self.network[l]
                sca = self.scale[idx]
                minsize = self.minSize[idx]
                if l in self.firsts:
                    if self.womincost:
                        is_full_buffer[idx] = True
                    else:
                        if start:
                            is_full_buffer[idx] = True
                        else:
                            is_full_buffer[idx] = False

                    if not is_full_buffer[idx]:
                        for src in self.network.prevs(l):
                            if src in ext_inputs:
                                if src is None:
                                    src_layer = self.network.input_layer()
                                else:
                                    src_layer = self.network[src]

                                min_feature_footprint += \
                                    (src_layer.nofm
                                     * min(((minsize.h - 1) * layer.hstd + layer.hfil), src_layer.hofm)
                                     * layer.wifm)
                                add_one_line_footprint += \
                                    (src_layer.nofm
                                     * sca.s_h * layer.hstd
                                     * layer.wifm)
                                ext_inputs.remove(src)
                    if isinstance(layer, LocalRegionLayer) and is_full_buffer[idx]:
                        min_feature_footprint += layer.nofm * minsize.h * layer.wofm
                        add_one_line_footprint += layer.nofm * sca.s_h * layer.wofm

                for src_idx in self.dag_prev_dict[idx]:
                    assert is_full_buffer[src_idx] is not None
                    if self.womincost:
                        is_full_buffer[idx] = True
                    else:
                        if isinstance(layer, LocalRegionLayer):
                            if is_full_buffer[idx] is None:
                                is_full_buffer[idx] = is_full_buffer[src_idx]
                            else:
                                is_full_buffer[idx] \
                                    = is_full_buffer[idx] or is_full_buffer[src_idx]
                        else:
                            if not is_full_buffer[src_idx]:
                                is_full_buffer[idx] = True
                            else:
                                is_full_buffer[idx] = False

                if isinstance(layer, ConvLayer):
                    if is_full_buffer[idx]:
                        min_feature_footprint += layer.nofm * minsize.h * layer.wofm
                        add_one_line_footprint += layer.nofm * sca.s_h * layer.wofm

                    else:
                        loop_lower_bound = self.loop_lower_bound(layer)
                        k = min(max(loop_lower_bound.k, loop_lower_bound.c), layer.nofm)
                        min_feature_footprint += k * minsize.h * layer.wofm
                        add_one_line_footprint += k * sca.s_h * layer.wofm

            if (s - min_feature_footprint) > 0 \
                    and (add_one_line_footprint / (s - min_feature_footprint)) \
                    < (add_one_line_footprint_t / (s - min_feature_footprint_t)):
                min_feature_footprint_t = min_feature_footprint
                is_full_buffer_t = is_full_buffer
                add_one_line_footprint_t = add_one_line_footprint
        return min_feature_footprint_t, is_full_buffer_t, add_one_line_footprint_t

