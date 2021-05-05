
import math
import numpy as np

from operator import add
from functools import reduce
from scipy.optimize import minimize
from queue import Queue

from .interlayer import InterLayerReuse
from .layer import LocalRegionLayer, ConvLayer
from .network import Network
from .resource import Resource
from .mapping_point import MappingPoint

from . import loop_enum as le
from .cost_model import CostModel


class ScheduleGenerator(object):
    """
    Search optimal scheduling for neural networks.
    """

    def __init__(self, network, resource, cost_model, loop_lower_bound,
                 z_fusion=False, d_fusion=False, womincost=False, wofusion=False, is_shiDianNao=False):
        if not isinstance(network, Network):
            raise TypeError("ScheduleGenerator: network must be a Network instance.")
        if not isinstance(resource, Resource):
            raise TypeError("ScheduleGenerator: resource must be a Resource instance.")
        if not isinstance(cost_model, CostModel):
            raise TypeError("ScheduleGenerator: cost_model must be a CostModel instance.")

        self.network = network
        self.resource = resource
        # self.args = args
        self.loop_lower_bound = loop_lower_bound
        self.cost_model = cost_model
        self.z_fusion = z_fusion
        self.d_fusion = d_fusion
        self.womincost = womincost
        self.wofusion = wofusion
        self.is_shiDianNao = is_shiDianNao

    def schedule_search(self):

        if self.z_fusion or self.d_fusion or self.wofusion:
            dptable = dict()
            res_map_0, res_0 = self.others(dptable, 0)
            dptable = dict()
            res_map_1, res_1 = self.others(dptable, 1)
            res_map = [res_map_1, res_map_0]
            res = [res_1, res_0]

        else:
            dptable = dict()
            if len(self.network.firsts()) > 1:
                nx = sorted(list(self.network.firsts()))
                dsv = self._dsv(nx)
                res_map, res = self.hafs([dsv], dptable)
            else:
                res_map, res = self.hafs(list(self.network.firsts()), dptable)

        return res_map, res

    def hafs(self, fusion_group, dptable):

        if not isinstance(fusion_group, list):
            raise TypeError('HaFS: fusion_group must be a list.')

        g = []
        for layer in fusion_group:
            if layer in self.network:
                g.append(layer)

        dpkey = tuple(sorted(fusion_group))
        if dpkey in dptable:
            return dptable[dpkey]

        nx = self._next(fusion_group)
        if len(g) > 0:
            cost, loop_block, loop_order, vertex_list, sfil_fit = self._find_schedule(g)
            if cost == float("inf"):
                dptable[dpkey] = [[]], float("inf")
                return dptable[dpkey]

        if len(nx) == 0:
            cost, loop_block, loop_order, vertex_list, sfil_fit = self._find_schedule(g)
            dptable[dpkey] = [[vertex_list, loop_block, loop_order, cost, sfil_fit]], cost
            return dptable[dpkey]

        optimal_s, min_cost = [[]], float("inf")

        masked = self._reachable(nx)
        for c in sorted(nx):
            if masked.get(c, False):
                continue
            fuse_node = fusion_group + [c]
            s, cost = self.hafs(fuse_node, dptable)
            if cost < min_cost:
                min_cost = cost
                optimal_s = s

        if not self._is_dsv(fusion_group):
            if len(nx) == 1:
                dsv = nx[0]
            else:
                dsv = self._dsv(nx)
            s, cost = self.hafs([dsv], dptable)
            g_cost, loop_block, loop_order, vertex_list, sfil_fit = self._find_schedule(g)
            if cost + g_cost < min_cost:
                min_cost = cost + g_cost
                optimal_s = [[vertex_list, loop_block, loop_order, g_cost, sfil_fit]] + s

        dptable[dpkey] = optimal_s, min_cost
        return optimal_s, min_cost

    def others(self, dptable, mode):
        q = Queue(maxsize=0)
        for layer in self.network.firsts():
            q.put(layer)
        L = []
        visit = dict()
        while not q.empty():
            layer = q.get()
            L.append(layer)

            for nx in reversed(self.network.nexts(layer)):
                if nx is None:
                    continue
                if nx not in visit:
                    visit[nx] = 1
                else:
                    visit[nx] += 1
                if len(self.network.prevs(nx)) == visit[nx]:
                    q.put(nx)

        res_map, res = self.others_m([L[0]], dptable, L, 1, mode)
        return res_map, res

    def others_m(self, fusion_group, dptable, L, idx, mode):
        if not isinstance(fusion_group, list):
            raise TypeError('l78z_m: fusion_group must be a list.')

        dpkey = tuple(fusion_group)
        if dpkey in dptable:
            return dptable[dpkey]

        if idx == len(L):
            cost, loop_block, loop_order, vertex_list, sfil_fit = self._find_schedule(fusion_group, mode)
            dptable[dpkey] = [[vertex_list, loop_block, loop_order, cost, sfil_fit]], cost
            return dptable[dpkey]

        optimal_s, min_cost = [[]], float("inf")

        fuse_node = fusion_group + [L[idx]]
        s, cost = self.others_m(fuse_node, dptable, L, idx+1, mode)
        if cost < min_cost:
            min_cost = cost
            optimal_s = s

        s, cost = self.others_m([L[idx]], dptable, L, idx+1, mode)
        g_cost, loop_block, loop_order, vertex_list, sfil_fit = self._find_schedule(fusion_group, mode)
        if cost + g_cost < min_cost:
            min_cost = cost + g_cost
            optimal_s = [[vertex_list, loop_block, loop_order, g_cost, sfil_fit]] + s

        dptable[dpkey] = optimal_s, min_cost
        return optimal_s, min_cost

    def _next(self, fusion_group):

        nexts = set()
        for layer in fusion_group:
            if layer not in self.network:
                for nx in layer.split("|")[1:]:
                    if nx not in fusion_group and nx:
                        nexts.add(nx)
            else:
                for nx in self.network.nexts(layer):
                    if nx not in fusion_group and nx:
                        nexts.add(nx)

        return list(nexts)

    def _is_dsv(self, fusion_group):

        return len(fusion_group) == 1 and fusion_group[0] not in self.network

    @ staticmethod
    def _dsv(nx):
        dsv = 'dsv'
        for n in sorted(nx):
            dsv += ('|' + n)

        return dsv

    def _reachable(self, child):
        masked = dict()

        def dfs(v):
            masked[v] = True
            for vc in self.network.nexts(v):
                if not masked.get(vc, False):
                    dfs(vc)

        for rc in child:
            if not masked.get(rc, False):
                for nn in self.network.nexts(rc):
                    dfs(nn)

        return masked

    def _find_uni_layer_schedule_optimus(self, g):
        layer = self.network[g]
        cost, loop_block, loop_order, is_filter_fit = float('inf'), [None], [None], False

        if isinstance(layer, LocalRegionLayer):
            cost = 0
            return cost, loop_block, loop_order, is_filter_fit

        level = self.resource.buffer_levels() - 2
        s = self.resource.buffer(level).capacity * self.resource.paras[level].count
        if s > layer.total_filter_size:
            is_filter_fit = True

        loop_lower_bound = self.loop_lower_bound(layer)

        if self.network.net_name == "SqueezeNet":
            for scheduling in _unilayer_schedule_list_v1:
                cost_t, loop_block_t, loop_order_t = scheduling(layer, s, loop_lower_bound)
                if cost_t < cost:
                    cost, loop_block, loop_order = cost_t, loop_block_t, loop_order_t
        else:
            for scheduling in _unilayer_schedule_list_v2:
                cost_t, loop_block_t, loop_order_t, _, _ \
                    = scheduling(layer, self.resource, loop_lower_bound)
                if cost_t < cost:
                    cost, loop_block, loop_order = cost_t, loop_block_t, loop_order_t

        return cost, [loop_block], [loop_order], is_filter_fit

    def _find_uni_layer_schedule_others(self, g, mode):
        layer = self.network[g]
        cost, loop_block, loop_order, is_filter_fit = float('inf'), [None], [None], False

        if isinstance(layer, LocalRegionLayer):
            cost = 0
            return cost, loop_block, loop_order, is_filter_fit

        level = self.resource.buffer_levels() - 2
        s = self.resource.buffer(level).capacity * self.resource.paras[level].count
        if s > layer.total_filter_size:
            is_filter_fit = True

        loop_lower_bound = self.loop_lower_bound(layer)

        if self.is_shiDianNao:
            if mode == 0:
                cost, loop_block, loop_order = _filterr_v1(layer, s, loop_lower_bound)
            else:
                cost, loop_block, loop_order, _, _ \
                        = _filterr_v2(layer, self.resource, loop_lower_bound)
        else:
            if mode == 0:
                cost, loop_block, loop_order = _psumsr_v1(layer, s, loop_lower_bound)
            else:
                cost, loop_block, loop_order, _, _ \
                        = _psumsr_v2(layer, self.resource, loop_lower_bound)

        return cost, [loop_block], [loop_order], is_filter_fit

    def _find_uni_layer_schedule(self, g, mode=0):
        if self.d_fusion or self.z_fusion or self.wofusion:
            return self._find_uni_layer_schedule_others(g, mode)
        else:
            return self._find_uni_layer_schedule_optimus(g)

    def _find_multi_layer_schedule(self, fusion_group, mode):

        ilr = InterLayerReuse(self.network, fusion_group, self.resource, self.loop_lower_bound,
                              z_fusion=self.z_fusion, d_fusion=self.d_fusion, womincost=self.womincost)
        ilr.sched(mode)
        if not ilr.valid or self.wofusion:
            return float('inf'), None, None, None, False
        else:
            cost, loop_block, loop_order, vertex_list = ilr.q, ilr.loop_block, ilr.loop_order, ilr.dag_vertex_list

        return cost, loop_block, loop_order, vertex_list, ilr.sfil_fit

    def _find_schedule(self, g, mode=0):

        if len(g) == 1:
            cost, loop_block, loop_order, sfil_fit = self._find_uni_layer_schedule(g[0], mode)
            vertex_list = g
        else:
            cost, loop_block, loop_order, vertex_list, sfil_fit = self._find_multi_layer_schedule(g, mode)

        return cost, loop_block, loop_order, vertex_list, sfil_fit

    def mapping(self, g, loop_block_g, loop_order_g):

        cost_inner_g = [None for _ in g]
        point_g = [None for _ in g]
        t = 0
        for layer_name, loop_block, loop_order in zip(g, loop_block_g, loop_order_g):
            layer = self.network[layer_name]
            if not isinstance(layer, ConvLayer):
                cost_inner_g[t], point_g[t] = 0, None
                t += 1
                continue

            cost_inner, loop_block_n, loop_order_n = float('inf'), None, None
            sublayer = ConvLayer(loop_block[le.C], loop_block[le.K], [loop_block[le.H], loop_block[le.W]],
                                 sfil=[loop_block[le.R], loop_block[le.D]],
                                 strd=[layer.hstd, layer.wstd], nimg=loop_block[le.B])
            loop_lower_bound = self.loop_lower_bound(sublayer)

            s = self.resource.buffer(0).capacity
            actual_s = s * self.resource.paras[0].count
            point_t, blocking_t, ordering_t, partition_t = None, None, None, None
            for scheduling in _unilayer_schedule_list_v1:
                cost_t, loop_block_t, loop_order_t = \
                    scheduling(sublayer, actual_s, loop_lower_bound)

                if cost_t < float('inf'):
                    blocking_t = [loop_block_t, loop_block, layer.dimension]

                    loop_order_innermost = [le.NUM - 1] * le.NUM
                    non_max_block = [i for i, e in enumerate(loop_block_t) if (e != 1)]
                    order = 0
                    for i in non_max_block:
                        loop_order_innermost[i] = order
                        order += 1

                    ordering_t = [loop_order_innermost, loop_order_t, loop_order]

                    # TODO support unroll in other levels
                    innermost_partition = [1] * le.NUM
                    innermost_para_dim = []
                    for i, lp in enumerate(self.loop_lower_bound.unroll_loop):
                        innermost_para_dim.append([lp])
                        partition_size = self.loop_lower_bound.loop_lower_bound_init[lp]
                        if loop_block_t[lp] >= partition_size:
                            innermost_partition[lp] = partition_size
                        else:
                            innermost_partition[lp] = loop_block_t[lp]
                            if self.resource.replication and self.loop_lower_bound.replication_loop is not None:
                                para = partition_size // loop_block_t[lp]
                                replp = self.loop_lower_bound.replication_loop[i]
                                if para > 1 and loop_block_t[replp] > 1:
                                    innermost_para_dim[i].append(replp)
                                    innermost_partition[replp] = para \
                                        if loop_block_t[replp] > para else loop_block_t[replp]

                    para_loop_dim_list = [innermost_para_dim, [], []]
                    partition_t = [innermost_partition, [1] * le.NUM, [1] * le.NUM]

                    point_t = MappingPoint(list(zip(*ordering_t)), list(zip(*blocking_t)),
                                           list(zip(*partition_t)), para_loop_dim_list)
                    buffer_access = self.cost_model.get_level_access([point_t], [layer_name], 1)
                    cost_t = math.ceil(reduce(add, buffer_access, 0))

                if cost_t < cost_inner:
                    point_g[t] = point_t
                    cost_inner_g[t] = cost_t
                    cost_inner = cost_t

            t += 1

        return cost_inner_g, point_g


def _bounded_factor(n, start, end):
    # TODO: start bound
    f = []
    for i in range(1, min(int(n ** 0.5) + 1, end + 1)):
        if n % i == 0 and n // i <= end and i >= start and (n // i) >= start:
            f.__iadd__([i, int(n // i)])
        elif n % i == 0 and n // i <= end and (n // i) >= start:
            f.__iadd__([int(n // i)])
        elif n % i == 0 and i >= start:
            f.__iadd__([i])

    return set(f)


def _bhw_factorization(layer, n, loop_lower_bound):
    b = h = w = 0
    while n > 0 and b * h * w == 0:
        for bb in _bounded_factor(n, loop_lower_bound.b, layer.nimg):
            for ww in _bounded_factor(n / bb, loop_lower_bound.w, layer.wofm):
                if n / (bb * ww) <= layer.hofm:
                    b, w, h = bb, ww, n / (bb * ww)
                    break
                if b * h * w > 0:
                    break
            if b * h * w > 0:
                break
        n -= 1
    n += 1
    return int(n), int(b), int(h), int(w)


def loop_order_generator(layer, loop_block, irrelevant):
    loop_order = [le.NUM - 1] * le.NUM

    order = 0
    for i in irrelevant:
        if loop_block[i] < layer.dimension[i]:
            loop_order[i] = order
            order += 1
    non_max_block = [i for i, e in enumerate(loop_block) if (e < layer.dimension[i] and i not in irrelevant)]
    for i in non_max_block:
        loop_order[i] = order
        order += 1

    return loop_order


def blocking_upper_bound(layer, block, q, resource, loop_lower_bound):

    k, c, bhw, r, d = block
    q0, q1, q2 = q
    s = resource.buffer(1).capacity

    bhw_upper_bound = layer.nimg * layer.hofm * layer.wofm
    bhw_lower_bound = loop_lower_bound.b * loop_lower_bound.h * loop_lower_bound.w

    if k > layer.nofm and c > layer.nifm and bhw > bhw_upper_bound:
        k = layer.nofm
        c = layer.nifm
        bhw = bhw_upper_bound
        return math.floor(k), math.floor(c), math.floor(bhw)
    elif k > layer.nofm and c > layer.nifm:
        k = layer.nofm
        c = layer.nifm
        bhw = min((s - r*d*c*k)/(c*layer.hstd*layer.wstd + k), bhw_upper_bound)
        return math.floor(k), math.floor(c), math.floor(bhw)
    elif k > layer.nofm and bhw > bhw_upper_bound:
        k = layer.nofm
        bhw = bhw_upper_bound
        c = min((s - k*bhw)/(bhw*layer.hstd*layer.wstd + r*d*k), layer.nifm)
        return math.floor(k), math.floor(c), math.floor(bhw)
    elif c > layer.nifm and bhw > bhw_upper_bound:
        c = layer.nifm
        bhw = bhw_upper_bound
        k = min((s - bhw*c*layer.hstd*layer.wstd)/(bhw + r*d*c), layer.nofm)
        return math.floor(k), math.floor(c), math.floor(bhw)
    elif k > layer.nofm:
        k = layer.nofm
        a = (0.5 / (layer.hstd * layer.wstd)) * (k + r * d * k * q1 / q2)
        if loop_lower_bound.c > 1:
            c = max(min(math.sqrt(s * q1 / (q2 * layer.hstd * layer.wstd) + a ** 2) - a, layer.nifm),
                    loop_lower_bound.c)
            bhw = min((s - r * d * c * k) / (c * layer.hstd * layer.wstd + k), bhw_upper_bound)
        else:
            bhw = max(min(q2*(math.sqrt(s * q1 / (q2 * layer.hstd * layer.wstd) + a ** 2) - a)/q1, bhw_upper_bound),
                      bhw_lower_bound)
            c = min((s - k*bhw)/(bhw*layer.hstd*layer.wstd + r*d*k), layer.nifm)

        return math.floor(k), math.floor(c), math.floor(bhw)

    elif c > layer.nifm:
        c = layer.nifm
        a = 0.5 * (c * layer.hstd * layer.wstd + r * d * c * q0 / q2)

        if loop_lower_bound.k > 1:
            k = max(min(math.sqrt(q0 * s / q2 + a ** 2) - a, layer.nofm), loop_lower_bound.k)
            bhw = min((s - r * d * c * k) / (c * layer.hstd * layer.wstd + k), bhw_upper_bound)
        else:
            bhw = max(min(q2*(math.sqrt(q0 * s / q2 + a ** 2) - a)/q0, bhw_upper_bound), bhw_lower_bound)
            k = min((s - bhw * c * layer.hstd * layer.wstd) / (bhw + r * d * c), layer.nofm)

        return math.floor(k), math.floor(c), math.floor(bhw)

    elif bhw > bhw_upper_bound:
        bhw = bhw_upper_bound
        a = (0.5/(r*d)) * (bhw*layer.hstd*layer.wstd + q0*bhw/q1)

        if loop_lower_bound.k > 1:
            k = max(min(math.sqrt(s * q0 / (q1 * r * d) + a ** 2) - a, layer.nofm), loop_lower_bound.k)
            c = min((s - k * bhw) / (bhw * layer.hstd * layer.wstd + r * d * k), layer.nifm)
        else:
            c = max(min(q1*(math.sqrt(s * q0 / (q1 * r * d) + a ** 2) - a)/q0, layer.nifm), loop_lower_bound.c)
            k = min((s - bhw * c * layer.hstd * layer.wstd) / (bhw + r * d * c), layer.nofm)
        return math.floor(k), math.floor(c), math.floor(bhw)

    return math.floor(k), math.floor(c), math.floor(bhw)


def blocking_lower_bound(layer, block, q, resource, loop_lower_bound):

    k, c, bhw, r, d = block
    q0, q1, q2 = q
    s = resource.buffer(1).capacity

    bhw_lower_bound = loop_lower_bound.b * loop_lower_bound.h * loop_lower_bound.w
    if k < loop_lower_bound.k and c < loop_lower_bound.c and bhw < bhw_lower_bound:
        return k, c, bhw
    elif k < loop_lower_bound.k and c < loop_lower_bound.c:
        k = loop_lower_bound.k
        c = loop_lower_bound.c
        bhw = (s - r*d*c*k)/(c*layer.hstd*layer.wstd + k)
        return k, c, bhw
    elif k < loop_lower_bound.k and bhw < bhw_lower_bound:
        k = loop_lower_bound.k
        bhw = bhw_lower_bound
        c = (s - k*bhw)/(bhw*layer.hstd*layer.wstd + r*d*k)
        return k, c, bhw
    elif c < loop_lower_bound.c and bhw < bhw_lower_bound:
        c = loop_lower_bound.c
        bhw = bhw_lower_bound
        k = (s - bhw*c*layer.hstd*layer.wstd)/(bhw + r*d*c)
        return k, c, bhw
    elif k < loop_lower_bound.k:
        k = loop_lower_bound.k
        a = (0.5 / (layer.hstd * layer.wstd)) * (k + r * d * k * q1 / q2)
        if loop_lower_bound.c > 1:
            c = max(math.sqrt(s * q1 / (q2 * layer.hstd * layer.wstd) + a ** 2) - a, loop_lower_bound.c)
            bhw = (s - r * d * c * k) / (c * layer.hstd * layer.wstd + k)
        else:
            bhw = max(q2*(math.sqrt(s * q1 / (q2 * layer.hstd * layer.wstd) + a ** 2) - a)/q1, bhw_lower_bound)
            c = (s - k*bhw)/(bhw*layer.hstd*layer.wstd + r*d*k)
        return k, c, bhw
    elif c < loop_lower_bound.c:
        c = loop_lower_bound.c
        a = 0.5 * (c * layer.hstd * layer.wstd + r * d * c * q0 / q2)

        if loop_lower_bound.k > 1:
            k = max(math.sqrt(q0 * s / q2 + a ** 2) - a, loop_lower_bound.k)
            bhw = (s - r * d * c * k) / (c * layer.hstd * layer.wstd + k)
        else:
            bhw = max(q2*(math.sqrt(q0 * s / q2 + a ** 2) - a)/q0, bhw_lower_bound)
            k = (s - bhw * c * layer.hstd * layer.wstd) / (bhw + r * d * c)
        return k, c, bhw
    elif bhw < bhw_lower_bound:
        bhw = bhw_lower_bound
        a = (0.5/(r*d)) * (bhw*layer.hstd*layer.wstd + q0*bhw/q1)
        k = math.sqrt(s*q0/(q1*r*d) + a**2) - a
        c = q1*k / q0

        return k, c, bhw
    return k, c, bhw


def _psumsr_v2(layer, resource, loop_lower_bound):
    irrelevant = [le.C]

    p2 = resource.access_cost[2]
    p1 = resource.access_cost[1]
    r = layer.hfil
    d = layer.wfil
    q0 = (p2[0] + p1[0]) * layer.hstd * layer.wstd / (r * d)
    q1 = 2 * p1[1] / (r * d)
    q2 = (p2[2] + p1[2])
    s = resource.buffer(1).capacity

    bhw_lower_bound = loop_lower_bound.b * loop_lower_bound.h * loop_lower_bound.w
    bhw_upper_bound = layer.nimg * layer.hofm * layer.wofm
    args = (q0, q1, q2)
    cons_args = (loop_lower_bound.k, layer.nofm,
                 loop_lower_bound.c, layer.nifm,
                 bhw_lower_bound, bhw_upper_bound,
                 layer.hfil*layer.wfil, 1, layer.hstd*layer.wstd, s)
    # constrain
    cons = con(cons_args)

    # init
    q = min([q0, q1, q2])/2
    x0 = np.asarray((q0/q, q1/q, q2/q))

    # optimize
    res = minimize(fun(args), x0, method='SLSQP', constraints=cons)

    k, c, bhw = res.x
    k, c, bhw = math.floor(k), math.floor(c), math.floor(bhw)
    bhw, b, h, w = _bhw_factorization(layer, bhw, loop_lower_bound)
    loop_block = [d, r, c, w, h, k, b]
    loop_order = [le.NUM - 1] * le.NUM
    if not res.success or any(i <= 0 for i in loop_block):
        return float('inf'), loop_block, loop_order, \
               [float('inf'), float('inf'), float('inf')], [float('inf'), float('inf'), float('inf')]
    else:
        loop_order = loop_order_generator(layer, loop_block, irrelevant)

    q = (q0 / k + q1 / c + q2 / bhw) * layer.total_ops + (p2[1]-p1[1]) * layer.total_ofmap_size

    if_glb_access_cost = p1[0] * layer.hstd * layer.wstd * layer.total_ops / (r * d * k)
    of_glb_access_cost = 2 * p1[1] * layer.total_ops / (r * d * c) - p1[1] * layer.total_ofmap_size
    fi_glb_access_cost = p1[2] * layer.total_ops / bhw
    if_dram_access_cost = p2[0] * layer.hstd * layer.wstd * layer.total_ops / (r * d * k)
    of_dram_access_cost = p2[1] * layer.total_ofmap_size
    fi_dram_access_cost = p2[2] * layer.total_ops / bhw
    glb_access_cost = [if_glb_access_cost, of_glb_access_cost, fi_glb_access_cost]
    dram_access_cost = [if_dram_access_cost, of_dram_access_cost, fi_dram_access_cost]

    return q, loop_block, loop_order, glb_access_cost, dram_access_cost


def _filterr_v2(layer, resource, loop_lower_bound):
    irrelevant = [le.W, le.H, le.B]

    p2 = resource.access_cost[2]
    p1 = resource.access_cost[1]
    r = layer.hfil
    d = layer.wfil
    q0 = (p2[0] + p1[0]) * layer.hstd * layer.wstd / (r * d)
    q1 = 2 * (p2[1] + p1[1]) / (r * d)
    q2 = p1[2]
    s = resource.buffer(1).capacity

    bhw_lower_bound = loop_lower_bound.b * loop_lower_bound.h * loop_lower_bound.w
    bhw_upper_bound = layer.nimg * layer.hofm * layer.wofm
    args = (q0, q1, q2)
    cons_args = (loop_lower_bound.k, layer.nofm,
                 loop_lower_bound.c, layer.nifm,
                 bhw_lower_bound, bhw_upper_bound,
                 r*d, 1, layer.hstd*layer.wstd, s)
    # constrain
    cons = con(cons_args)

    # init
    q = min([q0, q1, q2])/2
    x0 = np.asarray((q0/q, q1/q, q2/q))

    # optimize
    res = minimize(fun(args), x0, method='SLSQP', constraints=cons)

    k, c, bhw = res.x
    k, c, bhw = math.floor(k), math.floor(c), math.floor(bhw)
    bhw, b, h, w = _bhw_factorization(layer, bhw, loop_lower_bound)
    loop_block = [d, r, c, w, h, k, b]
    loop_order = [le.NUM - 1] * le.NUM
    if not res.success or any(i <= 0 for i in loop_block):
        return float('inf'), loop_block, loop_order, \
               [float('inf'), float('inf'), float('inf')], [float('inf'), float('inf'), float('inf')]
    else:
        loop_order = loop_order_generator(layer, loop_block, irrelevant)

    q = (q0 / k + q1 / c + q2 / bhw) * layer.total_ops + p2[2] * layer.total_filter_size \
        - (p2[1] + p1[1]) * layer.total_ofmap_size

    if_glb_access_cost = p1[0] * layer.hstd * layer.wstd * layer.total_ops / (r * d * k)
    of_glb_access_cost = 2 * p1[1] * layer.total_ops / (r * d * c) - p1[1] * layer.total_ofmap_size
    fi_glb_access_cost = p1[2] * layer.total_ops / bhw
    if_dram_access_cost = p2[0] * layer.hstd * layer.wstd * layer.total_ops / (r * d * k)
    of_dram_access_cost = 2 * p2[1] * layer.total_ops / (r * d * c) - p2[1] * layer.total_ofmap_size
    fi_dram_access_cost = p2[2] * layer.total_filter_size
    glb_access_cost = [if_glb_access_cost, of_glb_access_cost, fi_glb_access_cost]
    dram_access_cost = [if_dram_access_cost, of_dram_access_cost, fi_dram_access_cost]

    return q, loop_block, loop_order, glb_access_cost, dram_access_cost


def _ifmapr_v2(layer, resource, loop_lower_bound):
    # irrelevant loop: k
    irrelevant = [le.K]

    p2 = resource.access_cost[2]
    p1 = resource.access_cost[1]
    r = layer.hfil
    d = layer.wfil
    q0 = p1[0] * layer.hstd * layer.wstd / (r * d)
    q1 = 2 * (p2[1] + p1[1]) / (r * d)
    q2 = p2[2] + p1[2]
    s = resource.buffer(1).capacity

    bhw_lower_bound = loop_lower_bound.b * loop_lower_bound.h * loop_lower_bound.w
    bhw_upper_bound = layer.nimg * layer.hofm * layer.wofm
    args = (q0, q1, q2)
    cons_args = (loop_lower_bound.k, layer.nofm,
                 loop_lower_bound.c, layer.nifm,
                 bhw_lower_bound, bhw_upper_bound,
                 r*d, 1, layer.hstd*layer.wstd, s)
    # constrain
    cons = con(cons_args)

    # init
    q = min([q0, q1, q2])/2
    x0 = np.asarray((q0/q, q1/q, q2/q))

    # optimize
    res = minimize(fun(args), x0, method='SLSQP', constraints=cons)

    k, c, bhw = res.x
    k, c, bhw = math.floor(k), math.floor(c), math.floor(bhw)
    bhw, b, h, w = _bhw_factorization(layer, bhw, loop_lower_bound)
    loop_block = [d, r, c, w, h, k, b]
    loop_order = [le.NUM - 1] * le.NUM
    if not res.success or any(i <= 0 for i in loop_block):
        return float('inf'), loop_block, loop_order, \
               [float('inf'), float('inf'), float('inf')], [float('inf'), float('inf'), float('inf')]
    else:
        loop_order = loop_order_generator(layer, loop_block, irrelevant)

    q = (q0 / k + q1 / c + q2 / bhw) * layer.total_ops + p2[0] * layer.total_ifmap_size \
        - (p2[1] + p1[1]) * layer.total_ofmap_size

    if_glb_access_cost = p1[0] * layer.hstd * layer.wstd * layer.total_ops / (r * d * k)
    of_glb_access_cost = 2 * p1[1] * layer.total_ops / (r * d * c) - p1[1] * layer.total_ofmap_size
    fi_glb_access_cost = p1[2] * layer.total_ops / bhw
    if_dram_access_cost = p2[0] * layer.total_ifmap_size
    of_dram_access_cost = 2 * p2[1] * layer.total_ops / (r * d * c) - p2[1] * layer.total_ofmap_size
    fi_dram_access_cost = p2[2] * layer.total_ops / bhw
    glb_access_cost = [if_glb_access_cost, of_glb_access_cost, fi_glb_access_cost]
    dram_access_cost = [if_dram_access_cost, of_dram_access_cost, fi_dram_access_cost]

    return q, loop_block, loop_order, glb_access_cost, dram_access_cost


def _cwr_c_v2(layer, resource, loop_lower_bound):
    # irrelevant loop: k r d
    irrelevant = [le.D, le.R, le.K]
    p2 = resource.access_cost[2]
    p1 = resource.access_cost[1]
    r = layer.hfil
    d = layer.wfil
    q0 = p1[0] * layer.hstd * layer.wstd / (r * d)
    q1 = 2 * p1[1] / (r * d)
    q2 = p1[2] + p2[2]

    s = resource.buffer(1).capacity

    bhw_lower_bound = loop_lower_bound.b * loop_lower_bound.h * loop_lower_bound.w
    bhw_upper_bound = layer.nimg * layer.hofm * layer.wofm
    args = (q0, q1, q2)
    cons_args = (loop_lower_bound.k, layer.nofm,
                 layer.nifm, layer.nifm,
                 bhw_lower_bound, bhw_upper_bound,
                 r*d, 1, layer.hstd*layer.wstd, s)
    # constrain
    cons = con(cons_args)

    # init
    q = min([q0, q1, q2])/2
    x0 = np.asarray((q0/q, q1/q, q2/q))

    # optimize
    res = minimize(fun(args), x0, method='SLSQP', constraints=cons)

    k, c, bhw = res.x
    k, c, bhw = math.floor(k), math.floor(c), math.floor(bhw)
    bhw, b, h, w = _bhw_factorization(layer, bhw, loop_lower_bound)
    loop_block = [d, r, c, w, h, k, b]
    loop_order = [le.NUM - 1] * le.NUM
    if not res.success or any(i <= 0 for i in loop_block):
        return float('inf'), loop_block, loop_order, \
               [float('inf'), float('inf'), float('inf')], [float('inf'), float('inf'), float('inf')]
    else:
        loop_order = loop_order_generator(layer, loop_block, irrelevant)

    q = (q0 / k + q1 / c + q2 / bhw) * layer.total_ops \
        + p2[0] * layer.total_ifmap_size + (p2[1] - p1[1]) * layer.total_ofmap_size

    if_glb_access_cost = p1[0] * layer.hstd * layer.wstd * layer.total_ops / (r * d * k)
    of_glb_access_cost = 2 * p1[1] * layer.total_ops / (r * d * c) - p1[1] * layer.total_ofmap_size
    fi_glb_access_cost = p1[2] * layer.total_ops / bhw
    if_dram_access_cost = p2[0] * layer.total_ifmap_size
    of_dram_access_cost = p2[1] * layer.total_ofmap_size
    fi_dram_access_cost = p2[2] * layer.total_ops / bhw
    glb_access_cost = [if_glb_access_cost, of_glb_access_cost, fi_glb_access_cost]
    dram_access_cost = [if_dram_access_cost, of_dram_access_cost, fi_dram_access_cost]

    return q, loop_block, loop_order, glb_access_cost, dram_access_cost


def _cwr_k_v2(layer, resource, loop_lower_bound):
    # irrelevant loop: c r d
    irrelevant = [le.D, le.R, le.C]
    p2 = resource.access_cost[2]
    p1 = resource.access_cost[1]
    r = layer.hfil
    d = layer.wfil
    q0 = p1[0] * layer.hstd * layer.wstd / (r * d)
    q1 = 2 * p1[1] / (r * d)
    q2 = p1[2] + p2[2]

    s = resource.buffer(1).capacity

    bhw_lower_bound = loop_lower_bound.b * loop_lower_bound.h * loop_lower_bound.w
    bhw_upper_bound = layer.nimg * layer.hofm * layer.wofm
    args = (q0, q1, q2)
    cons_args = (layer.nofm, layer.nofm,
                 loop_lower_bound.c, layer.nifm,
                 bhw_lower_bound, bhw_upper_bound,
                 r*d, 1, layer.hstd*layer.wstd, s)
    # constrain
    cons = con(cons_args)

    # init
    q = min([q0, q1, q2])/2
    x0 = np.asarray((q0/q, q1/q, q2/q))

    # optimize
    res = minimize(fun(args), x0, method='SLSQP', constraints=cons)

    k, c, bhw = res.x
    k, c, bhw = math.floor(k), math.floor(c), math.floor(bhw)
    bhw, b, h, w = _bhw_factorization(layer, bhw, loop_lower_bound)
    loop_block = [d, r, c, w, h, k, b]
    loop_order = [le.NUM - 1] * le.NUM
    if not res.success or any(i <= 0 for i in loop_block):
        return float('inf'), loop_block, loop_order, \
               [float('inf'), float('inf'), float('inf')], [float('inf'), float('inf'), float('inf')]
    else:
        loop_order = loop_order_generator(layer, loop_block, irrelevant)

    q = (q0 / k + q1 / c + q2 / bhw) * layer.total_ops \
        + p2[0] * layer.total_ifmap_size + (p2[1] - p1[1]) * layer.total_ofmap_size

    if_glb_access_cost = p1[0] * layer.hstd * layer.wstd * layer.total_ops / (r * d * k)
    of_glb_access_cost = 2 * p1[1] * layer.total_ops / (r * d * c) - p1[1] * layer.total_ofmap_size
    fi_glb_access_cost = p1[2] * layer.total_ops / bhw
    if_dram_access_cost = p2[0] * layer.total_ifmap_size
    of_dram_access_cost = p2[1] * layer.total_ofmap_size
    fi_dram_access_cost = p2[2] * layer.total_ops / bhw
    glb_access_cost = [if_glb_access_cost, of_glb_access_cost, fi_glb_access_cost]
    dram_access_cost = [if_dram_access_cost, of_dram_access_cost, fi_dram_access_cost]

    return q, loop_block, loop_order, glb_access_cost, dram_access_cost


def _psumsr_v1(layer, capacity, loop_lower_bound):

    # irrelevant loop: c
    irrelevant = [le.C]
    c = loop_lower_bound.c
    r = layer.hfil
    d = layer.wfil
    a = layer.hstd * layer.wstd * (c + 1) / 2
    f = r * d / (layer.hstd * layer.wstd)
    s = capacity

    k = max(math.sqrt(s/f + a**2) - a, 1)
    bhw = math.floor(k * r * d / a)
    k = math.floor(k)

    # upper bound
    if k > layer.nofm and bhw > layer.nimg * layer.hofm * layer.wofm:
        k = layer.nofm
        bhw = layer.nimg * layer.hofm * layer.wofm
    elif k > layer.nofm:
        k = layer.nofm
        bhw = min(math.floor((s-r*d*c*k)/(k+c*layer.hstd*layer.wstd)), layer.nimg * layer.hofm * layer.wofm)
    elif bhw > layer.nimg * layer.hofm * layer.wofm:
        bhw = layer.nimg * layer.hofm * layer.wofm
        k = min(math.floor((s-bhw*c*layer.hstd*layer.wstd)/(bhw+r*d*c)), layer.nofm)

    # lower bound
    if k * bhw > loop_lower_bound.k * loop_lower_bound.b * loop_lower_bound.h * loop_lower_bound.w:
        if k < loop_lower_bound.k and bhw > loop_lower_bound.b * loop_lower_bound.h * loop_lower_bound.w:
            k = loop_lower_bound.k
            bhw = min(math.floor((s - r * d * c * k) / (k + c * layer.hstd * layer.wstd)),
                      layer.nimg * layer.hofm * layer.wofm)
        elif k > loop_lower_bound.k and bhw < loop_lower_bound.b * loop_lower_bound.h * loop_lower_bound.w:
            bhw = loop_lower_bound.b * loop_lower_bound.h * loop_lower_bound.w
            k = min(math.floor((s - bhw * c * layer.hstd * layer.wstd) / (bhw + r * d * c)), layer.nofm)

    bhw, b, h, w = _bhw_factorization(layer, bhw, loop_lower_bound)

    c = min(math.floor((s - bhw*k) / (bhw*layer.hstd*layer.wstd + r*d*k)), layer.nifm)

    loop_block = [d, r, c, w, h, k, b]
    loop_order = [le.NUM - 1] * le.NUM
    if any(i <= 0 for i in loop_block):
        return float('inf'), loop_block, loop_order
    else:
        loop_order = loop_order_generator(layer, loop_block, irrelevant)

    rho = bhw * k * r * d / (bhw * layer.hstd * layer.wstd + k * r * d)
    q = math.ceil(layer.total_ops / rho) + layer.total_ofmap_size

    return q, loop_block, loop_order


def _cwr_c_v1(layer, capacity, loop_lower_bound):

    # irrelevant loop: k r d
    irrelevant = [le.D, le.R, le.K]
    c = layer.nifm
    k = loop_lower_bound.k
    r = layer.hfil
    d = layer.wfil

    s = capacity - k * c * r * d
    rho = min(math.ceil(s / (c * layer.hstd * layer.wstd + k)), layer.nimg * layer.hofm * layer.wofm)
    rho, b, h, w = _bhw_factorization(layer, rho, loop_lower_bound)

    k = min(math.ceil((capacity - b*h*w*c*layer.hstd*layer.wstd) / (b*h*w + r*d*c)), layer.nofm)
    loop_block = [d, r, c, w, h, k, b]
    loop_order = [le.NUM - 1] * le.NUM
    if s < 0 or any(i <= 0 for i in loop_block):
        return float('inf'), loop_block, loop_order
    else:
        loop_order = loop_order_generator(layer, loop_block, irrelevant)

    q = layer.total_ifmap_size + math.ceil(layer.total_ops / rho) + layer.total_ofmap_size

    return q, loop_block, loop_order


def _cwr_k_v1(layer, capacity, loop_lower_bound):

    # irrelevant loop: c r d
    irrelevant = [le.D, le.R, le.C]
    k = layer.nofm
    c = loop_lower_bound.c
    r = layer.hfil
    d = layer.wfil

    s = capacity - k * c * r * d

    rho = min(math.ceil(s / (c * layer.hstd * layer.wstd + k)), layer.nimg * layer.hofm * layer.wofm)
    rho, b, h, w = _bhw_factorization(layer, rho, loop_lower_bound)

    c = min(math.ceil((capacity - b * h * w * k) / (b * h * w * layer.hstd * layer.wstd + r * d * k)), layer.nifm)
    loop_block = [d, r, c, w, h, k, b]
    loop_order = [le.NUM - 1] * le.NUM
    if s < 0 or any(i <= 0 for i in loop_block):
        return float('inf'), loop_block, loop_order
    else:
        loop_order = loop_order_generator(layer, loop_block, irrelevant)

    q = layer.total_ifmap_size + math.ceil(layer.total_ops / rho) + layer.total_ofmap_size

    return q, loop_block, loop_order


def _filterr_v1(layer, capacity, loop_lower_bound):

    # irrelevant loop: b h w
    irrelevant = [le.W, le.H, le.B]
    b = loop_lower_bound.b
    h = loop_lower_bound.h
    w = loop_lower_bound.w
    r = layer.hfil
    d = layer.wfil

    f = r * d / (layer.hstd * layer.wstd)
    a = b * h * w * 3 / (4 * f)

    s = capacity

    k = max(math.sqrt(s / (2 * f) + a ** 2) - a, 1)
    c = math.ceil(2 * k / (layer.hstd * layer.wstd))

    # upper bound
    if k > layer.nofm and c > layer.nifm:
        k = layer.nofm
        c = layer.nifm
    elif k > layer.nofm:
        k = layer.nofm
        c = min(math.ceil((s-b*h*w*k)/(r*d*k+b*h*w*layer.hstd*layer.wstd)), layer.nifm)
    elif c > layer.nifm:
        c = layer.nifm
        k = min(math.ceil((s-b*h*w*c*layer.hstd*layer.wstd)/(r*d*c+b*h*w)), layer.nofm)

    # lower bound
    if k * c > loop_lower_bound.k * loop_lower_bound.c:
        if k < loop_lower_bound.k and c > loop_lower_bound.c:
            k = loop_lower_bound.k
            c = min(math.ceil((s - b * h * w * k) / (r * d * k + b * h * w * layer.hstd * layer.wstd)), layer.nifm)
        elif k > loop_lower_bound.k and c < loop_lower_bound.c:
            c = loop_lower_bound.c
            k = min(math.ceil((s - b * h * w * c * layer.hstd * layer.wstd) / (r * d * c + b * h * w)), layer.nofm)

    k = math.floor(k)

    loop_block = [d, r, c, w, h, k, b]
    loop_order = [le.NUM - 1] * le.NUM
    if any(i <= 0 for i in loop_block):
        return float('inf'), loop_block, loop_order
    else:
        loop_order = loop_order_generator(layer, loop_block, irrelevant)

    rho = c * k * r * d / (c * layer.hstd * layer.wstd + (2 * k - 1))
    q = layer.total_filter_size + math.ceil(layer.total_ops / rho)

    return q, loop_block, loop_order


def _ifmapr_v1(layer, capacity, loop_lower_bound):

    # irrelevant loop: k
    irrelevant = [le.K]
    r = layer.hfil
    d = layer.wfil
    k = loop_lower_bound.k
    f = r * d / (layer.hstd * layer.wstd)
    a = k * 3 * f / 4

    s = capacity

    bhw = max(math.ceil(math.sqrt(s*f/2 + a**2) - a), 1)
    c = math.ceil(2 * bhw / (r * d))

    # upper bound
    if bhw > layer.nimg * layer.hofm * layer.wofm and c > layer.nifm:
        bhw = layer.nimg * layer.hofm * layer.wofm
        c = layer.nifm
    if bhw > layer.nimg * layer.hofm * layer.wofm:
        bhw = layer.nimg * layer.hofm * layer.wofm
        c = min(math.ceil((s-bhw*k)/(r*d*k+bhw*layer.hstd*layer.wstd)), layer.nifm)
    elif c > layer.nifm:
        c = layer.nifm
        bhw = min(math.ceil((s-r*d*c*k)/(k+c*layer.hstd*layer.wstd)), layer.nimg * layer.hofm * layer.wofm)

    # lower bound
    if bhw * c > loop_lower_bound.b * loop_lower_bound.h * loop_lower_bound.w * loop_lower_bound.c:
        if bhw < loop_lower_bound.b * loop_lower_bound.h * loop_lower_bound.w and c > loop_lower_bound.c:
            bhw = loop_lower_bound.b * loop_lower_bound.h * loop_lower_bound.w
            c = min(math.ceil((s - bhw * k) / (r * d * k + bhw * layer.hstd * layer.wstd)), layer.nifm)
        elif bhw > loop_lower_bound.b * loop_lower_bound.h * loop_lower_bound.w and c < loop_lower_bound.c:
            c = loop_lower_bound.c
            bhw = min(math.ceil((s - r * d * c * k) / (k + c * layer.hstd * layer.wstd)),
                      layer.nimg * layer.hofm * layer.wofm)

    bhw, b, h, w = _bhw_factorization(layer, bhw, loop_lower_bound)

    k = min(math.ceil((s - b*h*w*c*layer.hstd*layer.wstd) / (b*h*w + r*d*c)), layer.nofm)
    loop_block = [d, r, c, w, h, k, b]
    loop_order = [le.NUM - 1] * le.NUM
    if any(i <= 0 for i in loop_block):
        return float('inf'), loop_block, loop_order
    else:
        loop_order = loop_order_generator(layer, loop_block, irrelevant)

    rho = bhw * c * r * d / (c * r * d + (2 * bhw - 1))
    q = math.ceil(layer.total_ops / rho) + layer.total_ifmap_size

    return q, loop_block, loop_order


def fun(args):
    q0, q1, q2 = args
    v = lambda x: q0 / x[0] + q1 / x[1] + q2 / x[2]
    return v


def con(args):
    x1min, x1max, x2min, x2max, x3min, x3max, a, b, c, xmax = args
    cons = ({'type': 'ineq', 'fun': lambda x: x[0] - x1min},
            {'type': 'ineq', 'fun': lambda x: -x[0] + x1max},
            {'type': 'ineq', 'fun': lambda x: x[1] - x2min},
            {'type': 'ineq', 'fun': lambda x: -x[1] + x2max},
            {'type': 'ineq', 'fun': lambda x: x[2] - x3min},
            {'type': 'ineq', 'fun': lambda x: -x[2] + x3max},
            {'type': 'ineq', 'fun': lambda x: -(a*x[0]*x[1] + b*x[0]*x[2] + c*x[1]*x[2]) + xmax})
    return cons


_unilayer_schedule_list_v1 = [_cwr_c_v1, _cwr_k_v1, _filterr_v1, _ifmapr_v1, _psumsr_v1]
_unilayer_schedule_list_v2 = [_psumsr_v2, _filterr_v2, _ifmapr_v2, _cwr_c_v2, _cwr_k_v2]
