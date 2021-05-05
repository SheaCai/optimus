"""
cost model
"""
import numpy as np
from operator import mul, add

from . import loop_enum as le
from .layer import ConvLayer
from .network import Network
from .resource import Resource


class CostModel(object):
    def __init__(self, network, resource):
        if not isinstance(network, Network):
            raise TypeError("ScheduleGenerator: network must be a Network instance.")
        if not isinstance(resource, Resource):
            raise TypeError("ScheduleGenerator: resource must be a Resource instance.")
        self.network = network
        self.resource = resource

    def get_op(self, fusion_group):
        op = 0
        for l in fusion_group:
            layer = self.network[l]
            op += layer.total_ops
        return op

    @staticmethod
    def get_if_access(level, point, layer, mac_capacity=0):
        """
        Get per element # of access of Input at current level.
        Not accurate because [D, R] is not totally irrelevant terms for ifmap.
        """

        if level == 0 and mac_capacity == 0:
            return layer.wfil * layer.hfil * layer.nofm / (layer.wstd * layer.hstd)

        ex_order_index = min(point.loop_orders[le.W][level],
                             point.loop_orders[le.H][level],
                             point.loop_orders[le.C][level],
                             point.loop_orders[le.B][level])

        r_exclusive = point.loop_orders[le.D][level] < ex_order_index
        d_exclusive = point.loop_orders[le.R][level] < ex_order_index
        k_exclusive = point.loop_orders[le.K][level] < ex_order_index

        rdk_acc = (layer.wfil * layer.hfil * layer.nofm) / \
                  (point.loop_blockings[le.D][level - 1 + r_exclusive]
                   * point.loop_blockings[le.R][level - 1 + d_exclusive]
                   * point.loop_blockings[le.K][level - 1 + k_exclusive])

        if point.loop_blockings[le.D][level - 1 + r_exclusive] < layer.wfil:
            rdk_acc /= layer.wstd
        if point.loop_blockings[le.R][level - 1 + d_exclusive] < layer.hfil:
            rdk_acc /= layer.hstd
        return rdk_acc

    @staticmethod
    def get_of_access(level, point, layer, mac_capacity=0):
        """
        Get per element # of access of Output at current level.
        """

        if level == 0 and mac_capacity == 0:
            return layer.wfil * layer.hfil * layer.nifm

        ex_order_index = min(point.loop_orders[le.W][level],
                             point.loop_orders[le.H][level],
                             point.loop_orders[le.K][level],
                             point.loop_orders[le.B][level])

        r_exclusive = point.loop_orders[le.D][level] < ex_order_index
        d_exclusive = point.loop_orders[le.R][level] < ex_order_index
        c_exclusive = point.loop_orders[le.C][level] < ex_order_index

        rdc_acc = (layer.wfil * layer.hfil * layer.nifm) / \
                  (point.loop_blockings[le.D][level - 1 + r_exclusive]
                   * point.loop_blockings[le.R][level - 1 + d_exclusive]
                   * point.loop_blockings[le.C][level - 1 + c_exclusive])

        return rdc_acc

    @staticmethod
    def get_fl_access(level, point, layer, mac_capacity=0):
        """
        Get per element # of access of Weight at current level.
        """

        if level == 0 and mac_capacity == 0:
            return layer.wofm * layer.hofm * layer.nimg

        ex_order_index = min(point.loop_orders[le.D][level],
                             point.loop_orders[le.R][level],
                             point.loop_orders[le.C][level],
                             point.loop_orders[le.K][level])

        w_exclusive = point.loop_orders[le.W][level] < ex_order_index
        h_exclusive = point.loop_orders[le.H][level] < ex_order_index
        b_exclusive = point.loop_orders[le.B][level] < ex_order_index

        whb_acc = (layer.wofm * layer.hofm * layer.nimg) / \
                  (point.loop_blockings[le.W][level - 1 + w_exclusive]
                   * point.loop_blockings[le.H][level - 1 + h_exclusive]
                   * point.loop_blockings[le.B][level - 1 + b_exclusive])

        return whb_acc

    @staticmethod
    def get_layer_size(layer):
        """
        Get size of ifmap, ofmap, filter of the layer.
        """

        return [layer.total_ifmap_size, layer.total_ofmap_size, layer.total_filter_size]

    def get_level_access(self, point_list, fusion_group, level, is_filter_fit=False):

        if level == self.resource.buffer_levels() - 1 and len(fusion_group) > 1:
            buffer_access = \
                self.get_level_access_multilayer(point_list, fusion_group, level, is_filter_fit)
        else:
            buffer_access = \
                self.get_level_access_unilayer_and_innerlevel(point_list, fusion_group, level)

        return buffer_access

    def get_level_access_unilayer_and_innerlevel(self, point_list, fusion_group, level):
        """
        Get the energy from current level of memory access
        """
        buffer_access = list([0, 0, 0])
        for point, layer_name in zip(point_list, fusion_group):
            layer = self.network[layer_name]
            if not isinstance(layer, ConvLayer):
                continue
            buffer_access_one_layer = self.get_level_access_unilayer(point, layer, level)
            buffer_access = list(map(add, buffer_access_one_layer, buffer_access))

        return buffer_access

    def get_level_access_unilayer(self, point, layer, level):
        layer_size = self.get_layer_size(layer)
        mac_capacity = self.resource.mac_capacity

        level_access = [self.get_if_access(level, point, layer, mac_capacity),
                        2 * self.get_of_access(level, point, layer, mac_capacity) - 1,
                        self.get_fl_access(level, point, layer, mac_capacity)]

        buffer_access_one_layer = list(map(mul, level_access, layer_size))
        buffer_access_one_layer = np.ceil(buffer_access_one_layer).astype(int).tolist()

        return buffer_access_one_layer

    def get_level_access_multilayer(self, point_list, fusion_group, level, is_filter_fit=False):

        assert level == self.resource.buffer_levels() - 1

        mac_capacity = self.resource.mac_capacity
        fuse_ifmap_access, fuse_ofmap_access, fuse_filter_access = 0, 0, 0

        ext_outputs = set()
        ext_inputs = set()
        for point, layer_name in zip(point_list, fusion_group):
            layer = self.network[layer_name]
            if isinstance(layer, ConvLayer):
                fuse_filter_access += \
                    layer.total_filter_size \
                    * (1 if is_filter_fit else self.get_fl_access(level, point, layer, mac_capacity))
            for nx in self.network.nexts(layer_name):
                if nx not in fusion_group:
                    ext_outputs.add(layer_name)

            for pre in self.network.prevs(layer_name):
                if pre not in fusion_group:
                    ext_inputs.add(pre)
        for ip in ext_inputs:
            if ip is None:
                fuse_ifmap_access += self.network[self.network.INPUT_LAYER_KEY].total_ofmap_size
            else:
                fuse_ifmap_access += self.network[ip].total_ofmap_size
        for op in ext_outputs:
            fuse_ofmap_access += self.network[op].total_ofmap_size

        buffer_access = [fuse_ifmap_access, fuse_ofmap_access, fuse_filter_access]
        buffer_access = np.ceil(buffer_access).astype(int).tolist()
        return buffer_access

    def get_array_access_and_cost(self, point, layer, level):

        para = self.resource.paras[level]
        mac_capacity = self.resource.mac_capacity
        layer_size = self.get_layer_size(layer)

        para_mode = para.access_mode
        assert para_mode == 1 or para_mode == 2

        array_dim = para.array_dim
        para_cost = para.array_access_cost * 1.0
        nearest_pe_cost = para_cost

        if_block_access = self.get_if_access(level+1, point, layer, mac_capacity)
        of_block_access = self.get_of_access(level+1, point, layer, mac_capacity)
        fl_block_access = self.get_fl_access(level+1, point, layer, mac_capacity)

        partitions = list(zip(*point.loop_partitionings))[level]
        para_dim = point.para_loop_dim[level]

        partitions_nearest = [1, ] * le.NUM
        partitions_far = []
        across_block_cost = [0] * array_dim

        if para_mode == 1:
            for i in range(len(para_dim)):
                para_index = para_dim[i]
                partitions_far.append([1, ] * le.NUM)
                if len(para_index) == 1:
                    partitions_nearest[para_index[0]] = partitions[para_index[0]]
                else:
                    inner_loop, outer_loop = para_index
                    partitions_nearest[inner_loop] = partitions[inner_loop]
                    partitions_far[i][outer_loop] = partitions[outer_loop]
                    across_block_cost[i] = para_cost * partitions[inner_loop]

            array_if_block_access_nearest \
                = if_block_access * partitions_nearest[le.R] * partitions_nearest[le.D] * partitions_nearest[le.K]
            array_of_block_access_nearest \
                = of_block_access * partitions_nearest[le.R] * partitions_nearest[le.D] * partitions_nearest[le.C]
            array_fl_block_access_nearest \
                = fl_block_access * partitions_nearest[le.W] * partitions_nearest[le.H] * partitions_nearest[le.B]

            level_access \
                = [array_if_block_access_nearest, array_of_block_access_nearest, array_fl_block_access_nearest]
            buffer_access = [np.ceil(list(map(mul, level_access, layer_size))).astype(int).tolist()]
            # buffer_access = [level_access]

            for i in range(array_dim):  # Don't get it
                if_partitions_far = partitions_far[i][le.R] * partitions_far[i][le.D] * partitions_far[i][le.K]
                if_partitions_far = if_partitions_far if if_partitions_far != 1 else 0
                of_partitions_far = partitions_far[i][le.R] * partitions_far[i][le.D] * partitions_far[i][le.C]
                of_partitions_far = of_partitions_far if of_partitions_far != 1 else 0
                fl_partitions_far = partitions_far[i][le.W] * partitions_far[i][le.H] * partitions_far[i][le.B]
                fl_partitions_far = fl_partitions_far if fl_partitions_far != 1 else 0

                if_array_block_access = if_block_access * if_partitions_far
                of_array_block_access = of_block_access * of_partitions_far
                fl_array_block_access = fl_block_access * fl_partitions_far

                level_access \
                    = [if_array_block_access, of_array_block_access, fl_array_block_access]
                buffer_access.append(np.ceil(list(map(mul, level_access, layer_size))).astype(int).tolist())

            return [buffer_access, [nearest_pe_cost] + across_block_cost]

        elif para_mode == 2:
            for i in range(len(para_dim)):
                para_index = para_dim[i]
                for j in para_index:
                    partitions_nearest[j] = partitions[j]

            array_if_block_access_nearest \
                = if_block_access * partitions_nearest[le.R] * partitions_nearest[le.D] * partitions_nearest[le.K]
            array_of_block_access_nearest\
                = of_block_access * partitions_nearest[le.R] * partitions_nearest[le.D] * partitions_nearest[le.C]
            array_fl_block_access_nearest \
                = fl_block_access * partitions_nearest[le.W] * partitions_nearest[le.H] * partitions_nearest[le.B]

            level_access \
                = [array_if_block_access_nearest, array_of_block_access_nearest, array_fl_block_access_nearest]
            buffer_access = [np.ceil(list(map(mul, level_access, layer_size))).astype(int).tolist()]

            return [buffer_access, [nearest_pe_cost]]

    def get_access(self, point_list, fusion_group, is_filter_fit):

        # TODO support more customized memory
        # TODO more access at overlapped boundary

        num_levels = self.resource.buffer_levels()

        access_list = []
        for level in range(num_levels):
            buffer_access = self.get_level_access(point_list, fusion_group, level, is_filter_fit)
            access_list.append(buffer_access)

        return access_list

    def get_cost(self, point_list, fusion_group, is_filter_fit):

        access_list = self.get_access(point_list, fusion_group, is_filter_fit)
        num_levels = self.resource.buffer_levels()

        levels_cost = []
        levels_cost_breakdown = []
        for level in range(num_levels):
            if type(self.resource.access_cost[level]) is list:
                level_cost = sum(list(map(mul, access_list[level], self.resource.access_cost[level])))
                levels_cost_breakdown.append(list(map(mul, access_list[level], self.resource.access_cost[level])))
                levels_cost.append(level_cost)
            else:
                level_cost = sum(list(map(mul, access_list[level], [self.resource.access_cost[level]]*3)))
                levels_cost_breakdown.append(list(map(mul, access_list[level], [self.resource.access_cost[level]]*3)))
                levels_cost.append(level_cost)

        noc_cost = 0
        ops = 0
        for point, layer_name in zip(point_list, fusion_group):
            layer = self.network[layer_name]
            if isinstance(layer, ConvLayer):
                ops += layer.total_ops
                array_access, array_cost = self.get_array_access_and_cost(point, layer, 0)
                for i in range(len(array_access)):
                    noc_cost += sum(array_access[i]) * array_cost[i]

        return access_list, levels_cost_breakdown, noc_cost, ops, sum(levels_cost)+noc_cost+ops
