import os
import math
from operator import add
from functools import reduce
from . import loop_enum as le


def res_parse(schedule_info_list, resource, cost_model, sg, network,
              loop_lower_bound, path, arch_info, others=False, is_access=False, is_print=True):

    off_chip_overall = 0
    total_cost = 0
    costs = [0, 0, 0, 0, 0]
    ifmap, ofmap, filter = 0, 0, 0

    utilization = 0
    total_ops = 0
    alu = 0
    if others:
        if schedule_info_list[0] == [[]]:
            schedule_info_list[0] = schedule_info_list[1]
        for section_info in schedule_info_list[0]:
            fusion_group, loop_blocking_list, loop_ordering_list, off_chip_access_test, is_filter_fit = section_info
            cost_inner_list, point_list = sg.mapping(fusion_group, loop_blocking_list, loop_ordering_list)

            access_list, levels_cost, noc_cost, ops, cost = cost_model.get_cost(point_list, fusion_group, is_filter_fit)

            off_chip_access_breakdown, off_chip_access \
                = access_list[2], math.ceil(reduce(add, access_list[2], 0))
            ifmap += off_chip_access_breakdown[0]
            ofmap += off_chip_access_breakdown[1]
            filter += off_chip_access_breakdown[2]
            off_chip_overall += off_chip_access

            utilization_p = 0
            total_ops_p = 0
            for layer, point in zip(fusion_group, point_list):
                if point:
                    x = point.para_loop_dim[0][0]
                    y = point.para_loop_dim[0][1]
                    xx = 1
                    yy = 1
                    for idx in x:
                        xx *= point.loop_partitionings[idx][0]
                    for idx in y:
                        yy *= point.loop_partitionings[idx][0]
                    utilization += xx * yy * network[layer].total_ops / resource.para_count_list[0]
                    utilization_p += xx * yy * network[layer].total_ops / resource.para_count_list[0]
                    total_ops += network[layer].total_ops * 1.15
                    total_ops_p += network[layer].total_ops
            if total_ops_p:
                u = utilization_p / total_ops_p
                alu += total_ops_p / u

        for section_info in schedule_info_list[1]:
            fusion_group, loop_blocking_list, loop_ordering_list, off_chip_access_test, is_filter_fit = section_info
            cost_inner_list, point_list = sg.mapping(fusion_group, loop_blocking_list, loop_ordering_list)

            access_list, levels_cost, noc_cost, ops, cost = cost_model.get_cost(point_list, fusion_group, is_filter_fit)

            num_levels = resource.buffer_levels()

            for i in range(num_levels-1):
                costs[i] += sum(levels_cost[i])
            costs[3] += noc_cost
            costs[4] += ops
        costs[2] = off_chip_overall * resource.access_cost[2][0]
        total_cost = sum(costs)

    else:
        for section_info in schedule_info_list:
            fusion_group, loop_blocking_list, loop_ordering_list, off_chip_access_test, is_filter_fit = section_info
            cost_inner_list, point_list = sg.mapping(fusion_group, loop_blocking_list, loop_ordering_list)

            access_list, levels_cost, noc_cost, ops, cost = cost_model.get_cost(point_list, fusion_group, is_filter_fit)
            total_cost += cost

            num_levels = resource.buffer_levels()

            for i in range(num_levels):
                costs[i] += sum(levels_cost[i])
            costs[3] += noc_cost
            costs[4] += ops

            off_chip_access_breakdown, off_chip_access \
                = access_list[2], math.ceil(reduce(add, access_list[2], 0))
            ifmap += off_chip_access_breakdown[0]
            ofmap += off_chip_access_breakdown[1]
            filter += off_chip_access_breakdown[2]
            off_chip_overall += off_chip_access

            utilization_p = 0
            total_ops_p = 0
            for layer, point in zip(fusion_group, point_list):
                if point:
                    x = point.para_loop_dim[0][0]
                    y = point.para_loop_dim[0][1]
                    xx = 1
                    yy = 1
                    for idx in x:
                        xx *= point.loop_partitionings[idx][0]
                    for idx in y:
                        yy *= point.loop_partitionings[idx][0]
                    utilization += xx * yy * network[layer].total_ops / resource.para_count_list[0]
                    utilization_p += xx * yy * network[layer].total_ops / resource.para_count_list[0]
                    total_ops += network[layer].total_ops * 1.15
                    total_ops_p += network[layer].total_ops
            if total_ops_p:
                u = utilization_p / total_ops_p
                alu += total_ops_p / u

    access = off_chip_overall * resource.precision / 8 / 1024 / 1024
    energy = total_cost / 1e10
    costs[4] = alu
    for i in range(5):
        costs[i] = costs[i] / 1e10

    if is_print:
        print('total DRAM access(MB): ', off_chip_overall * resource.precision / 8 / 1024 / 1024)
        print('DRAM access breakdown[ifmap, ofmap, filter](MB):')
        print(ifmap * resource.precision / 8 / 1024 / 1024,
              ofmap * resource.precision / 8 / 1024 / 1024,
              filter * resource.precision / 8 / 1024 / 1024)
        print('\n')

        if not is_access:
            print('total energy(1e10 pJ):', total_cost / 1e10)
            print('energy breakdown [RFs_cost, buffer_cost, DRAM_cost, noc_cost, mac_cost]: (1e10 pJ):')
            print(costs)
            print('\n')
            print('DRAM access/MAC (1e3):', off_chip_overall * 1000 / total_ops)

    if path != "":
        path += '/' + network.net_name
        if not os.path.exists(path):
            os.makedirs(path)

        filename = path + '/batch' + str(network.input_layer().nimg)

        filename += '-memory_{}B_{}KB_{}GB-Dataflow' \
            .format(int(resource.buffer(0).capacity * resource.precision / 8),
                    int(resource.buffer(1).capacity * resource.precision / 8 / 1024),
                    int(resource.buffer(2).capacity * resource.precision / 8 / 1024 / 1024 / 1024))
        for ulp in loop_lower_bound.unroll_loop:
            filename += '_' + le.table[ulp] + str(loop_lower_bound.loop_lower_bound_init[ulp])

        txt_writer = open(filename, 'w')

        txt_writer.write(str(network))
        txt_writer.write('\n\narch:\n')
        for key in arch_info:
            txt_writer.write(key + ':' + str(arch_info[key]) + '\n')
        txt_writer.write('\n\n')

        txt_writer.write('\n\ntotal DRAM access(MB): \n')
        txt_writer.write(str(off_chip_overall * resource.precision / 8 / 1024 / 1024))
        txt_writer.write('\nDRAM access breakdown[ifmap, ofmap, filter](MB):\n')
        txt_writer.write(str([ifmap * resource.precision / 8 / 1024 / 1024,
                              ofmap * resource.precision / 8 / 1024 / 1024,
                              filter * resource.precision / 8 / 1024 / 1024]))
        if not is_access:
            txt_writer.write('\n\ntotal energy(1e10 pJ): \n')
            txt_writer.write(str(total_cost / 1e10))
            txt_writer.write('\nenergy breakdown [RFs_cost, buffer_cost, DRAM_cost, noc_cost, mac_cost]: (1e10 pJ):\n')
            txt_writer.write(str(costs))

            txt_writer.write('\n\nDRAM access/MAC (1e3):\n')
            txt_writer.write(str(off_chip_overall*1000/total_ops))
        txt_writer.close()

    return energy, access
