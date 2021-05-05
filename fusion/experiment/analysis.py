import sys
import os
import numpy as np
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('../'))
sys.path.append(os.path.abspath('../../'))

from fusion.scheduling import batch_size
from fusion.scheduling import Resource
from fusion.scheduling import LoopLowerBound
from fusion.scheduling import ScheduleGenerator
from fusion.scheduling import extract_info
from fusion.scheduling import CostModel
from fusion.scheduling import res_parse

from fusion.nn_models import all_networks
from fusion.nn_models import import_network
import matplotlib.pyplot as plt


arch_file = './fusion/arch/3_level_mem_64Reg.json'
dataflow_file = './fusion/dataflow/dataflow_Ow_Cout.json'


def do_scheduling_optimus():
    """
    Get optimal scheduling for given problem. Return a result schedule.
    """

    # Resource.
    arch_info, dataflow_info = extract_info(arch_file, dataflow_file)
    resource = Resource.arch(arch_info)

    # Unroll loop lower bound
    loop_lower_bound = LoopLowerBound.dataflow(dataflow_info)

    # batch size = 4
    batch_size.init(4)
    access_list = []
    for net in all_networks():
        # Network.
        network = import_network(net)
        print("\n============================================")
        print(network.net_name)
        print("waiting...")
        cost_model = CostModel(network, resource)

        # optimal schedule
        sg = ScheduleGenerator(network, resource, cost_model, loop_lower_bound)
        schedule_info_list, _ = sg.schedule_search()
        print("done!\n\n")
        _, access = res_parse(schedule_info_list, resource,
                              cost_model, sg, network,
                              loop_lower_bound,
                              './result/analysis/hafs', arch_info, is_access=True)

        access_list.append(int(access))

    return access_list


def do_scheduling_woMinCost():
    """
    Get optimal scheduling for given problem. Return a result schedule.
    """

    # Resource.
    arch_info, dataflow_info = extract_info(arch_file, dataflow_file)
    resource = Resource.arch(arch_info)

    # Unroll loop lower bound
    loop_lower_bound = LoopLowerBound.dataflow(dataflow_info)

    # batch size = 4
    access_list = []
    batch_size.init(4)
    for net in all_networks():
        # Network.
        network = import_network(net)
        print("\n============================================")
        print(network.net_name)
        print("waiting...")
        cost_model = CostModel(network, resource)

        # optimal schedule
        sg = ScheduleGenerator(network, resource, cost_model, loop_lower_bound, z_fusion=True, womincost=True)
        schedule_info_list, _ = sg.schedule_search()
        print("done!\n\n")
        _, access = res_parse(schedule_info_list, resource,
                                   cost_model, sg, network,
                                   loop_lower_bound,
                                   './result/analysis/woMinCost', arch_info, True, is_access=True)

        access_list.append(int(access))

    return access_list


def do_scheduling_woFullSpace1():
    """
    Get optimal scheduling for given problem. Return a result schedule.
    """

    # Resource.
    arch_info, dataflow_info = extract_info(arch_file, dataflow_file)
    resource = Resource.arch(arch_info)

    # Unroll loop lower bound
    loop_lower_bound = LoopLowerBound.dataflow(dataflow_info)

    # batch size = 4
    access_list = []
    batch_size.init(4)
    for net in all_networks():
        # Network.
        network = import_network(net)
        print("\n============================================")
        print(network.net_name)
        print("waiting...")
        cost_model = CostModel(network, resource)

        # optimal schedule
        sg = ScheduleGenerator(network, resource, cost_model, loop_lower_bound, d_fusion=True)
        schedule_info_list, _ = sg.schedule_search()
        print("done!\n\n")
        _, access = res_parse(schedule_info_list, resource,
                              cost_model, sg, network,
                              loop_lower_bound,
                              './result/analysis/woFullSpace1', arch_info, True, is_access=True)

        access_list.append(int(access))

    return access_list


def do_scheduling_woFullSpace2():
    """
    Get optimal scheduling for given problem. Return a result schedule.
    """

    # Resource.
    arch_info, dataflow_info = extract_info(arch_file, dataflow_file)
    resource = Resource.arch(arch_info)

    # Unroll loop lower bound
    loop_lower_bound = LoopLowerBound.dataflow(dataflow_info)

    access_list = []
    # batch size = 4
    batch_size.init(4)
    for net in all_networks():
        # Network.
        network = import_network(net)
        print("\n============================================")
        print(network.net_name)
        print("waiting...")
        cost_model = CostModel(network, resource)

        # optimal schedule
        sg = ScheduleGenerator(network, resource, cost_model, loop_lower_bound, z_fusion=True)
        schedule_info_list, _ = sg.schedule_search()
        print("done!\n\n")
        _, access = res_parse(schedule_info_list, resource,
                              cost_model, sg, network,
                              loop_lower_bound,
                              './result/analysis/woFullSpace2', arch_info, True, is_access=True)

        access_list.append(int(access))

    return access_list


def do_scheduling_woFusion():
    """
    Get optimal scheduling for given problem. Return a result schedule.
    """

    # Resource.
    arch_info, dataflow_info = extract_info(arch_file, dataflow_file)
    resource = Resource.arch(arch_info)

    # Unroll loop lower bound
    loop_lower_bound = LoopLowerBound.dataflow(dataflow_info)

    access_list = []
    # batch size = 4
    batch_size.init(4)
    for net in all_networks():
        # Network.
        network = import_network(net)
        print("\n============================================")
        print(network.net_name)
        print("waiting...")
        cost_model = CostModel(network, resource)

        # optimal schedule
        sg = ScheduleGenerator(network, resource, cost_model, loop_lower_bound, wofusion=True)
        schedule_info_list, _ = sg.schedule_search()
        print("done!\n\n")
        _, access = res_parse(schedule_info_list, resource,
                              cost_model, sg, network,
                              loop_lower_bound,
                              './result/analysis/woFusion', arch_info, True, is_access=True)
        access_list.append(int(access))

    return access_list


def main():
    """
    Main function.
    """
    print('\n')
    print("*" * 60)
    print('HaFS:')
    access_hafs = do_scheduling_optimus()

    print('\n')
    print("*" * 60)
    print('w/o MinCost:')
    access_woMinCost = do_scheduling_woMinCost()

    print('\n')
    print("*" * 60)
    print('w/o full space1:')
    access_woFullSpace1 = do_scheduling_woFullSpace1()

    print('\n')
    print("*" * 60)
    print('w/o full space2:')
    access_woFullSpace2 = do_scheduling_woFullSpace2()

    print('\n')
    print("*" * 60)
    print('w/o fusion:')
    access_woFusion = do_scheduling_woFusion()

    x = list(range(len(access_hafs)))
    total_width, n = 1, 6
    width = total_width / n

    plt.figure(figsize=(10, 4))
    plt.bar(x, list(np.array(access_hafs) / np.array(access_hafs)),
            width=width, label='HaFS', color='r', edgecolor='black')
    for i in range(len(x)):
        x[i] = x[i] + width
    plt.bar(x, list(np.array(access_woMinCost) / np.array(access_hafs)),
            width=width, label='w/o MinCost', color='r', edgecolor='black')
    for i in range(len(x)):
        x[i] = x[i] + width
    plt.bar(x, list(np.array(access_woFullSpace1) / np.array(access_hafs)), width=width,
            label='w/o full space1', tick_label=list(all_networks()), color='r', edgecolor='black')
    for i in range(len(x)):
        x[i] = x[i] + width
    plt.bar(x, list(np.array(access_woFullSpace2) / np.array(access_hafs)),
            width=width, label='w/o full space2', color='r', edgecolor='black')
    for i in range(len(x)):
        x[i] = x[i] + width
    plt.bar(x, list(np.array(access_woFusion) / np.array(access_hafs)),
            width=width, label='w/o fusion', color='r', edgecolor='black')

    plt.ylabel("Normalized Access")
    plt.xlabel("models")
    plt.ylim(0.0, 4.0)
    plt.legend()
    plt.savefig('./result/analysis/analysis.png')
    plt.show()
    return 0


if __name__ == '__main__':
    sys.exit(main())
