import sys
import os
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
        res_parse(schedule_info_list, resource,
                  cost_model, sg, network,
                  loop_lower_bound,
                  './result/overall_experiment/hafs', arch_info)


def do_scheduling_dnnvm():
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
    for net in all_networks():
        # Network.
        network = import_network(net)
        print("\n============================================")
        print(network.net_name)
        print("waiting...")
        cost_model = CostModel(network, resource)

        # optimal schedule
        sg = ScheduleGenerator(network, resource, cost_model, loop_lower_bound, d_fusion=True, womincost=True)
        schedule_info_list, _ = sg.schedule_search()
        print("done!\n\n")
        res_parse(schedule_info_list, resource,
                  cost_model, sg, network,
                  loop_lower_bound,
                  './result/overall_experiment/dnnvm', arch_info, True)


def do_scheduling_efficients():
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
        res_parse(schedule_info_list, resource,
                  cost_model, sg, network,
                  loop_lower_bound,
                  './result/overall_experiment/efficients', arch_info, True)


def main():
    """
    Main function.
    """
    print('\n')
    print("*" * 60)
    print('HaFS:')
    do_scheduling_optimus()
    print('\n')
    print("*" * 60)
    print('Efficient-S:')
    do_scheduling_efficients()
    print('\n')
    print("*" * 60)
    print('DNNVM:')
    do_scheduling_dnnvm()
    return 0


if __name__ == '__main__':
    sys.exit(main())
