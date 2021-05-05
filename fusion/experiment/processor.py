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

from fusion.nn_models import import_network


# googlenet too slow
# networks = ['alexnet', 'vggnet', 'googlenet', 'resnet18', 'squeezenet']
networks = ['alexnet', 'vggnet', 'resnet18', 'squeezenet']


def do_scheduling_optimus(arch_file, dataflow_file, store_path):
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
    for net in networks:
        # Network.
        network = import_network(net)
        # print("\n============================================")
        # print(network.net_name)
        # print("waiting...")
        cost_model = CostModel(network, resource)

        # optimal schedule
        sg = ScheduleGenerator(network, resource, cost_model, loop_lower_bound)
        schedule_info_list, _ = sg.schedule_search()
        # print("done!\n\n")
        _, access = res_parse(schedule_info_list, resource,
                              cost_model, sg, network,
                              loop_lower_bound,
                              store_path, arch_info, is_access=True, is_print=False)
        access_list.append(access)
    return access_list


def do_scheduling_woFusion(arch_file, dataflow_file, store_path, is_shiDianNao=False):
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
    for net in networks:
        # Network.
        network = import_network(net)
        # print("\n============================================")
        # print(network.net_name)
        # print("waiting...")
        cost_model = CostModel(network, resource)

        # optimal schedule
        sg = ScheduleGenerator(network, resource, cost_model, loop_lower_bound,
                               wofusion=True, is_shiDianNao=is_shiDianNao)
        schedule_info_list, _ = sg.schedule_search()
        # print("done!\n\n")
        _, access = res_parse(schedule_info_list, resource,
                              cost_model, sg, network,
                              loop_lower_bound,
                              store_path, arch_info, True, is_access=True, is_print=False)
        access_list.append(access)
    return access_list


def main():
    """
    Main function.
    """

    print('\n')
    print("*" * 60)
    print('Eyeriss:')
    print("waiting...")
    arch_file = './fusion/processor/eyeriss.json'
    dataflow_file = './fusion/processor/dataflow_eyeriss.json'
    store_path = './result/processor/eyeriss/hafs'
    access_hafs = do_scheduling_optimus(arch_file, dataflow_file, store_path)
    store_path = './result/processor/eyeriss/baseline'
    access_eyeriss= do_scheduling_woFusion(arch_file, dataflow_file, store_path)
    print("done!\n\n")
    print('Reduction in memory access (%) {}:'.format(networks))
    print((np.array(access_eyeriss) - np.array(access_hafs)) * 100 / np.array(access_eyeriss))

    print('\n')
    print("*" * 60)
    print('shiDianNao:')
    print("waiting...")
    arch_file = './fusion/processor/shiDianNao.json'
    dataflow_file = './fusion/processor/dataflow_shiDianNao.json'
    store_path = './result/processor/shiDianNao/hafs'
    access_hafs = do_scheduling_optimus(arch_file, dataflow_file, store_path)
    store_path = './result/processor/shiDianNao/baseline'
    access_shiDianNao = do_scheduling_woFusion(arch_file, dataflow_file, store_path, is_shiDianNao=True)
    print("done!\n\n")
    print('Reduction in memory access (%) {}:'.format(networks))
    print((np.array(access_shiDianNao) - np.array(access_hafs)) * 100 / np.array(access_shiDianNao))

    print('\n')
    print("*" * 60)
    print('LowerBound:')
    print("waiting...")
    arch_file = './fusion/processor/lowerbound.json'
    dataflow_file = './fusion/dataflow/dataflow_Ow_Cout.json'
    store_path = './result/processor/lowerbound/hafs'
    access_hafs = do_scheduling_optimus(arch_file, dataflow_file, store_path)
    store_path = './result/processor/lowerbound/baseline'
    access_lowerbound = do_scheduling_woFusion(arch_file, dataflow_file, store_path)
    print("done!\n\n")
    print('Reduction in memory access (%) {}:'.format(networks))
    print((np.array(access_lowerbound) - np.array(access_hafs)) * 100 / np.array(access_lowerbound))

    return 0


if __name__ == '__main__':
    sys.exit(main())
