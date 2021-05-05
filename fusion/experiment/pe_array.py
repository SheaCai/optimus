import sys
import os
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('../'))
sys.path.append(os.path.abspath('../../'))

import matplotlib.pyplot as plt
import numpy as np

from fusion.scheduling import batch_size
from fusion.scheduling import Resource
from fusion.scheduling import LoopLowerBound
from fusion.scheduling import ScheduleGenerator
from fusion.scheduling import extract_arch_info, extract_dataflow_info
from fusion.scheduling import CostModel
from fusion.scheduling import res_parse

from fusion.nn_models import import_network


def do_scheduling():
    """
    Get optimal scheduling for given problem. Return a result schedule.
    """

    buffer = [128, 128, 256, 256, 512, 512, 512, 512]
    pe_array = [16, 32, 16, 32, 16, 16, 32, 64]

    # Network.
    batch_size.init(4)
    network = import_network("squeezenet")

    dataflow_info = extract_dataflow_info('./fusion/dataflow/dataflow_Ow_Cout.json')

    access_list = []
    energy_list = []

    for pe, bf in zip(pe_array, buffer):
        arch_file = './fusion/arch/3_level_mem_{}KB.json'.format(bf)
        arch_info = extract_arch_info(arch_file)
        arch_info["parallel_count"][0] = pe ** 2
        if pe == 8:
            arch_info["parallel_cost"][0] = 0.05
        resource = Resource.arch(arch_info)

        # Unroll loop lower bound
        dataflow_info["partitioning_size"] = [pe] * len(dataflow_info["partitioning_size"])
        loop_lower_bound = LoopLowerBound.dataflow(dataflow_info)

        print("\n===========================================================")
        print('PE-array: {}x{}, buffer size: {}(KB)'.format(pe, pe, bf))
        print("waiting...")
        cost_model = CostModel(network, resource)

        # optimal schedule
        sg = ScheduleGenerator(network, resource, cost_model, loop_lower_bound)
        schedule_info_list, _ = sg.schedule_search()
        print("done!\n\n")
        energy, access = res_parse(schedule_info_list, resource,
                                   cost_model, sg, network,
                                   loop_lower_bound,
                                   './result/pe_array', arch_info)
        energy_list.append(energy)
        access_list.append(access)

    x = ["16x16,128", "32x32,128", "16x16,256", "32x32,256", "8x8,512", "16x16,512", "32x32,512", "64x64,512"]
    energy_list = np.array(energy_list) / energy_list[0]
    access_list = np.array(access_list) / access_list[0]
    plt.figure(figsize=(8, 2))
    plt.plot(x, energy_list, label="Normalized Energy")
    plt.plot(x, access_list, label="Normalized DRAM Access")
    plt.ylim(0.2, 1.2)
    plt.legend()
    plt.savefig('./result/pe_array/pe_array.png')
    plt.show()


def main():
    """
    Main function.
    """

    do_scheduling()
    return 0


if __name__ == '__main__':
    sys.exit(main())
