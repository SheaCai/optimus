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

from fusion.nn_models import import_network


def do_scheduling():
    """
    Get optimal scheduling for given problem. Return a result schedule.
    """

    dataflow_dir = './fusion/dataflow/'
    name = os.listdir(dataflow_dir)
    # batch size = 4
    batch_size.init(4)
    network = import_network("squeezenet")
    for rfs in [64, 512]:
        print("\n\n"+"*"*80)
        print("\nRFs: {}B/PE".format(rfs))
        arch = './fusion/arch/3_level_mem_{}Reg.json'.format(rfs)
        for dataflow in name:
            if dataflow[-4:] == "json":
                # Resource.
                arch_info, dataflow_info = extract_info(arch,
                                                    dataflow_dir+dataflow)

                resource = Resource.arch(arch_info)

                # Unroll loop lower bound
                loop_lower_bound = LoopLowerBound.dataflow(dataflow_info)

                print("\n")
                print("="*50)
                print(dataflow[:-5])
                print("waiting...")
                cost_model = CostModel(network, resource)

                # optimal schedule
                sg = ScheduleGenerator(network, resource, cost_model, loop_lower_bound)
                schedule_info_list, _ = sg.schedule_search()
                print("done!\n\n")
                energy, access = res_parse(schedule_info_list, resource,
                                           cost_model, sg, network,
                                           loop_lower_bound,
                                           './result/dataflow', arch_info)


def main():
    """
    Main function.
    """
    do_scheduling()
    return 0


if __name__ == '__main__':
    sys.exit(main())