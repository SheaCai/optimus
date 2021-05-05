import json

from . import loop_enum as le


def extract_arch_info(arch_file):
    with open(arch_file) as json_data_file:
        data = json.load(json_data_file)
    assert data["mem_levels"] == len(data["capacity"]), \
        "capacity list is invalid, too many or too few elements"
    assert data["mem_levels"] == len(data["access_cost"]), \
        "access_cost list is invalid, too many or too few elements"
    assert data["mem_levels"] == len(data["parallel_count"]), \
        "parallel_count list is invalid, too many or too few elements"

    if "precision" not in data:
        data["precision"] = 16
    num_bytes = data["precision"] / 8

    if type(data["capacity"][0]) is list:
        capacity_list = [[x / num_bytes for x in data["capacity"][i]] for i in range(len(data["capacity"]))]
    else:
        capacity_list = [x / num_bytes for x in data["capacity"]]
    data["capacity"] = capacity_list
    if "static_cost" not in data:
        data["static_cost"] = [0, ] * data["mem_levels"]
    else:
        assert data["mem_levels"] == len(data["static_cost"]), \
            "static_cost list is invalid, too many or too few elements"

    if "mac_capacity" not in data:
        data["mac_capacity"] = 0
    if "parallel_mode" not in data:
        data["parallel_mode"] = [0, ] * data["mem_levels"]
        for level in range(data["mem_levels"]):
            if data["parallel_count"][level] != 1:
                data["parallel_mode"][level] = 1
    else:
        assert data["mem_levels"] == len(data["parallel_mode"]), \
            "parallel_mode list is invalid, too many or too few elements"

    if "array_dim" not in data:
        data["array_dim"] = None
    if "utilization_threshold" not in data:
        data["utilization_threshold"] = 0.0
    if "replication" not in data:
        data["replication"] = True
    if "invalid_underutilized" not in data:
        data["invalid_underutilized"] = True
    if "memory_partitions" not in data:
        data["memory_partitions"] = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

    return data


def extract_dataflow_info(dataflow_file):
    with open(dataflow_file) as json_data_file:
        data = json.load(json_data_file)

    assert len(data["partitioning_size"]) == len(data["unroll_loop"]), \
        "unroll_loop list is invalid, too many or too few elements"
    if "replication_loop" in data:
        assert len(data["partitioning_size"]) == len(data["unroll_loop"]), \
            "replication_loop list is invalid, too many or too few elements"

    data["loop_lower_bound"] = [1, ] * le.NUM
    for i in range(len(data["unroll_loop"])):
        data["loop_lower_bound"][data["unroll_loop"][i]] = data["partitioning_size"][i]

    if "replication_loop" not in data:
        data["replication_loop"] = None

    return data


def extract_info(arch, dataflow):
    arch_info = extract_arch_info(arch)
    dataflow_info = extract_dataflow_info(dataflow) if dataflow else None

    return arch_info,  dataflow_info
