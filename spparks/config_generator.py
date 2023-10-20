"""
Last modified on Oct 18, 2023

Configuration Generation Script:

- Contains all functions related to reading parameters from the YAML file.
- Generates configurations and writes them to the config_file.
- This script can be run whenever there's a need to update or generate new configurations.

"""

import os
import itertools
from argparse import ArgumentParser
from typing import List, Tuple, Optional
import numpy as np

from concurrent.futures import ProcessPoolExecutor

from potts_param import Potts_Param


def _trans_coord(v, h):
    # v = v*100 # [site/mcs]
    v = [number * 100 for number in v]
    h = h * 1  # [sites]
    return v, h


def _create_config_name(config_map):
    config_name_list = []

    for config in config_map:
        # Convert tuple elements to strings, handle lists inside the tuple separately
        elements = [
            str(item).replace(".", "_")
            if not isinstance(item, list)
            else "_".join(map(lambda x: str(x).replace(".", "_"), item))
            for item in config
        ]

        # Join all elements with underscores and prepend the "vHpdV_"
        config_name = "vHpdV_" + "_".join(elements)
        config_name_list.append(config_name)

    return config_name_list


def create_config_map(
    params: Potts_Param,
    V_laser: List[List[float]],
) -> Tuple[List[Tuple[float, int, str, str, List[float]]], List[str]]:
    """
    Generate a configuration map and associated configuration names based on the provided parameters and V_laser values.
    """

    # coordinate transform
    v_mcs, hatch_site = _trans_coord(params.v_scan, params.hatch)
    all_list = [v_mcs, hatch_site, params.starting_pos, params.heading, V_laser]

    config_map = list(itertools.product(*all_list))
    config_name_list = _create_config_name(config_map)

    return config_map, config_name_list


def _write_chunk(chunk: List[str], output_file: str) -> bool:
    try:
        with open(output_file, "w") as file:
            for config_name in chunk:
                file.write(config_name + "\t\n")
        return True
    except IOError:
        return False


def amend_config_file_chunks(
    config_names: List[str], output_dir: str, num_chunks: int = 10
) -> List[Optional[str]]:
    # Calculate chunk size
    chunk_size = len(config_names) // num_chunks
    chunks = [
        config_names[i : i + chunk_size]
        for i in range(0, len(config_names), chunk_size)
    ]

    output_files = [
        os.path.join(output_dir, f"config_file_{i}") for i in range(1, num_chunks + 1)
    ]

    # Write chunks in parallel
    successful_files = []
    cpus_per_task = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
    with ProcessPoolExecutor(max_workers=cpus_per_task) as executor:
        results = list(executor.map(_write_chunk, chunks, output_files))

    for was_successful, output_file in zip(results, output_files):
        if was_successful:
            successful_files.append(output_file)
        else:
            print(f"Error: Failed to write to {output_file}.")
            successful_files.append(None)

    return successful_files


def amend_config_file(config_names: List[str], output_dir: str) -> Optional[str]:
    """
    Write the provided configuration names to a file in the specified directory.

    Args:
    - config_names (List[str]): A list of configuration names to write.
    - working_dir (str): The directory where the configuration file should be written.

    Returns:
    - str: The path to the written configuration file or None if there was an error.
    """

    config_file = "config_file"
    config_path = os.path.join(output_dir, config_file)

    try:
        with open(config_path, "w") as new_config_file:
            for config_name in config_names:
                new_config_file.write(config_name + "\t\n")

    except IOError as e:
        print(f"Error: Could not amend config file. Reason: {e}")
        return None

    return config_path


def create_HAZ_permutations(params: Potts_Param) -> List[List[int]]:
    # spot_width melt_tail_length melt_depth cap_height HAZ_width HAZ_tail depth_HAZ cap_HAZ exp_factor
    HAZ_list = [
        params.spot_width,
        params.melt_tail_length,
        params.melt_depth,
        params.cap_height,
        params.HAZ_width,
        params.HAZ_tail,
        params.depth_HAZ,
        params.cap_HAZ,
        params.exp_factor,
    ]

    def _valid_combination(combination):
        return (
            combination[0] < combination[4]  # spot_width < HAZ_width
            and combination[1] < combination[5]  # melt_tail_length < HAZ_tail
            and combination[2] < combination[6]  # melt_depth < depth_HAZ
            and combination[3] < combination[7]  # cap_height < cap_HAZ
        )

    # HAZ zone/ Laser power profile in units of [sites]
    # HAZ_map = list(itertools.product(*HAZ_list))
    HAZ_map = filter(_valid_combination, itertools.product(*HAZ_list))
    HAZ_map_list = [list(item) for item in HAZ_map]
    return HAZ_map_list


def main(args):
    working_dir = args.working_dir
    yaml_file = args.yaml_file
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    params = Potts_Param(yaml_file)
    V_laser = create_HAZ_permutations(params)

    config_map, config_name_list = create_config_map(params, V_laser)

    # write config file
    path = amend_config_file_chunks(config_name_list, output_dir)
    print("config_file created: ", path)


if __name__ == "__main__":
    parser = ArgumentParser()
    home_dir = os.environ["HOME"]
    parser.add_argument(
        "--working_dir",
        type=str,
        default=f"{home_dir}/esa/IN100_SLM_AI_Training_Set_II/spparks",
        help="define working dir",
    )
    parser.add_argument(
        "--yaml_file",
        type=str,
        default=f"{home_dir}/esa/ml-materials-engineering/spparks/small_params.yaml",
        help="yaml file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=f"{home_dir}/esa/IN100_SLM_AI_Training_Set_II/spparks",
        help="define dir where to output config_file",
    )

    args = parser.parse_args()
    main(args)
