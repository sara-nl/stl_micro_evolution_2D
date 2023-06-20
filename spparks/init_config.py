"""
Last modified on Jun 20, 2023

@author: MR 
"""

# READ_ME
# 1. Go to the "/spparks" folder and make sure to have the "/init" subdirectory included ('/init/'+'IN100_3d.init')
# 2. copy the spk_mpi build into this folder in order to get spparks started.

# import vtk
# import numpy as np
import shutil
import os
import subprocess
import itertools
import functools
import argparse
from typing import List, Tuple

import shlex


def list2Str(lisConv, sep=", "):
    new_list = functools.reduce(lambda x, y: str(x) + sep + str(y), lisConv)
    return new_list


def check_folder_exists(config_name: str, working_dir: str) -> bool:
    folder_exists: bool = False  # Initialize with default value

    for root, dirs, files in os.walk(working_dir):
        for name in dirs:
            if name.endswith(config_name):
                folder_exists = True
                break
        break

    return folder_exists


def create_config_name(config_map):
    config_name_list = []
    for config in range(len(config_map)):
        # if config ==0:
        #     config = config+1
        list2Str(config_map[config], "_")
        config_name = "vHpdV_" + list2Str(config_map[config], "_")
        config_name = config_name.replace(", ", "_")
        config_name = config_name.replace("[", "")
        config_name = config_name.replace("]", "")
        config_name = config_name.replace(".", "_")
        config_name_list.append(config_name)
    return config_name_list


def trans_coord(v, h):
    # v = v*100 # [site/mcs]
    v = [number * 100 for number in v]
    h = h * 1  # [sites]
    return v, h


def create_config_map(
    v_scan: List[float],
    hatch: List[int],
    starting_pos: List[str],
    heading: List[str],
    V_laser: List[List[float]],
) -> Tuple[List[Tuple[float, int, str, str, List[float]]], List[str]]:
    # coordinate transform
    v_mcs, hatch_site = trans_coord(v_scan, hatch)
    all_list = [v_mcs, hatch_site, starting_pos, heading, V_laser]

    # using itertools.product() to compute all possible permutations
    config_map = list(itertools.product(*all_list))
    config_name_list = create_config_name(config_map)
    return config_map, config_name_list


def amend_config_file(config_name: str, working_dir: str) -> None:
    config_file = "config_file"
    config_path = os.path.join(working_dir, config_file)

    try:
        with open(config_path, "w") as new_config_file:
            new_config_file.write(config_name + "\t\n")
        print("Config file amended successfully.")
    except IOError:
        print("Error: Could not amend config file.")


def create_initial_condition(working_dir, directory):
    # local_dir = os.getcwd()
    # src = working_dir + "/init/" + "IN100_3d.init"
    src = working_dir + "/" + "IN100_3d.init"
    dst = directory + "/" + "IN100_3d.init"
    shutil.copyfile(src, dst)
    # src= working_dir+'/template/'+'in.potts_am_IN100_3d'
    # dst = directory+'/in.potts_am_IN100_3d'
    # shutil.copyfile(src, dst)


def create_folder(config_name, working_dir):
    directory = os.path.join(working_dir, config_name)
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Creating directory. " + directory)
    create_initial_condition(working_dir, directory)


def amend_spparks_file(case_name, config_map, working_dir):
    # working_dir = os.getcwd()

    # open files from template & copy new file in the case directory
    with open(
        # working_dir + "/template/" + "in.potts_am_IN100_3d", "r"
        working_dir + "/" + "in.potts_am_IN100_3d",
        "r",
    ) as template, open(
        working_dir + "/" + case_name + "/" + "in.potts_am_IN100_3d", "a"
    ) as new_spparks_file:
        # calculate single hatch line coordinates
        V_x = config_map[0]
        V_y = config_map[0]
        hatch_x = config_map[1]
        hatch_y = config_map[1]
        ATOI = config_map[4]
        # ATOI = [int(i) for i in ATOI_str[0:8]]
        # ATOI.append(float(ATOI_str[8:9]))

        if config_map[3] == "x":
            if config_map[2] == "LL":
                LAYER = "am cartesian_layer 1 start LL pass_id 1 thickness 25 offset -100.0 0.0"
            if config_map[2] == "UL":
                LAYER = "am cartesian_layer 1 start UL pass_id 1 thickness 25 offset 100.0 0.0"
            if config_map[2] == "LR":
                LAYER = "am cartesian_layer 1 start LR pass_id 1 thickness 25 offset -100.0 0.0"
            if config_map[2] == "UR":
                LAYER = "am cartesian_layer 1 start UR pass_id 1 thickness 25 offset 100.0 0.0"
        elif config_map[3] == "y":
            if config_map[2] == "LL":
                LAYER = "am cartesian_layer 1 start LL pass_id 1 thickness 25 offset 0.0 -100.0"
            if config_map[2] == "UL":
                LAYER = "am cartesian_layer 1 start UL pass_id 1 thickness 25 offset 0.0 100.0"
            if config_map[2] == "LR":
                LAYER = "am cartesian_layer 1 start LR pass_id 1 thickness 25 offset 0.0 -100.0"
            if config_map[2] == "UR":
                LAYER = "am cartesian_layer 1 start UR pass_id 1 thickness 25 offset 0.0 100.0"

        # open file corresponding to the selected file name & write coordinates
        # in new structure according to template
        # read content from first file
        for num, line in enumerate(template):
            # append content to second file
            if num <= 10:
                new_spparks_file.write(line)
            elif num > 10 and num <= 11:
                new_spparks_file.write("variable V_x equal " + str(V_x) + "\n")
            elif num > 11 and num <= 12:
                new_spparks_file.write("variable V_y equal " + str(V_y) + "\n")
            elif num > 12 and num <= 14:
                new_spparks_file.write(line)
            elif num > 14 and num <= 15:
                new_spparks_file.write("variable HATCH_x equal " + str(hatch_x) + "\n")
            elif num > 15 and num <= 16:
                new_spparks_file.write("variable HATCH_y equal " + str(hatch_y) + "\n")
            elif num > 16 and num <= 24:
                new_spparks_file.write(line)
            elif num > 24 and num <= 25:
                new_spparks_file.write(
                    "variable case_name universe " + case_name + "\n"
                )
            elif num > 25 and num <= 30:
                new_spparks_file.write(line)
            elif num > 30 and num <= 31:
                new_spparks_file.write("variable ATOI_1 equal " + str(ATOI[0]) + "\n")
            elif num > 31 and num <= 32:
                new_spparks_file.write("variable ATOI_2 equal " + str(ATOI[1]) + "\n")
            elif num > 32 and num <= 33:
                new_spparks_file.write("variable ATOI_3 equal " + str(ATOI[2]) + "\n")
            elif num > 33 and num <= 34:
                new_spparks_file.write("variable ATOI_4 equal " + str(ATOI[3]) + "\n")
            elif num > 34 and num <= 35:
                new_spparks_file.write("variable ATOI_5 equal " + str(ATOI[4]) + "\n")
            elif num > 35 and num <= 36:
                new_spparks_file.write("variable ATOI_6 equal " + str(ATOI[5]) + "\n")
            elif num > 36 and num <= 37:
                new_spparks_file.write("variable ATOI_7 equal " + str(ATOI[6]) + "\n")
            elif num > 37 and num <= 38:
                new_spparks_file.write("variable ATOI_8 equal " + str(ATOI[7]) + "\n")
            elif num > 38 and num <= 39:
                new_spparks_file.write("variable ATOI_9 equal " + str(ATOI[8]) + "\n")
            elif num > 39 and num <= 93:
                new_spparks_file.write(line)
            elif num > 93 and num <= 94:
                new_spparks_file.write(LAYER + "\n")
            elif num > 94 and num <= 97:
                new_spparks_file.write(line)
            else:
                new_spparks_file.write(line)


##############################################################################
# create reader for vtkPolyData file
# template_dir  = '/home/icme/iris/chains/IN100_SLM_150_1_0311/dream3d/data_large'
# template_name = '/SmallIN100_dream3d_synth.spparks'
# case_dir      = '/home/icme/iris/chains/IN100_SLM_150_1_0311/spparks/data_large'
# case_name     = '/SmallIN100_dream3d_synth.spparks'


# Creates a folder in the current directory called data
# template_name =  ['fLP','lD','lP']
# template_dir = 'templates'
# createFolder('./'+template_dir+'/')
# case_name = 'case_single_hatch_1mm'
# createFolder('./'+case_name+'/')
def main():
    working_dir = os.getcwd()

    v_scan = [0.2]  # scan speed in [m/s]
    hatch = [20]  # hatch line distance in [mue_m]
    starting_pos = ["LL", "LR", "UL", "UR"]
    heading = ["x", "y"]
    V_laser = [
        [30, 70, 30, 7, 50, 90, 45, 12, 0.1],
        [50, 90, 50, 14, 100, 120, 45, 24, 0.1],
        [30, 70, 30, 7, 50, 90, 45, 12, 0.15],
    ]  # HAZ zone/ Laser power profile in units of [sites]

    config_map, config_name_list = create_config_map(
        v_scan, hatch, starting_pos, heading, V_laser
    )

    for config in range(len(config_map)):
        config_exists = check_folder_exists(config_name_list[config], working_dir)

        if not config_exists:
            amend_config_file(config_name_list[config], working_dir)
            create_folder(config_name_list[config], working_dir)
            amend_spparks_file(
                config_name_list[config], config_map[config], working_dir
            )

    print("Writing config file and creating folders - DONE")


if __name__ == "__main__":
    main()
