"""last modified on 3 July"""

import os
from typing import List, Tuple
import numpy as np

from argparse import ArgumentParser

import h5py
import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk

# TODO
# the path of the subfolders are in the config file
# for now read only one and understand what it's format should be:
# every subfolder is a sample: a video, an instance of my time evolution - this is a sample
# every sample has a list of images:
# every image is a temporal scrrenshot of my evolution - single vtk instance - ex vtk.0 is my image at time 0


def read_vtk_instance(reader: vtk.vtkXMLImageDataReader, filename: str) -> np.ndarray:
    """Read single vti.n file"""
    reader.SetFileName(filename)
    reader.Update()
    data = reader.GetOutput()

    cell_data = data.GetCellData()
    spin_data = cell_data.GetArray("Spin")

    dimx, dimy, dimz = data.GetDimensions()

    # get additional info (not needed for now)
    spacing = data.GetSpacing()
    range_min, range_max = spin_data.GetRange()  # useful for normalize my data
    num_components = spin_data.GetNumberOfComponents()

    np_array = vtk_to_numpy(spin_data)

    # Subtract 1 to get the correct dimensions for cell data
    dimx, dimy, dimz = dimx - 1, dimy - 1, dimz - 1
    np_array = np_array.reshape(dimx, dimy, dimz)

    return np_array


def read_vtk_sample(reader: vtk.vtkXMLImageDataReader, path: str) -> List[np.ndarray]:
    """We are inside a vHpdV_ folder: read the temporal sequence"""
    # Iterate over the range of numbers from vti.0 to vti.N
    temporal_sample = []
    n = 0
    while True:
        file_path = os.path.join(path, f"IN1003d.vti.{n}")
        if not os.path.isfile(file_path):
            break

        instance = read_vtk_instance(reader, file_path)
        temporal_sample.append(instance)

        n += 1

    return temporal_sample


def read_vtk_from_config(
    args: List[str], config_file: str, reader: vtk.vtkXMLImageDataReader()
) -> List[List[np.ndarray]]:
    """Read the samples path from config file"""
    data_list = []
    with open(config_file, "r") as file:
        for line in file:
            subfolder = line.strip()
            path = os.path.join(args.data_path, subfolder)

            sample = read_vtk_sample(reader, path)

            data_list.append(sample)

    # print(len(data_list[0]))  # single sample

    return data_list


def save_data_to_hdf5(data_lists: List[List[np.ndarray]], output_file: str) -> None:
    with h5py.File(output_file, "w") as f:
        for i, inner_list in enumerate(data_lists):
            group_name = f"sample_{i}"
            group = f.create_group(group_name)

            for j, data_array in enumerate(inner_list):
                dataset_name = f"data_instance{j}"
                group.create_dataset(dataset_name, data=data_array)

    print("dataset file saved")


def read_data_from_hdf5(input_file: str) -> List[List[np.ndarray]]:
    data_arrays = []

    with h5py.File(input_file, "r") as f:
        for group_name in f.keys():
            inner_list = []

            group = f[group_name]
            for dataset_name in group.keys():
                data_array = group[dataset_name][:]
                inner_list.append(data_array)

            data_arrays.append(inner_list)

    return data_arrays


def main(args):
    config_file = args.config_file
    reader = vtk.vtkXMLImageDataReader()
    data_list = read_vtk_from_config(args, config_file, reader)

    len_data = len(data_list)
    hdf5_file = os.path.join(args.data_path, "data_24.hdf5")
    save_data_to_hdf5(data_list, hdf5_file)

    # checks
    new_vti_path = os.path.join(args.data_path, "test1.vti.0")
    read_data = read_data_from_hdf5(hdf5_file)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--data_path", type=str, default="/home/monicar/esa/output_data"
    )
    parser.add_argument(
        "--config_file", type=str, default="/home/monicar/esa/output_data/config_file"
    )
    args = parser.parse_args()
    main(args)
