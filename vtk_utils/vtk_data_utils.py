"""last modified on 4 July"""

import os
from typing import List, Tuple
import numpy as np

from argparse import ArgumentParser

import h5py
import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk

# TODO
# the path of the subfolders are in the config file
# every subfolder is a sample: a video, an instance of my time evolution - this is a sample
# every sample has a list of images:
# every image is a temporal scrrenshot of my evolution - single vtk instance - ex vtk.0 is my image at time 0

"""Collection of methods for reading the "subfolder/*.vti.*" generated from spparks"""
"""generate hdf5 dataset from all subfolders"""
"""also generate vti back from hdf5 as test (and for future use)"""


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
    """We are inside a vHpdV_ folder (that's my sample): read the temporal sequence"""
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
    """
    Read the samples path from config file
    
    the np.ndarray shape is (100,100,50)
    """

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
    """Given a file hdf5 return a list of samples (my whole dataset)"""
    dataset_list = []

    with h5py.File(input_file, "r") as f:
        for group_name in f.keys():
            inner_list = []

            group = f[group_name]
            for dataset_name in group.keys():
                data_array = group[dataset_name][:]
                inner_list.append(data_array)

            dataset_list.append(inner_list)

    return dataset_list


def get_vtk_from_array(data_array: np.ndarray, output_path: str) -> None:
    """Given a np_array (an image) return a vti file"""
    array = data_array
    dimx, dimy, dimz = array.shape
    dimx, dimy, dimz = dimx + 1, dimy + 1, dimz + 1

    image_data = vtk.vtkImageData()
    image_data.SetDimensions(dimx, dimy, dimz)
    image_data.SetSpacing([1, 1, 1])
    image_data.SetOrigin([0, 0, 0])
    image_data.AllocateScalars(vtk.VTK_INT, 1)

    vtk_array = numpy_to_vtk(array.ravel(), deep=True)

    vtk_array.SetName("Spin")
    vtk_array.SetNumberOfComponents(1)

    image_data.GetCellData().SetScalars(vtk_array)
    image_data.GetPointData().RemoveArray(0)

    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(output_path)
    writer.SetInputData(image_data)
    writer.SetDataModeToAscii()
    writer.EncodeAppendedDataOff()
    writer.SetCompressor(None)

    writer.Write()


def get_vti_sample(data_list: List[np.ndarray], output_path: str) -> None:
    """Given a sample (aka vHpdV_ folder), generate vti.i temporal sequence"""
    for i, data_array in enumerate(data_list):
        vti_path = os.path.join(output_path, f"test.vti.{i}")
        get_vtk_from_array(data_array, vti_path)


def main(args):
    ### create hdf5 dataset from vti
    config_file = args.config_file
    reader = vtk.vtkXMLImageDataReader()
    data_list = read_vtk_from_config(args, config_file, reader)

    len_data = len(data_list)
    hdf5_file = os.path.join(args.data_path, f"data_{len_data}.hdf5")
    save_data_to_hdf5(data_list, hdf5_file)

    ### can i come back from hdf5 to vtis?
    hdf5_file = os.path.join(args.data_path, "data_24.hdf5")
    new_vti_path = os.path.join(args.data_path, "test_output")

    read_data = read_data_from_hdf5(hdf5_file)
    print(len(read_data))

    # test: get the first subfolder
    get_vti_sample(read_data[0], new_vti_path)


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
