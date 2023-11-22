"""
last modified on Nov 20

Utilities for handling VTK objects
"""

import os
from typing import List, Tuple
import numpy as np
from argparse import ArgumentParser

import h5py
import vtk

import pyvista as pv

from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk


# every subfolder is a sample: a video, an instance of my time evolution - this is a sample
# every sample has a list of images:
# every image is a temporal scrrenshot of my evolution - single vtk instance - ex vtk.0 is my image at time 0

"""Collection of methods for reading the "subfolder/*.vti.*" generated from spparks"""
"""generate hdf5 dataset from all subfolders"""
"""also generate vti back from hdf5 as test (and for future use)"""


def render_3D_from_numpy(numpy_array: np.ndarray, filename: str = "visual_np.png"):
    if numpy_array.shape != (100, 100, 50):
        raise ValueError(
            "The numpy array does not have the expected shape (100, 100, 50)."
        )

    image = pv.ImageData((101, 101, 51))

    image.spacing = (1, 1, 1)
    image.origin = (0, 0, 0)

    image.cell_data["Spin"] = numpy_array.flatten(order="C")

    pv.start_xvfb()
    plotter = pv.Plotter(off_screen=True)

    plotter.add_volume(image, cmap="viridis", scalar_bar_args={"title": "Spin"})

    plotter.show(auto_close=False)
    plotter.screenshot(filename)
    plotter.close()


def render_2D_from_numpy(numpy_array: np.ndarray, filename: str = "visual_np.png"):
    grid = pv.ImageData()  # pv.UniformGrid()

    # Set the dimensions of the grid (adding 1 because grid points define the cells)
    grid.dimensions = numpy_array.shape[0] + 1, numpy_array.shape[1] + 1, 1

    grid.spacing = (1, 1, 1)
    grid.origin = (0, 0, 0)

    # Flatten the array and assign it to the cell data
    grid.cell_data["Spin"] = numpy_array.flatten(order="C")

    pv.start_xvfb()
    plotter = pv.Plotter(off_screen=True)

    plotter.add_mesh(grid, cmap="viridis", show_edges=False)

    plotter.show(auto_close=False)
    plotter.screenshot(filename)
    plotter.close()


def render_vtk(vtk_data_object: vtk.vtkImageData, filename: str = "visualization.png"):
    """
    Visualize a vtkImageData object. Automatically handles both 2D and 3D data.

    Parameters:
    vtk_data_object (vtk.vtkImageData): The vtkImageData object to visualize.
    """
    dims = vtk_data_object.GetDimensions()

    pv.start_xvfb()  # Start an X virtual framebuffer
    pv_data = pv.wrap(vtk_data_object)
    plotter = pv.Plotter(off_screen=True)

    # Check if the data is 2D or 3D and visualize accordingly
    if 1 in dims:
        # Data is 2D
        plotter.add_mesh(pv_data, cmap="viridis")  # Use a suitable colormap
    else:
        # Data is 3D
        plotter.add_volume(pv_data)

    plotter.show(auto_close=False)
    plotter.screenshot(filename)
    plotter.close()


def extract_top_2D_slice_with_voi(
    vtk_data_object: vtk.vtkImageData,
) -> vtk.vtkImageData:
    """
    Extract the top 2D slice (highest Z-index) from the 3D vtkImageData object using vtkExtractVOI.

    Parameters:
    vtk_data_object (vtk.vtkImageData): The 3D image data to slice.

    Returns:
    vtk.vtkImageData: The extracted top 2D image slice.
    """
    # Get dimensions of the input data
    dims = vtk_data_object.GetDimensions()

    # Extract the VOI for the top slice
    extract_voi = vtk.vtkExtractVOI()
    extract_voi.SetInputData(vtk_data_object)
    extract_voi.SetVOI(0, dims[0] - 1, 0, dims[1] - 1, dims[2] - 1, dims[2] - 1)
    extract_voi.Update()

    return extract_voi.GetOutput()


def convert_vtk_instance_to_numpy(
    vtk_data_object: vtk.vtkImageData, array_name: str = "Spin", slicing: bool = False
) -> np.ndarray:
    """
    Convert single vti.n data object to numpy array, handling both 2D and 3D data.
    """

    # Decide between cell data and point data
    if vtk_data_object.GetCellData().GetNumberOfArrays() > 0:
        data = vtk_data_object.GetCellData()
    else:
        data = vtk_data_object.GetPointData()

    # Use the specified array or the first available array
    if array_name:
        vtk_array = data.GetArray(array_name)
    else:
        vtk_array = data.GetArray(0)

    if vtk_array is None:
        raise ValueError("No suitable data array found in vtkImageData.")

    # get additional info (not needed for now)
    spacing = vtk_data_object.GetSpacing()
    range_min, range_max = vtk_array.GetRange()  # useful for normalize my data
    num_components = vtk_array.GetNumberOfComponents()

    np_array = vtk_to_numpy(vtk_array)

    # Subtract 1 to get the correct dimensions for cell data
    dims = vtk_data_object.GetDimensions()

    # Handle 3D cell data
    if len(dims) == 3 and data == vtk_data_object.GetCellData():
        dims = (dims[0] - 1, dims[1] - 1, dims[2] - 1)

    # Handle 2D data (slices)
    if slicing:  # if 0 in dims:
        dims = tuple(d for d in dims if d > 1)

    np_array = np_array.reshape(dims)

    return np_array


def read_vtk_instance(filename: str) -> vtk.vtkImageData:
    """Read single vti.n file"""
    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(filename)
    reader.Update()
    vtk_data_object = reader.GetOutput()

    return vtk_data_object


def read_vtk_sample(path: str) -> List[np.ndarray]:
    """We are inside a vHpdV_ folder (that's my sample): read the temporal sequence"""
    """Files stored in file sys"""
    # Iterate over the range of numbers from vti.0 to vti.N
    temporal_sample = []
    n = 0
    while True:
        file_path = os.path.join(path, f"IN1003d.vti.{n}")
        if not os.path.isfile(file_path):
            break

        instance = read_vtk_instance(file_path)

        # visualization debugs
        instance_np = convert_vtk_instance_to_numpy(instance, slicing=False)

        # instance_2D = extract_top_2D_slice_with_voi(instance)
        # instance_np_2D = convert_vtk_instance_to_numpy(instance_2D, slicing=True)

        # render_vtk(instance_2D, f"instance_np_{n}.png")
        # render_2D_from_numpy(instance_np_2D, f"instance_np_{n}.png")
        # render_3D_from_numpy(instance_np, f"instance_np3_{n}.png")

        temporal_sample.append(instance)

        n += 1

    return temporal_sample


def read_vtk_from_path(data_path: str, config_file: str) -> List[List[np.ndarray]]:
    """
    Read the samples path from config file

    the np.ndarray shape is (100,100,50)
    """

    data_list = []
    with open(config_file, "r") as file:
        for line in file:
            subfolder = line.strip()
            path = os.path.join(data_path, subfolder)

            sample = read_vtk_sample(path)  # sample[0].shape = (100,100,50)

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
    data_path = args.data_path
    data_list = read_vtk_from_path(data_path, config_file)

    len_data = len(data_list)

    instance = data_list[0][0]
    # visualize_vtk(instance)

    # hdf5_file = os.path.join(args.data_path, f"data_{len_data}.hdf5")
    # save_data_to_hdf5(data_list, hdf5_file)

    ### can i come back from hdf5 to vtis?
    # hdf5_file = os.path.join(args.data_path, "data_24.hdf5")
    # new_vti_path = os.path.join(args.data_path, "test_output")

    # read_data = read_data_from_hdf5(hdf5_file)
    # print(len(read_data))

    # test: get the first subfolder
    # get_vti_sample(read_data[0], new_vti_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--data_path", type=str, default="/projects/1/monicar/experiment_1"
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default="/projects/1/monicar/experiment_1/config_exp1",
    )
    args = parser.parse_args()
    main(args)
