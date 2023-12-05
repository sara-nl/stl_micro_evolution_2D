import os
from typing import List, Tuple, Union
import numpy as np
import re
from argparse import ArgumentParser
import tarfile
import tempfile

import vtk
import h5py
from vtk_data_utils import (
    read_vtk_instance,
    convert_vtk_instance_to_numpy,
    extract_top_2D_slice_with_voi,
)

"""
Read raw vti files from compressed tar.gz, 
Extract 2D 3D images,
Save them to HDF5 format. 

To check: I am assuming that every temporal sample is made of 90 time frames

TODO:
single small tar:
- report the missing samples list to a output file
- convert the list of array to tensor or proper format for training? 
scale up:
- parallelize -> I have N config files -> N tar.gz 
"""


def count_samples_in_tar(tar_path: str) -> int:
    n = 0
    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            for member in iter(lambda: tar.next(), None):
                if member.isdir():
                    # print("processing directory:", member.name)
                    n += 1

    except EOFError:
        print("Warning: Reached corrupted section in tar file")

    except tarfile.ReadError:
        print(f"Error reading tar file: {tar_path}")

    return n


def _extract_to_temporary_file(tar_member, tar) -> str:
    """Extract a tar file member to a temporary file and return the file path."""
    with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
        content = tar.extractfile(tar_member).read()
        tmpfile.write(content)
        tmpfile.flush()  # Ensure all data is written to disk
        return tmpfile.name  # Return the path of the temporary file


def _append_temporal_instances(
    temporal_sequence: List[Tuple[int, vtk.vtkImageData]],
    all_sample: List[vtk.vtkImageData],
) -> List[vtk.vtkImageData]:
    if len(temporal_sequence) != 90:
        return all_sample  # do not append corrupted sample

    # temporal_sequence.sort(key=lambda x: x[0])

    for _, instance in sorted(temporal_sequence, key=lambda x: x[0]):
        # TODO do any preprocessing of instances
        # if slicing: 2D slicing, then nparray , otherwise 3D: nparray
        all_sample.append(instance)

    return all_sample


def _append_temporal_instances_with_sorted(
    temporal_sequence: List[Tuple[int, vtk.vtkImageData]],
    all_sample: List[vtk.vtkImageData],
) -> List[vtk.vtkImageData]:
    if len(temporal_sequence) != 90:
        return all_sample  # do not append corrupted sample

    return all_sample + [
        instance for _, instance in sorted(temporal_sequence, key=lambda x: x[0])
    ]


def extract_sample_from_tar(tar_path: str) -> List[List[vtk.vtkImageData]]:
    all_sample = []
    temporal_sequence = []
    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            for member in iter(lambda: tar.next(), None):
                if member is None:
                    continue
                elif member.isdir():
                    print("processing directory:", member.name)
                    if temporal_sequence:
                        # sort and append
                        all_sample = _append_temporal_instances(
                            temporal_sequence, all_sample
                        )
                        # reset temporal_sequence
                        temporal_sequence = []
                elif member.isfile():
                    # Skip members that do not contain '.vti.' in the name
                    if ".vti." not in member.name:
                        continue

                    match = re.search(r"\.vti\.(\d+)", member.name)
                    if match:
                        n = int(match.group(1))

                        # Create a temporary path for vtk reader
                        temp_file_path = _extract_to_temporary_file(member, tar)
                        instance = read_vtk_instance(temp_file_path)

                        # Add the positional index and instance to the list
                        temporal_sequence.append((n, instance))
                    else:
                        print("No valid index found in file name.")

        # Append the last sample
        all_sample = _append_temporal_instances(temporal_sequence, all_sample)

    except EOFError:
        print("Warning: Reached corrupted section in tar file")
        all_sample = _append_temporal_instances(temporal_sequence, all_sample)

    except tarfile.ReadError:
        print(f"Error reading tar file: {tar_path}")

    return all_sample


def save_data_to_hdf5(data_list: List[np.ndarray], output_file: str) -> None:
    with h5py.File(output_file, "w") as hdf_file:
        hdf_file.create_dataset("images", data=np.array(data_list))


def read_data_from_hdf5(input_file: str) -> List[np.ndarray]:
    """
    Given a file hdf5 return a list of samples (my whole dataset)
    Load everything to memory
    """
    with h5py.File(input_file, "r") as hdf_file:
        data_list = list(hdf_file["images"])

    return data_list


def generate_datasets(
    data_list: List[vtk.vtkImageData],
    output_path: str,
    output_name: str,
    slicing: bool = False,
) -> str:
    """
    Generate the datasets for ML training.

    Parameters:
    data_list (List[List[vtk.vtkImageData]]): A list of lists containing vtkImageData objects.
    slicing (bool): If set to True, a 2D dataset is generated by slicing the 3D data.
    hdf5_base_filename (str): Base filename for saving the datasets in HDF5 format.

    Returns:
    File paths of the saved 3D and 2D HDF5 datasets.

    """
    # before saving it to hdf5, vtk objects should be converted to numpy arrays.
    images_3D_list = []
    images_2D_list = []
    for vtkimage in data_list:
        images_3D_list.append(convert_vtk_instance_to_numpy(vtkimage, slicing=False))

        if slicing:
            vtkimage_2D = extract_top_2D_slice_with_voi(vtkimage)
            images_2D_list.append(
                convert_vtk_instance_to_numpy(vtkimage_2D, slicing=True)
            )

    # save to hd5f format
    datapath_3D = os.path.join(output_path, f"{output_name}_3D.h5")
    save_data_to_hdf5(images_3D_list, datapath_3D)

    datapath_2D = None
    if slicing:
        datapath_2D = os.path.join(output_path, f"{output_name}_2D.h5")
        save_data_to_hdf5(images_2D_list, datapath_2D)

    return datapath_2D, datapath_3D


def check_empty_elements_in_hdf5(input_file: str):
    empty_indices = []

    with h5py.File(input_file, "r") as hdf_file:
        images_dataset = hdf_file["images"]
        tmp = hdf_file["images"][1:20]
        for i in range(len(images_dataset)):
            if images_dataset[i].size == 0:  # Checking if the numpy array is empty
                empty_indices.append(i)

    return empty_indices


def main(args):
    tar_path = args.tar_path
    output_path = args.output_path
    output_name = args.output_name

    n = count_samples_in_tar(tar_path)

    print("tar name is", tar_path)
    print("number of samples in tar is: ", n)

    # Extract the samples from tar
    sample_list = extract_sample_from_tar(tar_path)
    print("Number of samples: ", len(sample_list))

    # generate 2D / 3D datasetsgenerate_datasets(
    dataset_2D, dataset_3D = generate_datasets(
        sample_list, output_path, output_name, slicing=True
    )
    print("2D data saved in: ", dataset_2D)
    print("3D data saved in: ", dataset_3D)

    # visualize images:


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--tar_path",
        type=str,
        default="/projects/1/monicar/exp_2.tar.gz",
    )
    parser.add_argument("--output_path", type=str, default="/projects/1/monicar")
    parser.add_argument("--output_name", type=str, default="prova")
    args = parser.parse_args()
    main(args)
