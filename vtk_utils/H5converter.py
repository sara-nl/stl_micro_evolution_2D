import os
from typing import List, Tuple, Union, Optional
import numpy as np
import re
from argparse import ArgumentParser
import tarfile
import tempfile
import concurrent.futures

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

To check: I am assuming that every temporal sample is made of 90 time frames -> not true always

TODO:
single small tar:
- report the missing samples list to a output file
scale up:
- parallelize -> I have N config files -> N tar.gz 
"""


def count_experiments_in_tar(tar_path: str) -> int:
    """
    Counts number of directories (experiments) in the compressed tar file
    """
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
    """
    Extract a tar file member to a temporary file and return the file path.
    """
    with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
        content = tar.extractfile(tar_member).read()
        tmpfile.write(content)
        tmpfile.flush()  # Ensure all data is written to disk
        return tmpfile.name  # Return the path of the temporary file


def _append_temporal_instances(
    temporal_sequence: List[Tuple[int, vtk.vtkImageData]],
    all_sample: List[vtk.vtkImageData],
) -> List[vtk.vtkImageData]:
    """
    Append and sort vtkImageData instances from a temporal sequence into an aggregated list.

    This function:
    - takes a list containing vtkImageData instances and their associated temporal indices,
    - sorts these instances based on their temporal indices,
    - appends them to a cumulative list of samples (all_sample).

    The temporal index is used solely for sorting purposes and is discarded in the final list.

    Parameters:
    - temporal_sequence: A list of tuples where each tuple consists of an integer (temporal index) and a vtk.vtkImageData instance.
    - all_sample: The existing list of vtk.vtkImageData instances to which the new, sorted instances will be appended.

    Returns:
      An updated list containing the original vtkImageData instances from all_sample, with the new, sorted instances from temporal_sequence appended.
    """
    return all_sample + [
        instance for _, instance in sorted(temporal_sequence, key=lambda x: x[0])
    ]


def _process_directory(
    temporal_sequence: List[vtk.vtkImageData],
    all_sample: List[List[vtk.vtkImageData]],
    num_samples_per_experiment: List[int],
) -> Tuple[List[vtk.vtkImageData], List[List[vtk.vtkImageData]], List[int]]:
    """
    Process the contents of a directory in a TAR file.

    It appends the current temporal sequence to the all_sample list, updates the count of samples per experiment,
    and resets the temporal sequence for the next directory.

    Parameters:
    - temporal_sequence: A list of data instances from the current directory.
    - all_sample: A nested list where each sublist represents the data instances from one directory.
    - num_samples_per_experiment: A list tracking the number of data instances per directory.

    Returns:
      A tuple containing an empty list (reset temporal sequence), the updated all_sample list, and the updated num_samples_per_experiment list.
    """
    all_sample = _append_temporal_instances(temporal_sequence, all_sample)
    num_samples_per_experiment.append(len(temporal_sequence))
    return [], all_sample, num_samples_per_experiment


def _process_file(
    member: tarfile.TarInfo, tar: tarfile.TarFile
) -> Optional[Tuple[int, vtk.vtkImageData]]:
    """
    Process an individual file within a TAR archive.

    This function extracts and processes a single file from the TAR archive,
    specifically a file containing '.vti.' in its name.
    It extracts the index from the file name and reads the corresponding vtkImageData instance.

    Parameters:
    - member: A member of the TAR archive representing a file.
    - tar: The TAR file object being processed.

    Returns:
      A tuple containing the extracted index and the vtkImageData instance, if the file matches the expected format.
      Returns None if the file does not match the format or if there is no valid index found.
    """
    match = re.search(r"\.vti\.(\d+)", member.name)
    if match:
        n = int(match.group(1))

        temp_file_path = _extract_to_temporary_file(member, tar)
        instance = read_vtk_instance(temp_file_path)
        return (n, instance)
    else:
        print("No valid index found in file name.")
        return None


def parallel_extract_sample_from_tar(tar_path: str) -> List[List[vtk.vtkImageData]]:
    raise NotImplementedError


def extract_sample_from_tar(tar_path: str) -> List[List[vtk.vtkImageData]]:
    """
    Extract and process samples from a compressed TAR file.

    Opens a TAR file,
    iterates through its contents,
    extracts data and organizes it into a structured format.
    It also tracks the number of samples per 'experiment' or directory.

    Parameters:
      The file path of the TAR file to be processed.

    Returns:
    - A nested list where each sublist contains vtk.vtkImageData objects from one directory (experiment) in the TAR file.
    - A list containing the count of vtk.vtkImageData objects in each directory (experiment).
    """
    all_sample = []
    temporal_sequence = []
    num_samples_per_experiment = []
    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            for member in iter(lambda: tar.next(), None):
                if member is None:
                    continue
                elif member.isdir():
                    print("processing directory:", member.name)
                    if temporal_sequence:
                        print("len of experiment: ", len(temporal_sequence))
                        # sort and append
                        (
                            temporal_sequence,
                            all_sample,
                            num_samples_per_experiment,
                        ) = _process_directory(
                            temporal_sequence, all_sample, num_samples_per_experiment
                        )
                elif member.isfile() and ".vti." in member.name:
                    # print("processing directory:", member.name)
                    n_instance = _process_file(member, tar)
                    if n_instance:
                        temporal_sequence.append(n_instance)

        # Process the last sequence after exiting the loop
        temporal_sequence, all_sample, num_samples_per_experiment = _process_directory(
            temporal_sequence, all_sample, num_samples_per_experiment
        )

    except EOFError:
        print("Warning: Reached corrupted section in tar file")
        temporal_sequence, all_sample, num_samples_per_experiment = _process_directory(
            temporal_sequence, all_sample, num_samples_per_experiment
        )

    except tarfile.ReadError:
        print(f"Error reading tar file: {tar_path}")

    return all_sample, num_samples_per_experiment


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
    slicing: bool = True,
    generate_3D: bool = False,
) -> str:
    """
    Generate the datasets in a suitable format (hdf5) for ML training.
    Before saving it to hdf5, vtk objects are converted to numpy arrays.

    Parameters:
    - data_list: A list of lists containing vtkImageData objects.
    - slicing (bool): If set to True, a 2D dataset is generated by slicing the 3D data.
    - generate_3D (bool): by default is False, if True, a #D dataset is generated.

    Returns:
      File paths of the saved 3D and 2D HDF5 datasets.

    """
    # before saving it to hdf5, vtk objects should be converted to numpy arrays.
    images_3D_list = []
    images_2D_list = []
    for vtkimage in data_list:
        if generate_3D:
            images_3D_list.append(
                convert_vtk_instance_to_numpy(vtkimage, slicing=False)
            )

        if slicing:
            vtkimage_2D = extract_top_2D_slice_with_voi(vtkimage)
            images_2D_list.append(
                convert_vtk_instance_to_numpy(vtkimage_2D, slicing=True)
            )

    # save to hd5f format
    datapath_3D = None
    if generate_3D:
        datapath_3D = os.path.join(output_path, f"{output_name}_3D.h5")
        save_data_to_hdf5(images_3D_list, datapath_3D)

    datapath_2D = None
    if slicing:
        datapath_2D = os.path.join(output_path, f"{output_name}_2D.h5")
        save_data_to_hdf5(images_2D_list, datapath_2D)

    return datapath_2D, datapath_3D


def check_empty_elements_in_hdf5(input_file: str):
    """
    control function for errors
    """
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

    # n = count_experiments_in_tar(tar_path)

    print("tar name is", tar_path)
    # print("number of samples in tar is: ", n)

    # Extract the samples from tar
    sample_list, num_samples_per_experiment = extract_sample_from_tar(tar_path)
    print("Number of samples: ", len(sample_list))

    for index, num in enumerate(num_samples_per_experiment, start=1):
        print(f"Experiment {index}: {num} samples")

    # File path for the output file
    output_file_path = os.path.join(output_path, f"{output_name}.txt")

    with open(output_file_path, "w") as file:
        for index, num in enumerate(num_samples_per_experiment, start=1):
            file.write(f"Experiment {index}: {num} samples\n")

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
