import os
from typing import List, Tuple, Union
import numpy as np
import re
from argparse import ArgumentParser
import tarfile
import tempfile

import vtk
from vtk_data_utils import (
    read_vtk_instance,
    convert_vtk_instance_to_numpy,
    extract_top_2D_slice_with_voi,
    save_data_to_hdf5,
)

"""
Read raw vti files from compressed tar.gz, perform preprocessing deterministic operations, and save it npz format. 

DONE:
- read config to get the sample name NOT NEEDED
- access the tar and get the samples or the missing sample DONE - CHECK if correct
- extract 2D or 3D modality at this stage? -> 2D or 3D hdf5 yes 
- do not remove the positional embeddings and do not convert to numpy -> do it later on so if possible 
- save it to hdf5 for be used as dataset during training 

TODO:
single small tar:
- report the missing samples list to a output file
- convert the list of array to tensor or proper format for training? 
scale up:
- parallelize -> I have N config files -> N tar.gz 
"""


def _write_missing_files_to_file(missing_files: List[str], output_path: str):
    """Write the list of missing files to a file."""
    file_path = os.path.join(output_path, "config_missing")
    with open(file_path, "w") as f:
        for file_name in missing_files:
            f.write(file_name + "\n")


def _list_subfolders_in_tar(tar_path):
    subfolders = set()
    with tarfile.open(tar_path, "r:gz") as tar:
        for (
            member
        ) in (
            tar.getmembers()
        ):  # TODO reads everything first - so if file corrupted then i get EOF error - use stream instead
            # Extract the directory name from the member's path
            case_name = member.name.split("/")[0]
            subfolders.add(case_name)
    return list(subfolders)


def _extract_to_temporary_file(tar_member, tar) -> str:
    """Extract a tar file member to a temporary file and return the file path."""
    with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
        content = tar.extractfile(tar_member).read()
        tmpfile.write(content)
        tmpfile.flush()  # Ensure all data is written to disk
        return tmpfile.name  # Return the path of the temporary file


def _get_sample_name(s):
    parts = s.split("/")
    # If the init part contains a prefix, return the second part
    if parts[0].startswith("vHpdV"):
        return parts[0]
    return parts[1]


def _append_temporal_sequence(
    temporal_sequence: List[Tuple[int, vtk.vtkImageData]],
    all_sample: List[List[Tuple[int, vtk.vtkImageData]]],
) -> Tuple[
    List[Tuple[int, vtk.vtkImageData]], List[List[Tuple[int, vtk.vtkImageData]]]
]:
    if temporal_sequence:
        # Sort and process the temporal_sample before appending
        # temporal_sequence.sort(key=lambda x: x[0])
        # sorted_instances = [instance for _, instance in temporal_sequence]
        all_sample.append(temporal_sequence)
        temporal_sequence = []
        return temporal_sequence, all_sample


def extract_sample_from_tar(tar_path: str) -> List[List[vtk.vtkImageData]]:
    all_sample = []
    temporal_sequence = []
    sample_name = None
    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            for member in iter(lambda: tar.next(), None):
                if member is None:
                    continue

                # check my samples
                target_folder = _get_sample_name(member.name)
                if sample_name is None:
                    sample_name = target_folder
                elif sample_name != target_folder:
                    # append and reset
                    temporal_sequence, all_sample = _append_temporal_sequence(
                        temporal_sequence, all_sample
                    )
                    sample_name = target_folder

                print("processing directory:", member.name)

                if member.isfile():
                    # Skip members that do not contain '.vti.' in the name
                    if ".vti." not in member.name:
                        continue

                    match = re.search(r"\.vti\.(\d+)", member.name)
                    if match:
                        n = int(match.group(1))
                        # vtk_name = f"{member.name}/IN1003d.vti.{n}"

                        # Create a temporary path for vtk reader
                        temp_file_path = _extract_to_temporary_file(member, tar)
                        instance = read_vtk_instance(temp_file_path)

                        # Add the positional index and instance to the list
                        temporal_sequence.append((n, instance))
                    else:
                        print("No valid index found in file name.")

        # Append the last sample
        temporal_sequence, all_sample = _append_temporal_sequence(
            temporal_sequence, all_sample
        )

    except EOFError:
        print("Warning: Reached corrupted section in tar file")
        temporal_sequence, all_sample = _append_temporal_sequence(
            temporal_sequence, all_sample
        )

    except tarfile.ReadError:
        print(f"Error reading tar file: {tar_path}")

    return all_sample


def _extract_2D(
    sample: List[vtk.vtkImageData], slice_idx: int = None
) -> List[vtk.vtkImageData]:
    """Convert the original 3D vtkImageData samples to 2D vtkImageData samples."""
    sample_2D = []
    for vtk_data_object in sample:
        instance_2D = extract_top_2D_slice_with_voi(vtk_data_object)
        sample_2D.append(instance_2D)

    return sample_2D


def _convert_to_array(
    sample: List[vtk.vtkImageData], slicing: bool = False
) -> List[np.ndarray]:
    """Convert vtkImageData to np.ndarray"""
    sample_np = []
    for vtk_data_object in sample:
        sample_np.append(
            convert_vtk_instance_to_numpy(vtk_data_object, slicing=slicing)
        )

    return sample_np


def preprocess_data(data_list: List[List[vtk.vtkImageData]]) -> List[List[np.ndarray]]:
    # TODO
    """
    order the samples based on time
    get rid of the positional indeces
    extract 2D by default
    if 3D enabled -> process to np array as well
    """
    all_sample_np = []

    return all_sample_np


def _order_temporal_sample(sample: List[vtk.vtkImageData]) -> List[vtk.vtkImageData]:
    """
    order the samples based on time
    get rid of the positional indeces
    extract 2D by default
    if 3D enabled -> process to np array as well
    """
    sample.sort(key=lambda x: x[0])
    sorted_sample = [instance for _, instance in sample]
    return sorted_sample


def generate_datasets(
    data_list: List[List[vtk.vtkImageData]],
    output_path: str,
    output_name: str,
    generate_3D: bool = False,
) -> Tuple[str, str]:
    """
    Generate the datasets for ML training.

    Parameters:
    data_list (List[List[vtk.vtkImageData]]): A list of lists containing vtkImageData objects.
    slicing (bool): If set to True, a 2D dataset is generated by slicing the 3D data.
    hdf5_base_filename (str): Base filename for saving the datasets in HDF5 format.

    Returns:
    File paths of the saved 3D and 2D HDF5 datasets.

    """
    # if slicing extract 2d data
    # convert to numpy array
    # then save to hdf5
    sample_3D_list = []
    sample_2D_list = []
    i = 0
    for sample in data_list:
        sample = _order_temporal_sample(sample)
        sample_2D = _extract_2D(sample)

        sample_2D_list.append(_convert_to_array(sample_2D, slicing=True))

        if generate_3D:
            sample_3D_list.append(_convert_to_array(sample, slicing=False))

        # print(sample_3D_list[0].shape)

        i = i + 1

    # save to hd5f format
    datapath_2D = os.path.join(output_path, output_name)
    save_data_to_hdf5(sample_2D_list, datapath_2D)

    datapath_3D = None
    if generate_3D:
        datapath_3D = os.path.join(output_path, "KMC_3D.h5")
        save_data_to_hdf5(sample_3D_list, datapath_3D)

    return datapath_2D, datapath_3D


def main(args):
    tar_path = args.tar_path
    output_path = args.output_path
    output_name = args.output_name

    # Extract the samples from tar
    sample_list = extract_sample_from_tar(tar_path)

    print("Number of samples: ", len(sample_list))

    ## generate 2D / 3D datasetsgenerate_datasets(
    dataset_2D, dataset_3D = generate_datasets(
        sample_list, output_path, output_name, generate_3D=False
    )
    print("2D data saved in: ", dataset_2D)
    print("3D data saved in: ", dataset_3D)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--tar_path", type=str, default="/projects/1/monicar/test_corrupted.tar.gz"
    )
    parser.add_argument("--output_path", type=str, default="/projects/1/monicar")
    parser.add_argument("--output_name", type=str, default="test.h5")
    args = parser.parse_args()
    main(args)
