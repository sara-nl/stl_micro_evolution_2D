import numpy as np
import h5py
import os
from typing import List, Dict

"""Utils functions for handling the frames stored in hd5 file"""


class H5_Handler:
    def __init__(self, file_path):
        self.file_path = file_path
        print("Initializing HDF5 dataset with path: ", file_path)

    def read_images(self, start_idx, end_idx):
        """
        Read and return a range of images from the HDF5 file.

        param start_idx: int, the starting index of the image range to be read.
        param end_idx: int, the ending index of the image range to be read.
        return: A list of images from the specified range.
        """
        with h5py.File(self.file_path, "r") as file:
            images = file["images"][start_idx:end_idx]
        return images

    def get_total_frames(self):
        """
        Get the total number of frames (images) in the HDF5 file.

        return: int, the total number of images in the file.
        """
        with h5py.File(self.file_path, "r") as file:
            num_frames = len(file["images"])
        return num_frames

    def cache_images(self):
        with h5py.File(self.file_path, "r") as file:
            cached_images = list(file["images"])
        return cached_images


def init_file_handlers(data_root: str, datafiles: List[tuple[str, int, int]]):
    """
    Initialize file handlers for each data file.

    :param data_root: str, root directory where data files are stored.
    :param datafiles: list of tuples, each tuple containing the filename and the experiment length.
    :return: list of dictionaries, each containing the file handler and total experiments for each file.
    """
    files_info = []
    for filename, experiment_length, num_experiments_to_sample in datafiles:
        file_path = os.path.join(data_root, filename)
        file_handler = H5_Handler(file_path)

        total_experiments = (
            file_handler.get_total_frames() // experiment_length
        )  # already takes care of the last corrupted experiment
        print(f"total experiment: {total_experiments} file {filename}")

        file_info = {
            "file_name": filename,
            "file_handler": file_handler,
            "total_experiments": total_experiments,
            "num_frames_per_experiment": experiment_length,
            "num_experiments_to_sample": num_experiments_to_sample,
        }
        files_info.append(file_info)

    return files_info


def randomly_sample_experiments_from_datafile(data_root, datafile, random_seed=None):
    """
    Sample a specific number of experiments from the initialized files.

    :param files_info: list of dictionaries, each containing the file handler and total experiments.
    :param num_experiments_to_sample: int, number of experiments to sample.
    :param random_seed: int, seed for random number generator for reproducibility (optional).
    :return: list of dictionaries, updated with the sampled experiment indices for each file.
    """
    file_infos: List[Dict] = init_file_handlers(data_root, datafile)

    if random_seed is not None:
        np.random.seed(random_seed)

    list_of_filename_and_experiment_index = []

    for file_info in file_infos:
        total_experiments = file_info["total_experiments"]
        num_experiments_to_sample = file_info["num_experiments_to_sample"]
        if (
            not num_experiments_to_sample is None
            and num_experiments_to_sample < total_experiments
        ):
            sampled_experiment_indices = np.random.choice(
                total_experiments, num_experiments_to_sample, replace=False
            )
        else:
            sampled_experiment_indices = np.arange(total_experiments)

        file_info["sampled_experiment_indices"] = sampled_experiment_indices
        for exp_index in sampled_experiment_indices.tolist():
            list_of_filename_and_experiment_index.append(
                (file_info["file_name"], exp_index)
            )

    return (
        file_infos,
        list_of_filename_and_experiment_index,
        len(list_of_filename_and_experiment_index),
    )


def build_list_of_samples(
    indices,
    file_infos,
    list_of_experiments,
    total_frames_per_sample,
    sliding_win=0,
):
    """
    Build a combined list of indices for samples from all data files, considering
    only the sampled experiments.

    Returns:
    - List[Tuple[int, int]]: A combined list of tuples, where each tuple contains
      start and end frame indices for each sample across all files.
    """
    samples_indices = []

    for item in indices:
        filename, exp_idx = list_of_experiments[item]
        if (
            len(
                (
                    num_frames_collection := [
                        file_info["num_frames_per_experiment"]
                        for file_info in file_infos
                        if file_info["file_name"] == filename
                    ]
                )
            )
            == 1
        ):
            num_frames_per_experiment = num_frames_collection[0]
        else:
            raise Exception(f"The input includes two files with {filename=}")

        experiment_start_frame = exp_idx * num_frames_per_experiment

        if sliding_win > 0:
            samples = get_samples_sliding_window(
                experiment_start_frame,
                num_frames_per_experiment,
                total_frames_per_sample,
                sliding_win,
            )
        else:
            samples = get_samples_fixed_window(
                experiment_start_frame,
                num_frames_per_experiment,
                total_frames_per_sample,
            )

        # Append file index to each tuple in the samples list
        samples_indices.extend(
            [(filename, start_idx, end_idx) for start_idx, end_idx in samples]
        )

    return samples_indices, len(samples_indices)


def get_samples_sliding_window(
    experiment_start_frame,
    num_frames_per_experiment,
    total_frames_per_sample,
    step_size,
):
    """
    Generate sample indices using a sliding window approach.

    Args:
    - experiment_start_frame (int): The starting frame index of the experiment.
    - num_frames_per_experiment (int): Total number of frames in the experiment.
    - total_frames_per_sample (int): Number of frames in each sample.
    - step_size (int): Step size for the sliding window.

    Returns:
    - List[Tuple[int, int]]: List of tuples with start and end frame indices.
    """
    samples_indices = []
    num_of_samples = max(
        0,
        (num_frames_per_experiment - total_frames_per_sample + step_size) // step_size,
    )
    for sample_idx in range(num_of_samples):
        start_idx = experiment_start_frame + sample_idx * step_size
        end_idx = start_idx + total_frames_per_sample
        if end_idx <= (experiment_start_frame + num_frames_per_experiment):
            samples_indices.append((start_idx, end_idx))
    return samples_indices


def get_samples_fixed_window(
    experiment_start_frame, num_frames_per_experiment, total_frames_per_sample
):
    """
    Generate sample indices using a fixed-size approach.

    Args:
    - experiment_start_frame (int): The starting frame index of the experiment.
    - num_frames_per_experiment (int): Total number of frames in the experiment.
    - total_frames_per_sample (int): Number of frames in each sample.

    Returns:
    - List[Tuple[int, int]]: List of tuples with start and end frame indices.
    """
    samples_indices = []
    len_sample = min(num_frames_per_experiment, total_frames_per_sample)
    num_of_samples = num_frames_per_experiment // len_sample
    for sample_idx in range(num_of_samples):
        start_idx = experiment_start_frame + sample_idx * len_sample
        end_idx = start_idx + len_sample
        samples_indices.append((start_idx, end_idx))
    return samples_indices




def build_list_of_samples_rolling(
    indices, files_info, list_of_experiments, total_frames_per_sample, step_size=1
):
    """
    Build a combined list of indices for samples from all data files, considering
    only the sampled experiments, using a rolling window technique.

    Args:
    - indices (List[int]): Indices of experiments to sample from.
    - files_info (List[Dict]): Information about files, including number of frames per experiment.
    - list_of_experiments (List[Tuple[int, int]]): List of tuples with file index and experiment index.
    - total_frames_per_sample (int): Number of frames in each sample.
    - step_size (int): Step size for the rolling window (default is 1 for maximum overlap).

    Returns:
    - List[Tuple[int, int, int]]: A list of tuples, each containing the file index, start, and end frame indices for each sample.
    - int: Total number of samples generated.
    """
    samples_indices = []
    print("building with rolling win")

    for item in indices:
        file_idx, exp_idx = list_of_experiments[item]
        num_frames_per_experiment = files_info[file_idx]["num_frames_per_experiment"]

        # Calculate the number of samples based on the rolling window approach
        num_of_samples = (
            max(0, num_frames_per_experiment - total_frames_per_sample + step_size)
            // step_size
        )

        # Adjust the starting point of the experiment based on the experiment index
        experiment_start_frame = exp_idx * num_frames_per_experiment
        for sample_idx in range(num_of_samples):
            start_idx = experiment_start_frame + sample_idx * step_size
            end_idx = start_idx + total_frames_per_sample
            if end_idx <= (experiment_start_frame + num_frames_per_experiment):
                samples_indices.append((file_idx, start_idx, end_idx))

    return samples_indices, len(samples_indices)


def build_combined_indices_sampled(files_info, total_frames_per_sample):
    """
    Build a combined list of indices for samples from all data files, considering
    only the sampled experiments.

    Returns:
    - List[Tuple[int, int]]: A combined list of tuples, where each tuple contains
      start and end frame indices for each sample across all files.
    """
    combined_indices = []

    for file_index, file_info in enumerate(files_info):
        # file_handler = file_info["file_handler"]
        num_frames_per_experiment = file_info["num_frames_per_experiment"]

        # Determine sample length based on the number of frames per experiment
        len_sample = min(num_frames_per_experiment, total_frames_per_sample)

        for exp_idx in file_info["sampled_experiment_indices"]:
            experiment_start_frame = exp_idx * num_frames_per_experiment
            # Calculate number of samples that can be extracted from this experiment
            num_of_samples = num_frames_per_experiment // len_sample

            for sample_idx in range(num_of_samples):
                start_idx = experiment_start_frame + sample_idx * len_sample
                end_idx = start_idx + len_sample
                combined_indices.append((file_index, start_idx, end_idx))

    with open("print_combined_indices.txt", "w") as file:
        for item in combined_indices:
            # Write each item to the file, followed by a newline character
            file.write(f"{item}\n")

    return combined_indices, len(combined_indices)



if __name__ == "__main__":
    datafile = [
        ("exp_1_complete_2D.h5", 90, 4000),
        ("exp_2_complete_2D.h5", 55, 4000),
        ("exp_3_complete_2D.h5", 90, 4000),
        # ("exp_4_len_15_2D.h5", 15, 4000),
        # ("exp_4_len_28_2D.h5", 28, 4000),
        # ("exp_5_len_28_2D.h5", 28, 4000),
        # ("exp_6_len_30_2D.h5", 30, 4000),
        # ("exp_6_len_28_2D.h5", 28, 4000),
    ]

    exp_indices = [(1, 0, 90)]

    assert [(0, 24)] == get_samples_fixed_window(
        experiment_start_frame=0,
        num_frames_per_experiment=28,
        total_frames_per_sample=24,
    )
    assert [(10, 34), (34, 58)] == get_samples_fixed_window(
        experiment_start_frame=10,
        num_frames_per_experiment=55,
        total_frames_per_sample=24,
    )
