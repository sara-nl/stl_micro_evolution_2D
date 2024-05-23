import os
import os.path as osp
import random
import numpy as np
from PIL import Image
from typing import List
import h5py

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset
from torchvision import transforms

from openstl.datasets.utils import create_loader
from openstl.datasets.image_utils import (
    preprocess_data,
    resize_images,
    pad_images,
    unpad_images,
    pad_temporal_sequence,
    tensor_to_original_array,
)
from openstl.datasets.file_utils import (
    init_file_handlers,
    build_combined_indices_sampled,
    resolve_index,
    build_list_of_samples,
    randomly_sample_experiments_from_datafile,
)


class Experiments_Handler:
    """Kinetic Monte Carlo Potts Microstructure Experiment

    Args:
        data_root (str): Path where the dataset (hdf5) is stored.
        datafile (List[Tuple[str, int]]): List of tuples containing file names and
          their corresponding number of frames per experiment.
        n_frames_input, n_frames_output (int): The number of input and prediction
            video frames.
        is_3D (bool): Whether to use the 3D or 2D data.
        in_shape (tuple of ints): expected input shape of the data. This should be in the format (time frames, channels, height, width).
    """

    def __init__(
        self,
        data_root: str,
        datafile: List[tuple[str, int, int]],
        is_3D: bool = False,
        cache: bool = False,
        seed: bool = None,
    ):
        self.data_root = data_root
        self.datafile = datafile
        self.is_3D = is_3D
        self.cache = cache

        # perform experiment sampling from each file
        (
            self.files_info,
            self.list_of_experiments,
            self.num_of_experiments,
        ) = randomly_sample_experiments_from_datafile(
            self.data_root, self.datafile, random_seed=seed
        )

        if cache:
            self.cached_images = self.cache_images()

    def cache_images(self):
        """
        Cache images from all data files.
        """
        cached_images = {}

        for file_info in self.files_info:
            file_name = os.path.basename(file_info["file_handler"].file_path)
            cached_images[file_name] = file_info["file_handler"].cache_images()

        return cached_images

    def load_images(self, filename, start_idx, end_idx):
        if (
            len(
                (
                    file_handler_collection := [
                        file_info["file_handler"]
                        for file_info in self.files_info
                        if file_info["file_name"] == filename
                    ]
                )
            )
            == 1
        ):
            file_handler = file_handler_collection[0]
        else:
            raise Exception(f"The input includes two files with {filename=}")

        if self.cache:
            file_name = os.path.basename(file_handler.file_path)
            images = self.cached_images[file_name][start_idx:end_idx]
        else:
            images = file_handler.read_images(start_idx, end_idx)
        return images

    def __len__(self):
        return self.num_of_experiments


class KMC_Dataset(Dataset):
    """
    A custom subset that allows access to additional attributes of the parent dataset.
    """

    def __init__(
        self,
        experiments_handler: Experiments_Handler,
        indices,
        n_frames_input,
        n_frames_output,
        in_shape,
        sliding_win,
    ):
        self.experiments_handler = experiments_handler
        self.files_info = self.experiments_handler.files_info
        self.indices = indices
        self.sliding_win = sliding_win
        self.mean = 0.5
        self.std = 0.3

        self.n_frames_input = n_frames_input
        self.n_frames_output = n_frames_output
        self.total_frames_per_sample = n_frames_input + n_frames_output
        self.target_size = (in_shape[-1], in_shape[-1])

        self.idx_mapping, self.dataset_len = build_list_of_samples(
            self.indices,
            self.files_info,
            self.experiments_handler.list_of_experiments,
            self.total_frames_per_sample,
            self.sliding_win,
        )

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, index):
        """
        Since we have a list of file handlers, the method must determine which file a given index belongs to and then fetch the corresponding images.
        """
        if index < 0 or index >= self.dataset_len:
            raise IndexError("Index out of range")

        file_name, start_idx, end_idx = self.idx_mapping[index]
        # file_name = self.files_info[file_index]["file_name"]

        images = self.experiments_handler.load_images(file_name, start_idx, end_idx)

        # _visualize_sample(images, index, exp_name="example")

        input_sample, label_sample = preprocess_data(
            images, self.target_size, self.n_frames_input, self.n_frames_output
        )
        # print("input shape is:", input_sample.shape)

        return input_sample, label_sample


def _visualize_sample(images, index, exp_name="example"):
    for i, frame in enumerate(images):
        if isinstance(frame, torch.Tensor):
            frame = tensor_to_original_array(frame)

        data_uint8 = frame.astype(np.uint8)
        image = Image.fromarray(data_uint8)
        image.save(f"visuals/{exp_name}_{index}_{i}.png")


def create_train_vali_datasets(
    experiments_handler: Experiments_Handler,
    n_frames_input,
    n_frames_output,
    in_shape,
    random_split,
    random_seed,
    train_ratio=0.9,
    sliding_win=0,
):
    """
    Create training and validation dataset after shuffling
    Params: define split ration for training and validation, default: 80% for training
    Random_seed: Optional seed for the random number generator for reproducibility.
    """
    num_experiments = len(experiments_handler)
    indices = np.arange(num_experiments)

    num_train = int(num_experiments * train_ratio)

    if random_split:
        # Set the random seed if specified
        if random_seed is not None:
            np.random.seed(random_seed)

        np.random.shuffle(indices)
        train_indices = indices[:num_train]
        val_indices = indices[num_train:]
    else:
        train_indices = range(0, num_train)
        val_indices = range(num_train, num_experiments)

    train_dataset = KMC_Dataset(
        experiments_handler,
        train_indices,
        n_frames_input,
        n_frames_output,
        in_shape,
        sliding_win=sliding_win,
    )
    vali_dataset = KMC_Dataset(
        experiments_handler,
        val_indices,
        n_frames_input,
        n_frames_output,
        in_shape,
        sliding_win=sliding_win,
    )

    return train_dataset, vali_dataset


def compute_mean_std(loader):
    """Compute the mean and std dev for a DataLoader of images."""
    mean = 0.0
    var = 0.0
    nb_samples = 0
    for data, _ in loader:
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        var += data.var(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std = torch.sqrt(var / nb_samples)
    return mean, std


def load_data(
    batch_size,
    val_batch_size,
    data_root,
    num_workers,
    datafile,
    pre_seq_length,
    aft_seq_length,
    in_shape,
    seed,
    random_split,
    sliding_win,
    distributed,
    use_augment,
    use_prefetcher,
    drop_last,
):
    experiments_handler = Experiments_Handler(
        data_root,
        datafile,
        is_3D=False,
        cache=False,
        seed=seed,
    )

    train_dataset, vali_dataset = create_train_vali_datasets(
        experiments_handler,
        n_frames_input=pre_seq_length,
        n_frames_output=aft_seq_length,
        in_shape=in_shape,
        random_split=random_split,
        random_seed=seed,
        train_ratio=0.8,
        sliding_win=sliding_win,
    )
    # i = 0
    # sample_i = vali_set[i]

    dataloader_train = create_loader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        is_training=True,
        pin_memory=True,
        drop_last=True,
        num_workers=num_workers,
        distributed=distributed,
        use_prefetcher=use_prefetcher,
    )
    dataloader_vali = None
    dataloader_test = create_loader(
        vali_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        is_training=False,
        pin_memory=True,
        drop_last=drop_last,
        num_workers=num_workers,
        distributed=distributed,
        use_prefetcher=use_prefetcher,
    )

    print(
        f"train set size: {len(dataloader_train)*batch_size}, valid size is {len(dataloader_test)*val_batch_size}"
    )
    print(f"num batches: {len(dataloader_train)}, valid size is {len(dataloader_test)}")

    dataset_mean, dataset_std = compute_mean_std(dataloader_train)
    print(f"Dataset Mean: {dataset_mean}, Dataset Std Dev: {dataset_std}")

    dataset_mean, dataset_std = compute_mean_std(dataloader_test)

    print(f"Dataset Mean: {dataset_mean}, Dataset Std Dev: {dataset_std}")

    return dataloader_train, dataloader_vali, dataloader_test


if __name__ == "__main__":
    datafile = [
        ("exp_1_complete_2D.h5", 90, 5),
    ]

    #

    dataloader_train, _, dataloader_test = load_data(
        batch_size=4,
        val_batch_size=1,
        data_root="/home/monicar/prjsp",
        datafile=datafile,
        num_workers=4,
        pre_seq_length=12,
        aft_seq_length=12,
        seed=42,
        sliding_win=0,
        in_shape=[12, 1, 100, 100],
        random_split=True,
        distributed=False,
        use_augment=False,
        use_prefetcher=False,
        drop_last=False,
    )

    first_batch = next(iter(dataloader_test))

    data_loader = dataloader_test
    results = []

    for idx, (batch_x, batch_y) in enumerate(data_loader):
        results.append(
            dict(
                zip(
                    ["inputs_dl", "trues_dl"],
                    [
                        batch_x.numpy(),
                        batch_y.numpy(),
                    ],
                )
            )
        )

    for np_data in ["inputs_dl", "trues_dl"]:
        np.save(osp.join("visuals", np_data + ".npy"), results[np_data])
