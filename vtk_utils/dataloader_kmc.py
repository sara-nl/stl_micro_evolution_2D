import os
import random
import numpy as np
from PIL import Image

import h5py
import cv2
import pyvista as pv

from vtk_data_utils import render_3D_from_numpy, render_2D_from_numpy

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset
from torchvision import transforms


# TODO
# class is modified to handle more h5 data
# test if the idxs are correctly built
# train on more data


class H5_Handler:
    def __init__(self, file_path):
        self.file_path = file_path
        print("Initializing HDF5 dataset with path: ", file_path)

    def read_images(self, start_idx, end_idx):
        with h5py.File(self.file_path, "r") as file:
            images = file["images"][start_idx:end_idx]
        return images

    def get_total_frames(self):
        with h5py.File(self.file_path, "r") as file:
            num_frames = len(file["images"])
        return num_frames

    def cache_images(self):
        with h5py.File(self.file_path, "r") as file:
            cached_images = list(file["images"])
        return cached_images


class KMC_Dataset(Dataset):
    """Kinetic Monte Carlo Potts Microstructure Dataset

    Args:
        data_root (str): Path to the dataset.
        n_frames_input, n_frames_output (int): The number of input and prediction
            video frames.
        transform (None): Apply transformation.
        is_3D (bool): Whether to use the 3D or 2D data.
    """

    def __init__(
        self,
        data_root,
        datafiles,  # TODO modified here
        n_frames_input=10,
        n_frames_output=10,
        is_3D=False,
        cache=False,
        in_shape=[12, 1, 128, 128],
    ):
        self.data_root = data_root
        self.datafiles = datafiles
        self.n_frames_input = n_frames_input
        self.n_frames_output = n_frames_output
        self.total_frames_per_sample = n_frames_input + n_frames_output
        self.is_3D = is_3D
        self.cache = cache
        self.target_size = (in_shape[-1], in_shape[-1])
        self.mean = None
        self.std = None

        # Initialize file handlers and store frames info
        self.files_info = self.init_file_handlers(data_root, datafiles)
        self.idx_mapping, self.dataset_len = self._build_combined_indices()

        if cache:
            self.cached_images = self.cache_images()

    def init_file_handlers(self, data_root, datafiles):
        """
        Initialize file handlers for each data file and store their frames information.

        Parameters:
        - data_root (str): The root directory where the data files are located.
        - datafiles (List[Tuple[str, int]]): List of tuples containing file names and
          their corresponding number of frames per experiment.

        Returns:
        - List[Dict]: A list of dictionaries, each containing the file handler,
          the number of frames per experiment, and other relevant information for each file.
        """
        files_info = []
        for filename, num_frames in datafiles:
            file_path = os.path.join(data_root, filename)
            file_handler = H5_Handler(file_path)

            file_info = {
                "file_handler": file_handler,
                "num_frames_per_experiment": num_frames,
            }
            files_info.append(file_info)

        return files_info

    def _build_combined_indices(self):
        """
        Build a combined list of indices for samples from all data files.

        Returns:
        - List[Tuple[int, int]]: A combined list of tuples, where each tuple contains
          start and end frame indices for each sample across all files.
        """
        combined_indices = []
        offset = 0  # Offset to adjust indices for each file

        for file_info in self.files_info:
            num_frames_per_experiment = file_info["num_frames_per_experiment"]
            file_handler = file_info["file_handler"]

            num_of_samples = (
                num_frames_per_experiment // self.total_frames_per_sample
            )  # num of samples that i could get from a directory (experiment)
            num_of_experiments = (
                file_handler.get_total_frames() // num_frames_per_experiment
            )

            for exp_idx in range(num_of_experiments - 1):
                for sample_idx in range(num_of_samples):
                    start_idx = (
                        offset
                        + exp_idx * num_frames_per_experiment
                        + sample_idx * self.total_frames_per_sample
                    )
                    end_idx = start_idx + self.total_frames_per_sample

                    combined_indices.append((start_idx, end_idx))

            # Update offset for the next file
            offset += file_handler.get_total_frames()

        return combined_indices, len(combined_indices)

    def cache_images(self):
        """
        Cache images from all data files.
        """
        cached_images = {}

        for file_info in self.files_info:
            file_name = os.path.basename(file_info["file_handler"].file_path)
            cached_images[file_name] = file_info["file_handler"].cache_images()

        return cached_images

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, index):
        """
        Since we have a list of file handlers, the method must determine which file a given index belongs to and then fetch the corresponding images.
        """
        if index < 0 or index >= self.dataset_len:
            raise IndexError("Index out of range")

        start_idx, end_idx = self.idx_mapping[index]
        file_index, local_start_idx, local_end_idx = self.resolve_index(
            start_idx, end_idx
        )

        if self.cache:
            file_name = os.path.basename(
                self.files_info[file_index]["file_handler"].file_path
            )
            images = self.cached_images[file_name][local_start_idx:local_end_idx]
        else:
            images = self.files_info[file_index]["file_handler"].read_images(
                local_start_idx, local_end_idx
            )

        input_sample, label_sample = self.preprocess_data(images)

        return input_sample, label_sample

    def resolve_index(self, global_start_idx, global_end_idx):
        """
        Resolve a global index to a specific file and its local index range.

        Parameters:
        - global_start_idx (int): The global start index.
        - global_end_idx (int): The global end index.

        Returns:
        - Tuple[int, int, int]: A tuple containing the file index, local start index, and local end index.
        """
        offset = 0
        for file_idx, file_info in enumerate(self.files_info):
            file_total_frames = file_info["file_handler"].get_total_frames()
            if global_start_idx < offset + file_total_frames:
                local_start_idx = global_start_idx - offset
                local_end_idx = min(global_end_idx - offset, file_total_frames)
                return file_idx, local_start_idx, local_end_idx
            offset += file_total_frames

        raise IndexError("Global index out of range")

    def preprocess_data(self, images):
        """
        Convert to numpy arrays and add channel dimension
        Normalize the range 0-199
        Convert to PyTorch tensors
        Apply transformations if any
        """
        # rescale
        resized_images = (
            self.resize_images(images, new_size=self.target_size)
            if self.target_size[-1] < images.shape[-1]
            else self.pad_images(images, new_size=self.target_size)
        )
        # normalize
        if resized_images.max() > 1.0:
            norm_images = resized_images.astype(np.float32) / 199.0

        norm_images = np.expand_dims(norm_images, axis=1)

        input_sample = norm_images[: self.n_frames_input]
        label_sample = norm_images[self.n_frames_input :]

        # visualize -> come back
        self.tensor_to_original_int_array(input_sample)

        return (
            torch.from_numpy(input_sample).unsqueeze(1),
            torch.from_numpy(label_sample).unsqueeze(1),
        )

    @staticmethod
    def resize_images(images, new_size=(96, 96)):
        if images.shape[-1] == new_size[-1]:
            return images
        resized_images = []
        for image in images:
            pil_img = Image.fromarray(image)
            resized_img = pil_img.resize(new_size, Image.ANTIALIAS)
            resized_images.append(np.array(resized_img))
        return np.array(resized_images)

    @staticmethod
    def pad_images(images, new_size=(128, 128), image_mode="L"):
        if images.shape[-1] == new_size[-1]:
            return images
        padded_images = []
        for image in images:
            # Create a new image with the desired size and black background
            # 'L' mode is for (8-bit pixels, black and white)
            padded_img = Image.new(image_mode, new_size, color=0)

            pil_img = Image.fromarray(image)
            # Calculate padding sizes
            width, height = pil_img.size
            left = (new_size[0] - width) // 2
            top = (new_size[1] - height) // 2

            padded_img.paste(pil_img, (left, top))
            padded_images.append(np.array(padded_img))

        return np.array(padded_images)

    def tensor_to_original_int_array(self, array):
        """
        Send to CPU
        Reverse the normalization, remove channel dimension, convert to numpy int
        """
        # tensor = tensor.cpu()

        # numpy_array = array * 199.0
        numpy_array = array.squeeze(axis=1)
        # numpy_array = numpy_array.astype(np.int32)

        input_rescaled = self.unpad_images(numpy_array, original_size=(100, 100))
        input_denorm = input_rescaled * 199.0
        render_2D_from_numpy(input_denorm[5], "./provadenorm.png")
        return numpy_array

    def unpad_images(self, padded_images, original_size=(100, 100)):
        # Assuming the padding was added equally on all sides
        # Calculate the starting and ending indices to slice
        pad_height = (padded_images.shape[1] - original_size[0]) // 2
        pad_width = (padded_images.shape[2] - original_size[1]) // 2

        # Adjust if the padding added an extra pixel for odd differences
        pad_height_extra = (padded_images.shape[1] - original_size[0]) % 2
        pad_width_extra = (padded_images.shape[2] - original_size[1]) % 2

        # Slice out the padding to get back the original images
        unpadded_images = padded_images[
            :,  # Keep all images/channels in the first dimension
            pad_height : padded_images.shape[1]
            - pad_height
            - pad_height_extra,  # Remove padding from the second dimension
            pad_width : padded_images.shape[2]
            - pad_width
            - pad_width_extra,  # Remove padding from the third dimension
        ]

        return unpadded_images


class KMC_Subset(Subset):
    """
    A custom subset that allows access to additional attributes of the parent dataset.
    """

    def __init__(self, dataset, indices):
        super(KMC_Subset, self).__init__(dataset, indices)

    @property
    def mean(self):
        return self.dataset.mean

    @property
    def std(self):
        return self.dataset.std

    def tensor_to_original_int_array(self, tensor):
        return self.dataset.tensor_to_original_int_array(tensor)


def create_loader(
    dataset,
    batch_size,
    shuffle=True,
    is_training=False,
    mean=None,
    std=None,
    num_workers=1,
    num_aug_repeats=0,
    input_channels=1,
    use_prefetcher=False,
    distributed=False,
    pin_memory=False,
    drop_last=False,
    fp16=False,
    collate_fn=None,
    persistent_workers=True,
    worker_seeding="all",
):
    sampler = None

    if collate_fn is None:
        collate_fn = torch.utils.data.dataloader.default_collate
    loader_class = torch.utils.data.DataLoader

    loader_args = dict(
        batch_size=batch_size,
        shuffle=shuffle
        and (not isinstance(dataset, torch.utils.data.IterableDataset))
        and sampler is None
        and is_training,
        num_workers=num_workers,
        sampler=sampler,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=persistent_workers,
    )
    try:
        loader = loader_class(dataset, **loader_args)
    except TypeError:
        loader_args.pop("persistent_workers")  # only in Pytorch 1.7+
        loader = loader_class(dataset, **loader_args)

    return loader


def _split_train_vali(dataset, train_ratio=0.8):
    """
    Create training and validation dataset
    Params: define split ration for training and validation, default: 80% for training
    """
    num_train = int(len(dataset) * train_ratio)
    num_val = len(dataset) - num_train

    # Generate indices for training and validation sets
    train_indices = range(0, num_train)
    val_indices = range(num_train, len(dataset))

    train_dataset = KMC_Subset(dataset, train_indices)
    vali_dataset = KMC_Subset(dataset, val_indices)

    return train_dataset, vali_dataset


def load_data(
    batch_size,
    val_batch_size,
    data_root,
    num_workers=4,
    datafile="kmc/exp_1_complete_2D.h5",
    pre_seq_length=12,
    aft_seq_length=12,
    in_shape=[12, 1, 100, 100],
    num_frames_per_experiment=90,
    distributed=False,
    use_augment=False,
    use_prefetcher=False,
    drop_last=False,
):
    dataset = KMC_Dataset(
        data_root,
        datafile,
        n_frames_input=pre_seq_length,
        n_frames_output=aft_seq_length,
        is_3D=False,
        cache=False,
        in_shape=in_shape,  #
    )

    train_set, vali_set = _split_train_vali(dataset, train_ratio=0.9)

    dataloader_train = create_loader(
        train_set,
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
        vali_set,
        batch_size=val_batch_size,
        shuffle=False,
        is_training=False,
        pin_memory=True,
        drop_last=drop_last,
        num_workers=num_workers,
        distributed=distributed,
        use_prefetcher=use_prefetcher,
    )

    return dataloader_train, dataloader_vali, dataloader_test


if __name__ == "__main__":
    datafiles = [
        ("exp_1_complete_2D.h5", 90),
        ("exp_1_complete_2D.h5", 45),
        ("exp_3_complete_2D.h5", 90),
    ]
    dataset = KMC_Dataset(
        data_root="/projects/1/monicar",
        datafiles=datafiles,
        n_frames_input=10,
        n_frames_output=10,
        is_3D=False,
        cache=False,
    )
    print("creating dataset")

    index = len(dataset) - 1  # Choose the index of the sample to access
    input_sample, label_sample = dataset[index]

    train_dataset, valid_dataset = _split_train_vali(dataset, train_ratio=0.8)

    for i in range(100):
        input_sample, label_sample = train_dataset[i]

    train_loader = create_loader(
        dataset,
        batch_size=6,
        shuffle=True,
        is_training=True,
        pin_memory=True,
        drop_last=True,
        num_workers=1,
        distributed=False,
        use_prefetcher=False,
    )

    first_batch = next(iter(train_loader))

    print("done")
