import os
import random
import numpy as np
from PIL import Image

import h5py

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset
from torchvision import transforms

from openstl.datasets.utils import create_loader


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
        datafile,
        n_frames_input=10,
        n_frames_output=10,
        is_3D=False,
        cache=False,
        in_shape=[12, 1, 128, 128],
        num_frames_per_experiment=90,  # this number is not always true
    ):
        self.data_root = data_root
        self.n_frames_input = n_frames_input
        self.n_frames_output = n_frames_output
        self.total_frames_per_sample = n_frames_input + n_frames_output
        self.is_3D = is_3D
        self.cache = cache
        self.target_size = (in_shape[-1], in_shape[-1])
        self.mean = None
        self.std = None

        self.num_frames_per_experiment = num_frames_per_experiment
        self.num_of_samples_per_experiment = (
            self.num_frames_per_experiment // self.total_frames_per_sample
        )

        self.data_h5 = os.path.join(data_root, datafile)
        self.file_handler = H5_Handler(self.data_h5)

        self.num_tot_frames = self.file_handler.get_total_frames()
        if cache:
            self.cached_images = self.file_handler.cache_images()

        self.idx_mapping, self.dataset_len = self._build_frames_indices()

    def _build_frames_indices(self):
        """Build a dictionary of sample indices."""
        self.num_of_experiments = self.num_tot_frames // self.num_frames_per_experiment
        idxs = [
            (
                index * self.num_frames_per_experiment
                + sample_index * self.total_frames_per_sample,
                index * self.num_frames_per_experiment
                + (sample_index + 1) * self.total_frames_per_sample,
            )
            for index in range(self.num_of_experiments)
            for sample_index in range(self.num_of_samples_per_experiment)
        ]
        return idxs, len(idxs)

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, index):
        if index < 0 or index >= self.dataset_len:
            raise IndexError("Index out of range")

        start_idx, end_idx = self.idx_mapping[index]

        images = (
            self.cached_images[start_idx:end_idx]
            if self.cache
            else self.file_handler.read_images(start_idx, end_idx)
        )

        # if self.cache:
        #     images = self.cached_images[start_idx:end_idx]
        # else:
        #     images = self.file_handler.read_images(start_idx, end_idx)

        input_sample, label_sample = self.preprocess_data(images)

        return input_sample, label_sample

    def preprocess_data(self, images):
        """
        Convert to numpy arrays and add channel dimension
        Normalize the range 1-199
        Convert to PyTorch tensors
        Apply transformations if any
        """
        resized_images = (
            self.resize_images(images, new_size=self.target_size)
            if self.target_size[-1] < images.shape[-1]
            else self.pad_images(images, new_size=self.target_size)
        )

        input_array = np.array(resized_images[: self.n_frames_input]).astype(np.float32)
        label_array = np.array(resized_images[self.n_frames_input :]).astype(np.float32)

        normalized_input_array = (input_array - 1) / 198.0
        normalized_label_array = (label_array - 1) / 198.0

        return (
            torch.from_numpy(normalized_input_array).unsqueeze(1),
            torch.from_numpy(normalized_label_array).unsqueeze(1),
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

    def tensor_to_original_int_array(self, tensor):
        """
        Send to CPU
        Reverse the normalization, remove channel dimension, convert to numpy int
        """
        tensor = tensor.cpu()

        numpy_array = tensor.numpy() * 198.0 + 1
        numpy_array = numpy_array.squeeze(axis=1)
        numpy_array = numpy_array.astype(np.int32)

        return numpy_array


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

    train_set, vali_set = _split_train_vali(dataset, train_ratio=0.8)

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
    dataloader_train, _, dataloader_test = load_data(
        batch_size=16,
        val_batch_size=4,
        data_root="../../data/",
        num_workers=4,
        pre_seq_length=10,
        aft_seq_length=10,
    )

    print(len(dataloader_train), len(dataloader_test))
    for item in dataloader_train:
        print(item[0].shape, item[1].shape)
        break
    for item in dataloader_test:
        print(item[0].shape, item[1].shape)
        break
