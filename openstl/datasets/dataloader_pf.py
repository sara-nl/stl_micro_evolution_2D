import os
import random
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from openstl.datasets.utils import create_loader


class PF_Dataset(Dataset):
    """Phase Field Microstructure Dataset <https://archive.materialscloud.org/record/2022.156>

    Args:
        data_root (str): Path to the dataset.
        n_frames_input, n_frames_output (int): The number of input and prediction
            video frames.
        transform (None): Apply transformation.
        is_train (bool): Whether to use the train or test set.
    """

    def __init__(
        self,
        data_root,
        n_frames_input=12,
        n_frames_output=12,
        transform=None,
        is_train=True,
    ):
        super(PF_Dataset, self).__init__()
        self.data_root = data_root
        self.is_train = is_train
        self.n_frames_input = n_frames_input
        self.n_frames_output = n_frames_output
        self.n_frames_total = self.n_frames_input + self.n_frames_output
        self.transform = transform
        self.mean = None
        self.std = None

        print(data_root)
        if self.is_train:
            self.data_npz = os.path.join(data_root, "pf/train.npz")
        else:
            self.data_npz = os.path.join(data_root, "pf/valid.npz")

        self.videos = self._sample_frames_from_npz(
            self.data_npz, self.n_frames_total, key="input_raw_data"
        )

        # if the data is in [0, 255], rescale it into [0, 1]
        if self.videos.max() > 1.0:
            self.videos = self.videos.astype(np.float32) / 255.0

    def _sample_frames_from_npz(self, npz_path, num_frames=20, key="input_raw_data"):
        with np.load(npz_path) as data:
            all_frames = data[key]

        # Calculate how many full samples can be formed and take only the frames that fit into full samples
        num_videos = len(all_frames) // num_frames
        all_frames = all_frames[: num_videos * num_frames]

        # Reshape the frames into videos of size (num_videos, num_frames, size)
        videos = all_frames.reshape((-1, num_frames) + all_frames.shape[1:])

        return videos

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        video = self.videos[idx]
        input_sample = torch.from_numpy(video[: self.n_frames_input]).float()
        label_sample = torch.from_numpy(video[self.n_frames_input :]).float()
        return input_sample, label_sample


def load_data(
    batch_size,
    val_batch_size,
    data_root,
    num_workers=4,
    pre_seq_length=12,
    aft_seq_length=12,
    in_shape=[12, 1, 64, 64],
    distributed=False,
    use_augment=False,
    use_prefetcher=False,
    drop_last=False,
):
    img_width = in_shape[-1] if in_shape is not None else 64
    input_param = {
        "paths": os.path.join(data_root, "pf"),
        "image_width": img_width,
        "minibatch_size": batch_size,
        "seq_length": (pre_seq_length + aft_seq_length),
        "input_data_type": "float32",
        "name": "pf",
    }

    train_set = PF_Dataset(
        data_root,
        n_frames_input=pre_seq_length,
        n_frames_output=aft_seq_length,
        transform=None,
        is_train=True,
    )
    vali_set = PF_Dataset(
        data_root,
        n_frames_input=pre_seq_length,
        n_frames_output=aft_seq_length,
        transform=None,
        is_train=False,
    )

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
