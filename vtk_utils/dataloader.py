import os
import h5py
import numpy as np

import torch

from vtk_data_utils import read_data_from_hdf5

# TODO
# import dataset from hdf5 and
#   apply transformation normalization
#   convert List[List[np.ndarray]] to torch.Tensor or any appropriate format
# create class dataset


class Hdf5Dataset(torch.utils.data.Dataset):
    """Dataset for packed HDF5/H5 files to pass to a PyTorch dataloader

    Args:
        path (str): path to h5 file
        mode (str): mode for dataset, choose from training/test/truth/test-truth
        scale_func (str): type of normalization/scaling, choose from fixed_norm/linear/scale
    Returns:
        torch Dataset instance: to pass to a PyTorch dataloader
    """

    def __init__(
        self,
        path,
        mode="training",
        scale_func="pop_norm",
    ):
        if mode not in ("training", "test"):
            raise ValueError('`mode\' must be "training" or "test".')
        self.path = path
        self.mode = mode
        self.scale_func = scale_func

        self.data = read_data_from_hdf5(self.path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]

        return sample


if __name__ == "__main__":
    print("here")
    path = "/home/monicar/esa/output_data/data_24.hdf5"  # Specify the path to your HDF5 file
    dataset = Hdf5Dataset(
        path,
    )

    # Access the data using indexing
    index = 0  # Choose the index of the sample to access
    sample = dataset[index]
    print("Sample:", sample)

    # dataset = H5Dataset(
    #    dataset_path,
    #    hp,
    #    sourcefile_re=sourcefile_re,
    #    scale_func="fixed_norm",
    #    fixed_norm_bounds=(1, 199),
    # )
    # loader = torch.utils.data.DataLoader(
    #    dataset,
    #    batch_size=sqrt_batch_size**2,
    #    num_workers=8,
    #    worker_init_fn=worker_init_fn,
    #    shuffle=True,
    # )
