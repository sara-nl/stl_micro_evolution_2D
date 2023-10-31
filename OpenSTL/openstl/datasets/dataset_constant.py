dataset_parameters = {
    "mmnist": {
        "in_shape": [10, 1, 64, 64],
        "pre_seq_length": 10,
        "aft_seq_length": 10,
        "total_length": 20,
        "data_name": "mnist",
        "metrics": ["mse", "mae", "ssim", "psnr"],
    },
    "mmnist_cifar": {
        "in_shape": [10, 3, 64, 64],
        "pre_seq_length": 10,
        "aft_seq_length": 10,
        "total_length": 20,
        "data_name": "mnist_cifar",
        "metrics": ["mse", "mae", "ssim", "psnr"],
    },
    "pf": {
        "in_shape": [12, 1, 64, 64],
        "pre_seq_length": 12,
        "aft_seq_length": 12,
        "total_length": 24,
        "data_name": "pf",
        "metrics": ["mse", "mae", "ssim", "psnr"],
    },
    "kmc": {
        "in_shape": [12, 1, 64, 64],  # TODO change it
        "pre_seq_length": 12,
        "aft_seq_length": 12,
        "total_length": 24,
        "data_name": "kmc",
        "metrics": ["mse", "mae", "ssim", "psnr"],
    },
}
