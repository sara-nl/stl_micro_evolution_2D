# Copyright (c) CAIRI AI Lab. All rights reserved


def load_data(
    dataname, batch_size, val_batch_size, num_workers, data_root, dist=False, **kwargs
):
    cfg_dataloader = dict(
        pre_seq_length=kwargs.get("pre_seq_length", 12),
        aft_seq_length=kwargs.get("aft_seq_length", 12),
        in_shape=kwargs.get("in_shape", None),
        distributed=dist,
        use_augment=kwargs.get("use_augment", False),
        use_prefetcher=kwargs.get("use_prefetcher", False),
        drop_last=kwargs.get("drop_last", False),
    )
    # DATASET IS PHASE FIELD
    if dataname == "pf":
        from .dataloader_pf import load_data

        return load_data(
            batch_size, val_batch_size, data_root, num_workers, **cfg_dataloader
        )
    elif "mnist" in dataname:  # 'mmnist', 'mfmnist'
        from .dataloader_moving_mnist import load_data

        cfg_dataloader["data_name"] = kwargs.get("data_name", "mnist")
        return load_data(
            batch_size, val_batch_size, data_root, num_workers, **cfg_dataloader
        )
    elif "kmc" in dataname:
        from .dataloader_kmc import load_data

        return load_data(
            batch_size, val_batch_size, data_root, num_workers, **cfg_dataloader
        )
    else:
        raise ValueError(f"Dataname {dataname} is unsupported")
