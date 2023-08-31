# inputs are videos BxChx TxHxW

from argparse import ArgumentParser
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from data import MicroS_Dataset, ImglistToTensor
from vt_unet import VTUNet


def load_data(root_dir, config_file, n_inputs, n_predictions, train=True):
    transform = transforms.Compose(
        [
            # transforms.Resize((224, 224)),
            # transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ImglistToTensor()  # list of PIL images to (CHANNELS x FRAMES x HEIGHT x WIDTH) tensor
        ]
    )
    dataset = MicroS_Dataset(
        root=root_dir,
        config_path=config_file,
        n_frames_input=n_inputs,
        n_frames_output=n_predictions,
        imagefile_template="RenderView1_{:06d}.jpg",
        transform=transform,
        is_train=train,
    )
    return dataset


def main(args):
    root_dir = args.root_dir
    config_file = args.config_file
    n_inputs = args.n_inputs
    n_predictions = args.n_predictions

    # [batch_size, channel, temporal_dim, height, width]
    dummy_x = torch.randn(1, 3, 32, 224, 224)

    model = VTUNet(
        patch_size=(4, 4, 4),
        in_chans=3,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=(2, 7, 7),
    )

    logits = model(dummy_x)
    print(logits.shape)

    # DONE
    # extend dataloader to video (2D+T) DONE
    # current implementation => 1 batch: 1 video of T+K frames (from the folder) DONE
    # TODO
    # understand what to append for prediction task
    dataset = load_data(root_dir, config_file, n_inputs, n_predictions, train=True)

    batch_size = 1
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for inputs, labels in dataloader:
        print(inputs.shape)
        logits = model(inputs)
        print(logits.shape)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "--root_dir",
        type=str,
        default="/home/monicar/swin_transformers/2d_imgs",
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default="/home/monicar/swin_transformers/2d_imgs/config_file",
    )
    parser.add_argument("--n_inputs", type=int, default=10)
    parser.add_argument("--n_predictions", type=int, default=10)

    args = parser.parse_args()
    main(args)
