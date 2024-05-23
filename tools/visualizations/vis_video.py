import argparse
import os
import numpy as np

from openstl.datasets import dataset_parameters
from openstl.utils import (
    show_video_line_combined,
    show_video_line,
)


def min_max_norm(data):
    _min, _max = np.min(data), np.max(data)
    data = (data - _min) / (_max - _min)
    return data


def _merge_truth_pred(truth, pred):
    """
    Merge truth and predicted sequences into a single image, with truth on top and predicted below.
    """
    return np.concatenate([truth, pred], axis=2)


def main(args):
    assert (
        args.dataname is not None and args.work_dirs is not None
    ), "The name of dataset and the path to work_dirs are required"

    # setup results of the STL methods
    base_dir = args.work_dirs
    assert os.path.isdir(args.work_dirs)

    method_list = [args.work_dirs.split("/")[-1]]
    base_dir = base_dir.split(method_list[0])[0]

    use_rgb = (
        False
        if args.dataname in ["mfmnist", "mmnist", "kth20", "kth", "kth40"]
        else True
    )
    config = args.__dict__
    config.update(dataset_parameters[args.dataname])

    if not os.path.isdir(args.save_dirs):
        os.mkdir(args.save_dirs)
    if args.vis_channel != -1:  # choose a channel
        c_surfix = f"_C{args.vis_channel}"
        assert (
            0 <= args.vis_channel <= config["in_shape"][1]
        ), "Channel index out of range"
    else:
        c_surfix = ""

    # loading results
    predicts_dict, inputs_dict, trues_dict = dict(), dict(), dict()
    empty_keys = list()
    for method in method_list:
        try:
            predicts_dict[method] = np.load(
                os.path.join(base_dir, method, "saved/preds.npy")
            )
        except:
            empty_keys.append(method)
            print("Failed to read the results of", method)
    assert len(predicts_dict.keys()) >= 1, "The results should not be empty"
    for k in empty_keys:
        method_list.pop(method_list.index(k))

    inputs_all = np.load(os.path.join(base_dir, method_list[0], "saved/inputs.npy"))
    trues_all = np.load(os.path.join(base_dir, method_list[0], "saved/trues.npy"))

    # plot figures of the STL methods
    for idx in args.indices:
        print(idx)
        inputs, trues = inputs_all[idx], trues_all[idx]
        print(inputs.shape)
        for i, method in enumerate(method_list):
            print(method, predicts_dict[method][idx].shape)

            preds = predicts_dict[method][idx]

            # print input and truth only for the first method
            # print preds for all methods
            if i == 0:
                show_video_line(
                    inputs.copy(),
                    ncols=config["pre_seq_length"],
                    vmax=0.6,
                    cbar=False,
                    out_path="{}/{}_input_{}".format(
                        args.save_dirs,
                        args.dataname + c_surfix,
                        str(idx) + ".png",
                    ),
                    format="png",
                    use_rgb=use_rgb,
                )
                show_video_line(
                    trues.copy(),
                    ncols=config["aft_seq_length"],
                    vmax=0.6,
                    cbar=False,
                    out_path="{}/{}_true_{}".format(
                        args.save_dirs,
                        args.dataname + c_surfix,
                        str(idx) + "_nuovo.png",
                    ),
                    format="png",
                    use_rgb=use_rgb,
                )
                show_video_line(
                    preds.copy(),
                    ncols=config["aft_seq_length"],
                    vmax=0.6,
                    cbar=False,
                    out_path="{}/{}_pred_{}".format(
                        args.save_dirs,
                        args.dataname + c_surfix,
                        str(idx) + "_nuovo.png",
                    ),
                    format="png",
                    use_rgb=use_rgb,
                )

            show_video_line(
                preds,
                ncols=config["aft_seq_length"],
                vmax=0.6,
                cbar=False,
                out_path="{}/{}_{}_{}".format(
                    args.save_dirs,
                    args.dataname + c_surfix,
                    method,
                    str(idx) + "_next_nuovo.png",
                ),
                format="png",
                use_rgb=use_rgb,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataname",
        "-d",
        default="mmnist",
        type=str,
        help='The name of dataset ("mmnist" (default) or "pf")',
    )
    parser.add_argument(
        "--indices",
        "-i",
        nargs="+",
        default=[0],
        type=int,
        help="Indicate the indices of video sequences to show, for example --indices 0 5 10",
    )
    parser.add_argument(
        "--in_sequence",
        default=10,
        type=int,
        help="The number of the frames in the input video sequences to show",
    )
    parser.add_argument(
        "--out_sequence",
        default=10,
        type=int,
        help="The number of the frames in the output video sequences to show",
    )
    parser.add_argument(
        "--work_dirs",
        "-w",
        default="/home/monicar/predictive_zoo/OpenSTL/work_dirs/mmnist_simvp_gsta",
        # "/home/monicar/predictive_zoo/OpenSTL/work_dirs/pf_swin",
        type=str,
        help="Path to the work_dir or the path to a set of work_dirs (OpenSTL/work_dirs/mmnist_simvp_gsta)",
    )
    parser.add_argument(
        "--save_dirs",
        "-s",
        default="vis_figures",
        type=str,
        help="The path to save visualization results",
    )
    parser.add_argument(
        "--vis_channel",
        "-vc",
        default=-1,
        type=int,
        help="Select a channel to visualize as the heatmap",
    )

    args = parser.parse_args()
    main(args)
