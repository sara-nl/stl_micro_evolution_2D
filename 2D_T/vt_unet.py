# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging

import torch
import torch.nn as nn

from swin_transformer import SwinTransformer3D_Sys


class VTUNet(nn.Module):
    """class of VT-Unet
    Args:
        patch_size (int | tuple(int)): Patch size. Default: (4,4,4).
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        depths (tuple[int]): Depths of each Swin Transformer stage.
        num_classes (int): Number of classes for classification head. Default: 1000
        num_heads (tuple[int]): Number of attention head of each stage.
        window_size (int): Window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: Truee
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
        drop_rate (float): Dropout rate.
        attn_drop_rate (float): Attention dropout rate. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Default: 0.2.
        norm_layer: Normalization layer. Default: nn.LayerNorm.
        patch_norm (bool): If True, add normalization after patch embedding. Default: False.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
    """

    def __init__(
        self,
        config=None,
        img_size=(50, 100, 100),
        patch_size=(4, 4, 4),
        in_chans=3,
        embed_dim=96,
        depths=[2, 2, 2, 1],
        num_heads=[3, 6, 12, 24],
        window_size=(2, 7, 7),
        zero_head=False,
        num_classes=3,
    ):
        super(VTUNet, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.zero_head = zero_head
        self.config = config
        self.embed_dim = embed_dim
        self.depths = depths
        self.window_size = window_size

        self.swin_unet = SwinTransformer3D_Sys(
            img_size=self.img_size,
            patch_size=self.patch_size,
            in_chans=self.in_chans,
            num_classes=self.num_classes,
            embed_dim=self.embed_dim,
            depths=self.depths,
            depths_decoder=[1, 2, 2, 2],
            num_heads=self.num_heads,
            window_size=self.window_size,
            mlp_ratio=4.0,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.1,
            norm_layer=nn.LayerNorm,
            patch_norm=True,
            use_checkpoint=False,
            frozen_stages=-1,
            final_upsample="expand_first",
            ape=False,
        )

    def forward(self, x):
        logits = self.swin_unet(x)
        return logits

    def load_from(self, config):
        pretrained_path = config.MODEL.PRETRAIN_CKPT
        if pretrained_path is not None:
            print("pretrained_path:{}".format(pretrained_path))
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            if "model" not in pretrained_dict:
                print("---start load pretrained modle by splitting---")
                pretrained_dict = {k[17:]: v for k, v in pretrained_dict.items()}
                for k in list(pretrained_dict.keys()):
                    if "output" in k:
                        print("delete key:{}".format(k))
                        del pretrained_dict[k]
                self.swin_unet.load_state_dict(pretrained_dict, strict=False)

                return
            pretrained_dict = pretrained_dict["model"]
            print("---start load pretrained modle of swin encoder---")

            model_dict = self.swin_unet.state_dict()
            full_dict = copy.deepcopy(pretrained_dict)
            for k, v in pretrained_dict.items():
                if "layers." in k:
                    current_layer_num = 3 - int(k[7:8])
                    current_k = "layers_up." + str(current_layer_num) + k[8:]
                    full_dict.update({current_k: v})
            for k in list(full_dict.keys()):
                if k in model_dict:
                    if full_dict[k].shape != model_dict[k].shape:
                        print(
                            "delete:{};shape pretrain:{};shape model:{}".format(
                                k, v.shape, model_dict[k].shape
                            )
                        )
                        del full_dict[k]

            self.swin_unet.load_state_dict(full_dict, strict=False)
        else:
            print("none pretrain")
