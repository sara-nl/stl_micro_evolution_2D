import torch
import torch.nn as nn

from openstl.modules import SwinTransformer3D_Sys


class VTUNet_Model(nn.Module):
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

    def __init__(self, configs, **kwargs):
        super(VTUNet_Model, self).__init__()
        T, C, H, W = configs.in_shape 
        self.configs = configs
        self.img_size = configs.img_size
        self.patch_size = configs.patch_size
        self.in_chans = configs.in_chans
        self.embed_dim = configs.embed_dim
        self.depths = configs.depths
        self.num_heads = configs.num_heads
        self.window_size = configs.window_size
        self.zero_head = configs.zero_head
        self.num_classes = configs.num_classes

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
        future_x = self.swin_unet(x)
        return future_x
