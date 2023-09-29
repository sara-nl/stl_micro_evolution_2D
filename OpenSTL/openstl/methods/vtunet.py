import time
import torch
import torch.nn as nn
from timm.utils import AverageMeter
from tqdm import tqdm

from openstl.models import VTUNet_Model
from openstl.utils import (
    reduce_tensor,
    reshape_patch,
    reshape_patch_back,
    reserve_schedule_sampling_exp,
    schedule_sampling,
)
from .base_method import Base_method


class VTUNet(Base_method):
    r"""VT UNet

    Implementation of VT Unet.

    Adaptation from the following references:
    - Official implementation: https://github.com/SwinTransformer/Video-Swin-Transformer
    - Swin Unet in: 
    # Swin Unet 2D data, 2021 https://github.com/HuCaoFighting/Swin-Unet
    # Swin Unet 2D+T data, 2022  https://github.com/himashi92/VT-UNet

    """

    def __init__(self, args, device, steps_per_epoch):
        Base_method.__init__(self, args, device, steps_per_epoch)
        self.model = self._build_model(self.args)
        self.model_optim, self.scheduler, self.by_epoch = self._init_optimizer(
            steps_per_epoch
        )
        self.criterion = nn.MSELoss()

    def _build_model(self, args):

        return VTUNet_Model(args).to(self.device)

    def _predict(self, batch_x, batch_y, **kwargs):
        """Forward the model"""
        x = batch_x.permute(0, 2, 1, 3, 4).contiguous()
        output = self.model(x)
        # permute back
        pred_y = output.permute(0, 2, 1, 3, 4).contiguous()

        return pred_y

    def train_one_epoch(
        self, runner, train_loader, epoch, num_updates, eta=None, **kwargs
    ):
        """Train the model with train_loader."""
        data_time_m = AverageMeter()
        losses_m = AverageMeter()
        self.model.train()
        if self.by_epoch:
            self.scheduler.step(epoch)
        train_pbar = tqdm(train_loader) if self.rank == 0 else train_loader

        end = time.time()
        for batch_x, batch_y in train_pbar:
            data_time_m.update(time.time() - end)
            self.model_optim.zero_grad()

            if not self.args.use_prefetcher:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            runner.call_hook("before_train_iter")

            input_frames = batch_x.permute(
                0, 2, 1, 3, 4
            ).contiguous() 
            true_frames = batch_y.permute(0, 2, 1, 3, 4).contiguous()

            with self.amp_autocast():
                pred_frames = self.model(input_frames)

            # calculate loss
            loss = self.criterion(pred_frames, true_frames)

            if not self.dist:
                losses_m.update(loss.item(), batch_x.size(0))

            if self.loss_scaler is not None:
                if torch.any(torch.isnan(loss)) or torch.any(torch.isinf(loss)):
                    raise ValueError("Inf or nan loss value. Please use fp32 training!")
                self.loss_scaler(
                    loss,
                    self.model_optim,
                    clip_grad=self.args.clip_grad,
                    clip_mode=self.args.clip_mode,
                    parameters=self.model.parameters(),
                )
            else:
                loss.backward()
                self.clip_grads(self.model.parameters())
                self.model_optim.step()

            torch.cuda.synchronize()
            num_updates += 1

            if self.dist:
                losses_m.update(reduce_tensor(loss), batch_x.size(0))

            if not self.by_epoch:
                self.scheduler.step()
            runner.call_hook("after_train_iter")
            runner._iter += 1

            if self.rank == 0:
                log_buffer = "train loss: {:.4f}".format(loss.item())
                log_buffer += " | data time: {:.4f}".format(data_time_m.avg)
                train_pbar.set_description(log_buffer)

            end = time.time()  # end for

        if hasattr(self.model_optim, "sync_lookahead"):
            self.model_optim.sync_lookahead()

        return num_updates, losses_m, eta
