# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import gc
import logging
import math
from typing import Iterable, Any, Callable
import time

import torch
from torchmetrics.aggregation import MeanMetric
from models.augment import AugmentPipe
import models.rng as rng


logger = logging.getLogger(__name__)


def get_compiled_counts():
    metrics = torch._dynamo.utils.get_compilation_metrics()
    return len(metrics)


augment_pipe = AugmentPipe(p=0.12, xflip=1e8, yflip=0, scale=1, rotate_frac=0, aniso=1, translate_frac=1)  # turn off yflip and rotate

def train_step(model_without_ddp, *args, **kwargs):
    loss = model_without_ddp.forward_with_loss(*args, **kwargs)
    loss.backward(create_graph=False)
    return loss


def train_one_epoch(
    model: torch.nn.Module,
    compiled_train_step: Callable,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    lr_schedule: torch.torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    epoch: int,
    log_writer: Any,
    args: argparse.Namespace,
    meters: dict[str, MeanMetric],
):
    gc.collect()
    model.train(True)

    batch_loss = meters['batch_loss']
    batch_time = meters['batch_time']

    # declare the unwrapped model
    model_without_ddp = model

    tic = time.time()
    for data_iter_step, (samples, index) in enumerate(data_loader):
        steps = data_iter_step + len(data_loader) * epoch  # global step

        optimizer.zero_grad()            
        if data_iter_step > 0 and args.test_run:
            break

        samples = samples.to(device, non_blocking=True)
        samples = samples * 2.0 - 1.0

        samples, aug_cond = rng.augment_with_rng_control(augment_pipe, samples, args.seed, steps) if args.use_edm_aug else (samples, None)

        if args.compile and epoch == args.start_epoch and data_iter_step == 0:
            logging.info(f"Compiling the first train step, this may take a while...")
        
        loss = rng.train_step_with_rng_control(compiled_train_step, model_without_ddp, steps, args.seed, samples, aug_cond)
        if args.compile:
            assert get_compiled_counts() > 0, "Compilation not triggered."

        loss_value = loss.item()
        batch_loss.update(loss_value)

        if not math.isfinite(loss_value):
            raise ValueError(f"Loss is {loss_value}, stopping training")

        # update the parameters
        optimizer.step()
        model_without_ddp.update_ema()  # moved to begin of train_step

        # logging
        toc = time.time()
        batch_time.update(toc - tic)
        tic = toc

        lr = optimizer.param_groups[0]["lr"]
        lr_schedule.step()  # per-iteration lr
        if (steps + 1) % args.log_per_step == 0:
            loss_ave = batch_loss.compute().detach().cpu().numpy() # logging only
            sec_per_iter = batch_time.compute()
            batch_time.reset()
            batch_loss.reset()
            logger.info(
                f"Epoch {epoch} [{data_iter_step}/{len(data_loader)}]: loss = {loss_ave:.6f}, lr = {lr:.6f}, steps = {steps}, sec_per_iter = {sec_per_iter:.4f}"
            )
            epoch_1000x = int(steps / len(data_loader) * 1000)
            metrics = {
                "loss": loss_ave,
                "lr": lr,
                "epoch": steps / len(data_loader),
                "steps": steps,
                "sec_per_iter": sec_per_iter,
            }
            if log_writer is not None:
                for k, v in metrics.items():
                    log_writer.add_scalar(f"ep_{k}", v, epoch_1000x)  # we use epoch * 1000 to plot, for calibrating different batch sizes

    return
