# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
from pathlib import Path

import torch
import logging


def save_on_master(*args, **kwargs):
    torch.save(*args, **kwargs)


def save_model(
    args, epoch, model, optimizer, lr_schedule,
):
    output_dir = Path(args.output_dir)
    epoch_name = str(epoch)

    checkpoint_paths = [output_dir / "checkpoint-last.pth",]
    if (epoch + 1) % 1000 == 0:  # hack to save disk
        checkpoint_paths.append(output_dir / ("checkpoint-%s.pth" % epoch_name))
    for checkpoint_path in checkpoint_paths:
        to_save = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_schedule": lr_schedule.state_dict(),
            "epoch": epoch,
            "args": args,
        }

        save_on_master(to_save, checkpoint_path)


def load_model(args, model, optimizer, lr_schedule):
    if args.resume:
        if args.resume.startswith("https"):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location="cpu", check_hash=True
            )
        else:
            checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"], strict=False)
        logging.info("Resume checkpoint %s" % args.resume)
        if (
            "optimizer" in checkpoint
            and "epoch" in checkpoint
            and not args.eval_only
        ):
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_schedule.load_state_dict(checkpoint["lr_schedule"])
            args.start_epoch = checkpoint["epoch"] + 1
            logging.info(f"Start epoch set to {args.start_epoch}")
            logging.info("With optim & sched!")
