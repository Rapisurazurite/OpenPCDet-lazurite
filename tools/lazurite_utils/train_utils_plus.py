# -*- coding: utf-8 -*-
# @Time    : 2022/3/7 14:43
# @Author  : Lazurite
# @Email   : lazurite@tom.com
# @File    : train_utils_plus.py
# @Software: PyCharm
import glob
import os

import numpy as np
import torch
import tqdm
import time

from pcdet.models import load_data_to_gpu
from torch.nn.utils import clip_grad_norm_
from pcdet.utils import common_utils, commu_utils
from tools.train_utils.train_utils import train_one_epoch, save_checkpoint, checkpoint_state


def test_one_epoch(model, optimizer, test_loader, model_func, rank, epoch, tb_log=None, leave_pbar=False):
    total_it_each_epoch = len(test_loader)
    dataloader_iter = iter(test_loader)

    if rank == 0:
        pbar = tqdm.tqdm(total=total_it_each_epoch, leave=leave_pbar, desc='test', dynamic_ncols=True)

    all_loss = []
    all_tb_dict = []

    with torch.no_grad():
        for cur_it in range(total_it_each_epoch):
            try:
                batch = next(dataloader_iter)
            except StopIteration:
                dataloader_iter = iter(test_loader)
                batch = next(dataloader_iter)
                print('new iters')

            model.train()  # in order to get loss of test data
            optimizer.zero_grad()

            # get model result and loss
            loss, tb_dict, disp_dict = model_func(model, batch)

            # log to console and tensorboard
            if rank == 0:
                disp_dict.update({
                    'loss': loss.item()
                })

                pbar.set_postfix(disp_dict)
                pbar.update()
                pbar.refresh()

                all_loss.append(loss.item())
                all_tb_dict.append(tb_dict)

    if rank == 0:
        if tb_log is not None:
            mean_loss = np.array(all_loss).mean()
            tb_log.add_scalar('test/loss', mean_loss, epoch)
            for k, v in tb_dict.items():
                mean_v = np.array([tb[k] for tb in all_tb_dict]).mean()
                tb_log.add_scalar(f'test/{k}', mean_v, epoch)

        pbar.write(f"epoch :{epoch}, test loss: {mean_loss}")
        pbar.close()


def train_model_with_test(model, optimizer, train_loader, test_loader, model_func, lr_scheduler, optim_cfg, start_epoch,
                          total_epochs, start_iter, rank, tb_log, ckpt_save_dir, train_sampler=None,
                          lr_warmup_scheduler=None, ckpt_save_interval=1, test_interval=1, max_ckpt_save_num=50,
                          epoch_to_test=40):
    """
    train model with test data
    Args:
        model:
        optimizer:
        train_loader:
        test_loader:
        model_func:
        lr_scheduler:
        optim_cfg:
        start_epoch:
        total_epochs:
        start_iter:
        rank:
        tb_log:
        ckpt_save_dir:
        train_sampler:
        lr_warmup_scheduler:
        ckpt_save_interval:
        test_interval: how many epochs to test one time
        max_ckpt_save_num:
        epoch_to_test: if current epoch is larger than epoch_to_test, loss of test data will be computed

    Returns:

    """
    accumulated_iter = start_iter
    with tqdm.trange(start_epoch, total_epochs, desc='epochs', dynamic_ncols=True, leave=(rank == 0)) as tbar:
        total_it_each_epoch = len(train_loader)
        dataloader_iter = iter(train_loader)
        for cur_epoch in tbar:
            if train_sampler is not None:
                train_sampler.set_epoch(cur_epoch)

            # train one epoch
            if lr_warmup_scheduler is not None and cur_epoch < optim_cfg.WARMUP_EPOCH:
                cur_scheduler = lr_warmup_scheduler
            else:
                cur_scheduler = lr_scheduler

            accumulated_iter = train_one_epoch(
                model, optimizer, train_loader, model_func,
                lr_scheduler=cur_scheduler,
                accumulated_iter=accumulated_iter, optim_cfg=optim_cfg,
                rank=rank, tbar=tbar, tb_log=tb_log,
                leave_pbar=(cur_epoch + 1 == total_epochs),
                total_it_each_epoch=total_it_each_epoch,
                dataloader_iter=dataloader_iter
            )

            trained_epoch = cur_epoch + 1

            # save trained model
            if trained_epoch % ckpt_save_interval == 0 and rank == 0:

                ckpt_list = glob.glob(str(ckpt_save_dir / 'checkpoint_epoch_*.pth'))
                ckpt_list.sort(key=os.path.getmtime)

                if ckpt_list.__len__() >= max_ckpt_save_num:
                    for cur_file_idx in range(0, len(ckpt_list) - max_ckpt_save_num + 1):
                        os.remove(ckpt_list[cur_file_idx])

                ckpt_name = ckpt_save_dir / ('checkpoint_epoch_%d' % trained_epoch)
                save_checkpoint(
                    checkpoint_state(model, optimizer, trained_epoch, accumulated_iter), filename=ckpt_name,
                )

            """
            test model
            """
            if trained_epoch % test_interval == 0 and rank == 0 and trained_epoch >= epoch_to_test:
                test_one_epoch(model, optimizer, test_loader, model_func, rank=rank, epoch=cur_epoch, tb_log=tb_log,
                               leave_pbar=(cur_epoch + 1 == total_epochs))
