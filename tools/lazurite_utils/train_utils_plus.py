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
from torch.nn.utils import clip_grad_norm_
from pcdet.utils import common_utils, commu_utils
from tools.train_utils.train_utils import train_one_epoch, save_checkpoint, checkpoint_state


def test_one_epoch(model, optimizer, test_loader, model_func, accumulated_iter,
                   rank, tbar, epoch, tb_log=None,leave_pbar=False):
    total_it_each_epoch = len(test_loader)
    dataloader_iter = iter(test_loader)

    if rank == 0:
        pbar = tqdm.tqdm(total=total_it_each_epoch, leave=leave_pbar, desc='test', dynamic_ncols=True)
        data_time = common_utils.AverageMeter()
        batch_time = common_utils.AverageMeter()
        forward_time = common_utils.AverageMeter()

    all_loss = []
    all_tb_dict = []

    for cur_it in range(total_it_each_epoch):
        end = time.time()
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(test_loader)
            batch = next(dataloader_iter)
            print('new iters')

        data_timer = time.time()
        cur_data_time = data_timer - end

        model.train()
        optimizer.zero_grad()

        loss, tb_dict, disp_dict = model_func(model, batch)

        forward_timer = time.time()
        cur_forward_time = forward_timer - data_timer

        accumulated_iter += 1

        cur_batch_time = time.time() - end
        # average reduce
        avg_data_time = commu_utils.average_reduce_value(cur_data_time)
        avg_forward_time = commu_utils.average_reduce_value(cur_forward_time)
        avg_batch_time = commu_utils.average_reduce_value(cur_batch_time)

        # log to console and tensorboard
        if rank == 0:
            data_time.update(avg_data_time)
            forward_time.update(avg_forward_time)
            batch_time.update(avg_batch_time)
            disp_dict.update({
                'loss': loss.item(), 'd_time': f'{data_time.val:.2f}({data_time.avg:.2f})',
                'f_time': f'{forward_time.val:.2f}({forward_time.avg:.2f})',
                'b_time': f'{batch_time.val:.2f}({batch_time.avg:.2f})'
            })

            pbar.update()
            pbar.set_postfix(dict(total_it=accumulated_iter))
            tbar.set_postfix(disp_dict)
            tbar.refresh()

            all_loss.append(loss.item())
            all_tb_dict.append(tb_dict)

    if tb_log is not None:
        tb_log.add_scalar('test/loss', np.array(all_loss).mean(), epoch)
        for k, v in tb_dict.items():
            mean_v = np.array([tb[k] for tb in all_tb_dict]).mean()
            tb_log.add_scalar(f'test/{k}', mean_v, epoch)

    if rank == 0:
        pbar.close()
    return accumulated_iter


def train_model_with_test(model, optimizer, train_loader, test_loader, model_func, lr_scheduler, optim_cfg,
                          start_epoch, total_epochs, start_iter, rank, tb_log, ckpt_save_dir, train_sampler=None,
                          lr_warmup_scheduler=None, ckpt_save_interval=1, test_interval=1, max_ckpt_save_num=50):
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
            if trained_epoch % test_interval == 0 and rank == 0:
                test_one_epoch(
                    model, optimizer, test_loader, model_func,
                    accumulated_iter=accumulated_iter, rank=rank, tbar=tbar, epoch=cur_epoch, tb_log=tb_log,
                    leave_pbar=(cur_epoch + 1 == total_epochs)
                )