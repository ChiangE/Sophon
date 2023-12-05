import os
import sys
import warnings

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
warnings.filterwarnings('ignore')

import argparse
import functools
import re
import time
from torch.cuda.amp import autocast
import torch
from torch.utils.data import DataLoader
import sys
sys.path.append('../')
from utils import (get_logger, set_seed, worker_seed_init_fn,
                         build_optimizer, Scheduler, build_training_mode)

class AverageMeter:
    '''Computes and stores the average and current value'''

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def all_reduce_operation_in_group_for_variables(variables, operator, group):
    for i in range(len(variables)):
        if not torch.is_tensor(variables[i]):
            variables[i] = torch.tensor(variables[i]).cuda()
        torch.distributed.all_reduce(variables[i], op=operator, group=group)
        variables[i] = variables[i].item()

    return variables

def train_diffusion_model(train_loader, model, criterion, trainer, optimizer,
                          scheduler, epoch, logger, config):
    '''
    train diffusion model for one epoch
    '''
    losses = AverageMeter()

    # switch to train mode
    model.train()

    local_rank = torch.distributed.get_rank()
    iters = len(train_loader.dataset) // config.batch_size
    iter_index = 1
    assert config.accumulation_steps >= 1, 'illegal accumulation_steps!'

    for _, data in enumerate(train_loader):
        images = data['image']
        images = images.cuda()

        labels = None
        if 'label' in data.keys() and config.num_classes:
            labels = data['label']
            labels = labels.cuda()

        if torch.any(torch.isinf(images)):
            continue

        if torch.any(torch.isnan(images)):
            continue

        with autocast():
            if iter_index % config.accumulation_steps == 0:
                pred_noise, noise = trainer(model,
                                            images,
                                            class_label=labels)
                loss = criterion(pred_noise, noise)
            else:
                # not reduce gradient while iter_index % config.accumulation_steps != 0
                with model.no_sync():
                    pred_noise, noise = trainer(model,
                                                images,
                                                class_label=labels)
                    loss = criterion(pred_noise, noise)


        if loss == 0. or torch.any(torch.isinf(loss)) or torch.any(
                torch.isnan(loss)):
            log_info = f'zero loss or nan loss or inf loss!'
            logger.info(log_info) if local_rank == 0 else None
            optimizer.zero_grad()
            continue

        loss = loss / config.accumulation_steps

        if iter_index % config.accumulation_steps == 0:
            config.scaler.scale(loss).backward()
        else:
            # not reduce gradient while iter_index % config.accumulation_steps != 0
            with model.no_sync():
                config.scaler.scale(loss).backward()

        if iter_index % config.accumulation_steps == 0:
            if hasattr(config,
                        'clip_max_norm') and config.clip_max_norm > 0:
                config.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                config.clip_max_norm)
            config.scaler.step(optimizer)
            config.scaler.update()
            optimizer.zero_grad()


        if iter_index % config.accumulation_steps == 0:
            [loss] = all_reduce_operation_in_group_for_variables(
                variables=[loss],
                operator=torch.distributed.ReduceOp.SUM,
                group=config.group)
            loss = loss / float(config.gpus_num)
            losses.update(loss, images.size(0))

        if iter_index % config.accumulation_steps == 0:
            scheduler.step(optimizer, iter_index / iters + (epoch - 1))

        accumulation_iter_index, accumulation_iters = int(
            iter_index // config.accumulation_steps), int(
                iters // config.accumulation_steps)
        if iter_index % int(
                config.print_interval * config.accumulation_steps) == 0:
            log_info = f'train: epoch {epoch:0>4d}, iter [{accumulation_iter_index:0>5d}, {accumulation_iters:0>5d}], lr: {scheduler.current_lr:.6f}, loss: {loss*config.accumulation_steps:.4f}'
            logger.info(log_info) if local_rank == 0 else None

        iter_index += 1

    avg_loss = losses.avg
    avg_loss = avg_loss * config.accumulation_steps

    return avg_loss

def parse_args():
    parser = argparse.ArgumentParser(
        description='PyTorch Diffusion Model Training')
    parser.add_argument(
        '--work-dir',
        type=str,
        help='path for get training config and saving log/models')

    return parser.parse_args()


def main():
    torch.cuda.empty_cache()
    args = parse_args()
    sys.path.append(args.work_dir)
    from train_config import config
    log_dir = os.path.join(args.work_dir, 'log')
    checkpoint_dir = os.path.join(args.work_dir, 'checkpoints')
    resume_model = os.path.join(checkpoint_dir, 'latest.pth')
    config.gpus_num = torch.cuda.device_count()

    local_rank = int(os.environ['LOCAL_RANK'])
    # start init process
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    torch.cuda.set_device(local_rank)
    config.group = torch.distributed.new_group(list(range(config.gpus_num)))

    if local_rank == 0:
        os.makedirs(
            checkpoint_dir) if not os.path.exists(checkpoint_dir) else None
        os.makedirs(log_dir) if not os.path.exists(log_dir) else None

    torch.distributed.barrier()

    logger = get_logger('train', log_dir)

    batch_size, num_workers = config.batch_size, config.num_workers
    assert config.batch_size % config.gpus_num == 0, 'config.batch_size is not divisible by config.gpus_num!'
    assert config.num_workers % config.gpus_num == 0, 'config.num_workers is not divisible by config.gpus_num!'
    batch_size = int(config.batch_size // config.gpus_num)
    num_workers = int(config.num_workers // config.gpus_num)

    init_fn = functools.partial(worker_seed_init_fn,
                                num_workers=num_workers,
                                local_rank=local_rank,
                                seed=config.seed)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        config.train_dataset, shuffle=True)
    train_loader = DataLoader(config.train_dataset,
                              batch_size=batch_size,
                              shuffle=False,
                              pin_memory=True,
                              drop_last=True,
                              num_workers=num_workers,
                              collate_fn=config.train_collater,
                              sampler=train_sampler,
                              worker_init_fn=init_fn)


    model = config.model.cuda()
    train_criterion = config.train_criterion.cuda()
    trainer = config.trainer.cuda()
    optimizer, model_layer_weight_decay_list = build_optimizer(config, model)

    for per_layer_list in model_layer_weight_decay_list:
        layer_name_list, layer_lr, layer_weight_decay = per_layer_list[
            'name'], per_layer_list['lr'], per_layer_list['weight_decay']

        lr_scale = 'not setting!'
        if 'lr_scale' in per_layer_list.keys():
            lr_scale = per_layer_list['lr_scale']

        for name in layer_name_list:
            log_info = f'name: {name}, lr: {layer_lr}, weight_decay: {layer_weight_decay}, lr_scale: {lr_scale}'
            logger.info(log_info) if local_rank == 0 else None

    scheduler = Scheduler(config, optimizer)
    model, config.ema_model, config.scaler = build_training_mode(config, model)

    start_epoch, train_time = 1, 0
    best_loss, train_loss = 1e9, 0
    if os.path.exists(resume_model):
        checkpoint = torch.load(resume_model, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        saved_epoch = checkpoint['epoch']
        start_epoch += saved_epoch
        used_time = checkpoint['time']
        train_time += used_time

        best_loss, train_loss, lr = checkpoint['best_loss'], checkpoint[
            'train_loss'], checkpoint['lr']

        log_info = f'resuming model from {resume_model}. resume_epoch: {saved_epoch:0>3d}, used_time: {used_time:.3f} hours, best_loss: {best_loss:.4f}, lr: {lr:.6f}'
        logger.info(log_info) if local_rank == 0 else None

        if 'ema_model_state_dict' in checkpoint.keys():
            config.ema_model.ema_model.load_state_dict(
                checkpoint['ema_model_state_dict'])

    for epoch in range(start_epoch, config.epochs + 1):
        per_epoch_start_time = time.time()

        log_info = f'epoch {epoch:0>3d} lr: {scheduler.current_lr:.6f}'
        logger.info(log_info) if local_rank == 0 else None

        torch.cuda.empty_cache()

        train_sampler.set_epoch(epoch)
        train_loss = train_diffusion_model(train_loader, model,
                                           train_criterion, trainer, optimizer,
                                           scheduler, epoch, logger, config)
        log_info = f'train: epoch {epoch:0>3d}, train_loss: {train_loss:.4f}'
        logger.info(log_info) if local_rank == 0 else None

        torch.cuda.empty_cache()

        train_time += (time.time() - per_epoch_start_time) / 3600

        if local_rank == 0:
            # save best acc1 model and each epoch checkpoint
            if train_loss < best_loss:
                best_loss = train_loss
                if config.use_ema_model:
                    save_best_model = config.ema_model.ema_model.module.state_dict(
                    )
                elif config.use_compile:
                    save_best_model = model._orig_mod.module.state_dict()
                else:
                    save_best_model = model.module.state_dict()

                torch.save(save_best_model,
                           os.path.join(checkpoint_dir, 'best.pth'))

            if config.use_compile:
                save_checkpoint_model = model._orig_mod.state_dict()
            else:
                save_checkpoint_model = model.state_dict()
            torch.save(
                {
                    'epoch': epoch,
                    'time': train_time,
                    'best_loss': best_loss,
                    'train_loss': train_loss,
                    'lr': scheduler.current_lr,
                    'model_state_dict': save_checkpoint_model,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                }, os.path.join(checkpoint_dir, 'latest.pth'))

        log_info = f'until epoch: {epoch:0>3d}, best_loss: {best_loss:.4f}'
        logger.info(log_info) if local_rank == 0 else None

    if local_rank == 0:
        if os.path.exists(os.path.join(checkpoint_dir, 'best.pth')):
            os.rename(os.path.join(checkpoint_dir, 'best.pth'),
                      os.path.join(checkpoint_dir, f'loss{best_loss:.3f}.pth'))

    log_info = f'train done. train time: {train_time:.3f} hours, best_loss: {best_loss:.4f}'
    logger.info(log_info) if local_rank == 0 else None

    return


if __name__ == '__main__':
    main()
