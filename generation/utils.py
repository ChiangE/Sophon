import os
import sys
import warnings
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
warnings.filterwarnings('ignore')
from tqdm import tqdm
import argparse
import logging
import logging.handlers
import copy
import math
import wandb
import numpy as np
import os
from scipy import linalg
import random
import time
from torch.cuda.amp import autocast
import itertools
import cv2
from torch.optim.optimizer import Optimizer
from thop import profile
from thop import clever_format
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader,TensorDataset
import torchvision.transforms as transforms



class Lion(Optimizer):
    """Implements Lion algorithm.
    Initialize the hyperparameters.
    Args:
      params (iterable): iterable of parameters to optimize or dicts defining
        parameter groups
      lr (float, optional): learning rate (default: 1e-4)
      betas (Tuple[float, float], optional): coefficients used for computing
        running averages of gradient and its square (default: (0.9, 0.99))
      weight_decay (float, optional): weight decay coefficient (default: 0)
    Symbolic Discovery of Optimization Algorithms
    https://arxiv.org/abs/2302.06675
    https://github.com/google/automl/blob/master/lion/lion_pytorch.py
    """

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        if not 0.0 <= lr:
            raise ValueError('Invalid learning rate: {}'.format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError('Invalid beta parameter at index 0: {}'.format(
                betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError('Invalid beta parameter at index 1: {}'.format(
                betas[1]))
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
    Args:
      closure (callable, optional): A closure that reevaluates the model
        and returns the loss.
    Returns:
      the loss.
    """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                # Perform stepweight decay
                p.data.mul_(1 - group['lr'] * group['weight_decay'])

                grad = p.grad
                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p)

                exp_avg = state['exp_avg']
                beta1, beta2 = group['betas']

                # Weight update
                update = exp_avg * beta1 + grad * (1 - beta1)
                p.add_(torch.sign(update), alpha=-group['lr'])
                # Decay the momentum running average coefficient
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)

        return loss


def parse_args_example():
    '''
    args backup
    '''
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--string-variable',
                        type=str,
                        default='string',
                        help='explain variable')
    parser.add_argument('--float-variable',
                        type=float,
                        default=0.01,
                        help='explain variable')
    parser.add_argument('--int-variable',
                        type=int,
                        default=10,
                        help='explain variable')
    parser.add_argument('--list-variable',
                        type=list,
                        default=[1, 10, 100],
                        help='explain variable')

    parser.add_argument('--bool-variable',
                        default=False,
                        action='store_true',
                        help='explain variable')


    return parser.parse_args()


def get_logger(name, log_dir):
    '''
    Args:
        name(str): name of logger
        log_dir(str): path of log
    '''

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')

    file_name = os.path.join(log_dir, '{}.info.log'.format(name))
    file_handler = logging.handlers.TimedRotatingFileHandler(file_name,
                                                             when='W0',
                                                             encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def set_seed(seed):
    # for hash
    os.environ['PYTHONHASHSEED'] = str(seed)
    # for python and numpy
    random.seed(seed)
    np.random.seed(seed)
    # for cpu gpu
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # for cudnn
    cudnn.benchmark = False
    cudnn.deterministic = True


def worker_seed_init_fn(worker_id, num_workers, local_rank, seed):
    # worker_seed_init_fn function will be called at the beginning of each epoch
    # for each epoch the same worker has same seed value,so we add the current time to the seed
    worker_seed = num_workers * local_rank + worker_id + seed + int(
        time.time())
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def compute_macs_and_params(config, model):
    assert isinstance(config.input_image_size, int) == True or isinstance(
        config.input_image_size,
        list) == True, 'Illegal input_image_size type!'

    if isinstance(config.input_image_size, int):
        macs_input = torch.randn(1, 3, config.input_image_size,
                                 config.input_image_size).cpu()
    elif isinstance(config.input_image_size, list):
        macs_input = torch.randn(1, 3, config.input_image_size[0],
                                 config.input_image_size[1]).cpu()

    model = model.cpu()

    macs, params = profile(model, inputs=(macs_input, ), verbose=False)
    macs, params = clever_format([macs, params], '%.3f')

    return macs, params


class EmaModel(nn.Module):
    """ Model Exponential Moving Average V2
    https://github.com/rwightman/pytorch-image-models/blob/a2727c1bf78ba0d7b5727f5f95e37fb7f8866b1f/timm/utils/model_ema.py
    decay=0.9999 means that when updating the model weights, we keep 99.99% of the previous model weights and only update 0.01% of the new weights at each iteration.
    ema_model_weights = decay * ema_model_weights + (1 - decay) * model_weights

    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    A smoothed version of the weights is necessary for some training schemes to perform well.
    E.g. Google's hyper-params for training MNASNet, MobileNet-V3, EfficientNet, etc that use
    RMSprop with a short 2.4-3 epoch decay period and slow LR decay rate of .96-.99 requires EMA
    smoothing of weights to match results. Pay attention to the decay constant you are using
    relative to your update count per epoch.
    """

    def __init__(self, model, decay=0.9999):
        super(EmaModel, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval()
        self.decay = decay
        self.update_fn = lambda e, m: self.decay * e + (1. - self.decay) * m

    def update(self, model):
        with torch.no_grad():
            for ema_v, model_v in zip(self.ema_model.state_dict().values(),
                                      model.state_dict().values()):
                assert ema_v.shape == model_v.shape, 'wrong ema model!'
                ema_v.copy_(self.update_fn(ema_v, model_v))


def build_training_mode(config, model):
    ema_model, scaler = None, None
    if config.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).cuda()

    if hasattr(config, 'use_ema_model') and config.use_ema_model:
        ema_model = EmaModel(model, decay=config.ema_model_decay)


    if hasattr(config, 'use_amp') and config.use_amp:
        scaler = GradScaler()

    return model, ema_model, scaler


def compute_diffusion_model_metric(test_images_dataloader,
                                   generate_images_dataloader, test_image_num,
                                   generate_image_num, fid_model, config):

    for param in fid_model.parameters():
        param.requires_grad = False

    # switch to evaluate mode
    fid_model.eval()
    test_images_pred = np.empty((test_image_num, 2048))
    with torch.no_grad():
        test_images_start_idx = 0
        model_on_cuda = next(fid_model.parameters()).is_cuda
        for data in tqdm(test_images_dataloader, desc='calculating fid'):
            if model_on_cuda:
                if isinstance(data, list):
                    data = data[0].to(torch.float32)
                data = data.cuda() 
            preds = fid_model(data)
            per_batch_pred_features = preds[0].squeeze(-1).squeeze(
                -1).cpu().numpy()

            test_images_pred[test_images_start_idx:test_images_start_idx +
                             per_batch_pred_features.
                             shape[0]] = per_batch_pred_features
            test_images_start_idx = test_images_start_idx + per_batch_pred_features.shape[
                0]

    generate_images_pred = np.empty((generate_image_num, 2048))
    generate_images_cls_pred = np.empty((generate_image_num, 1008))

    with torch.no_grad():
        generate_images_start_idx = 0
        model_on_cuda = next(fid_model.parameters()).is_cuda
        for data in tqdm(generate_images_dataloader):
            if model_on_cuda:
                if isinstance(data, list):
                    data = data[0].to(torch.float32)
                data = data.cuda() 
            preds = fid_model(data)
            per_batch_pred_features = preds[0].squeeze(-1).squeeze(
                -1).cpu().numpy()
            per_batch_pred_probs = preds[1].cpu().numpy()

            generate_images_pred[
                generate_images_start_idx:generate_images_start_idx +
                per_batch_pred_features.shape[0]] = per_batch_pred_features

            generate_images_cls_pred[
                generate_images_start_idx:generate_images_start_idx +
                per_batch_pred_probs.shape[0]] = per_batch_pred_probs

            generate_images_start_idx = generate_images_start_idx + per_batch_pred_features.shape[
                0]

    mu1 = np.mean(test_images_pred, axis=0)
    sigma1 = np.cov(test_images_pred, rowvar=False)

    mu2 = np.mean(generate_images_pred, axis=0)
    sigma2 = np.cov(generate_images_pred, rowvar=False)

    fid_value = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)

    is_score_mean, is_score_std = compute_inception_score(
        generate_images_cls_pred, config.is_data_split_num)

    return fid_value, is_score_mean, is_score_std

def compute_inception_score(images_cls_pred, data_split_num=10):
    image_nums = images_cls_pred.shape[0]
    # compute generate images mean KL Divergence
    split_scores = []
    for k in range(data_split_num):
        # split the whole data into many parts,parts num = data_split_num
        part = images_cls_pred[(k * image_nums //
                                data_split_num):((k + 1) * image_nums //
                                                 data_split_num), :]

        kl = part * (np.log(part) -
                     np.log(np.expand_dims(np.mean(part, 0), 0)))
        kl = np.mean(np.sum(kl, 1))
        split_scores.append(np.exp(kl))

    # Inception Score = is_score_mean ± is_score_std
    is_score_mean = np.mean(split_scores)
    is_score_std = np.std(split_scores)

    return is_score_mean, is_score_std

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            # raise ValueError('Imaginary component {}'.format(m))
            print('Note: fid is too large to caculate... skip')
            return 0
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    out = (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) -
           2 * tr_covmean)

    return out


def check_gradients(model):
    total_gradients = 0
    num_parameters = 0
    count = 0
    for param in model.parameters():
        count+=1
        if param.grad is not None:
            # total_gradients += torch.sum(abs(param.grad.data)).item()
            total_gradients += torch.sum(torch.norm(param.grad.data,2)).item()
            # print(torch.norm(param.grad.data,2))
            num_parameters += param.grad.data.numel()
    # import pdb;pdb.set_trace()
    return total_gradients*1.0/count#/num_parameters


class Scheduler:

    def __init__(self, config, optimizer):
        self.scheduler_name = config.scheduler[0]
        self.scheduler_parameters = config.scheduler[1]
        self.warm_up_epochs = self.scheduler_parameters['warm_up_epochs']
        self.epochs = config.epochs
        self.optimizer_parameters = config.optimizer[1]
        self.lr = self.optimizer_parameters['lr']
        self.current_lr = self.lr

        self.init_param_groups_lr = [
            param_group["lr"] for param_group in optimizer.param_groups
        ]

        assert self.scheduler_name in ['MultiStepLR', 'CosineLR',
                                       'PolyLR'], 'Unsupported scheduler!'
        assert self.warm_up_epochs >= 0, 'Illegal warm_up_epochs!'
        assert self.epochs > 0, 'Illegal epochs!'

    def step(self, optimizer, epoch):
        if self.scheduler_name == 'MultiStepLR':
            gamma = self.scheduler_parameters['gamma']
            milestones = self.scheduler_parameters['milestones']
        elif self.scheduler_name == 'CosineLR':
            min_lr = 0. if 'min_lr' not in self.scheduler_parameters.keys(
            ) else self.scheduler_parameters['min_lr']
        elif self.scheduler_name == 'PolyLR':
            power = self.scheduler_parameters['power']
            min_lr = 0. if 'min_lr' not in self.scheduler_parameters.keys(
            ) else self.scheduler_parameters['min_lr']

        assert len(self.init_param_groups_lr) == len(optimizer.param_groups)

        for idx, param_group in enumerate(optimizer.param_groups):
            param_group_init_lr = self.init_param_groups_lr[idx]

            if self.scheduler_name == 'MultiStepLR':
                param_group_current_lr = (
                    epoch
                ) / self.warm_up_epochs * param_group_init_lr if epoch < self.warm_up_epochs else gamma**len(
                    [m
                     for m in milestones if m <= epoch]) * param_group_init_lr
            elif self.scheduler_name == 'CosineLR':
                param_group_current_lr = (
                    epoch
                ) / self.warm_up_epochs * param_group_init_lr if epoch < self.warm_up_epochs else 0.5 * (
                    math.cos((epoch - self.warm_up_epochs) /
                             (self.epochs - self.warm_up_epochs) * math.pi) +
                    1) * (param_group_init_lr - min_lr) + min_lr
            elif self.scheduler_name == 'PolyLR':
                param_group_current_lr = (
                    epoch
                ) / self.warm_up_epochs * param_group_init_lr if epoch < self.warm_up_epochs else (
                    (1 - (epoch - self.warm_up_epochs) /
                     (self.epochs - self.warm_up_epochs))**
                    power) * (param_group_init_lr - min_lr) + min_lr

            param_group["lr"] = param_group_current_lr

        if self.scheduler_name == 'MultiStepLR':
            self.current_lr = (
                epoch
            ) / self.warm_up_epochs * self.lr if epoch < self.warm_up_epochs else gamma**len(
                [m for m in milestones if m <= epoch]) * self.lr
        elif self.scheduler_name == 'CosineLR':
            self.current_lr = (
                epoch
            ) / self.warm_up_epochs * self.lr if epoch < self.warm_up_epochs else 0.5 * (
                math.cos((epoch - self.warm_up_epochs) /
                         (self.epochs - self.warm_up_epochs) * math.pi) +
                1) * (self.lr - min_lr) + min_lr
        elif self.scheduler_name == 'PolyLR':
            self.current_lr = (
                epoch
            ) / self.warm_up_epochs * self.lr if epoch < self.warm_up_epochs else (
                (1 - (epoch - self.warm_up_epochs) /
                 (self.epochs - self.warm_up_epochs))**
                power) * (self.lr - min_lr) + min_lr

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items()}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

def generate_diffusion_model_images(test_loader, model, sampler, config):
    # print('Generating......')
    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        batch_idx = 0
        model_on_cuda = next(model.parameters()).is_cuda
        for data in tqdm(test_loader, desc='generating iterations'):
            batch_idx += 1
            if batch_idx > 5: 
                break
            images = data['image']
            if model_on_cuda:
                images = images.cuda()

            labels = None
            if 'label' in data.keys(
            ) and config.num_classes and config.use_condition_label:
                labels = data['label']
                if model_on_cuda:
                    labels = labels.cuda()

            # torch.cuda.synchronize()
            input_images, input_masks = None, None
            if config.use_input_images:
                input_images = images

            _, outputs = sampler(model,
                                 images.shape,
                                 class_label=labels,
                                 input_images=input_images,
                                 input_masks=input_masks,
                                 return_intermediates=True)

            # torch.cuda.synchronize()

            mean = np.expand_dims(np.expand_dims(config.mean, axis=0), axis=0)
            std = np.expand_dims(np.expand_dims(config.std, axis=0), axis=0)

            for image_idx, (per_image,
                            per_output) in enumerate(zip(images, outputs)): 
                per_image = per_image.cpu().numpy()
                per_image = per_image.transpose(1, 2, 0) # 3*32*32 --> 32*32*3
                per_image = (per_image * std + mean) * 255.

                per_output = per_output.transpose(1, 2, 0)
                per_output = (per_output * std + mean) * 255.

                per_image = np.ascontiguousarray(per_image, dtype=np.uint8)
                per_image = cv2.cvtColor(per_image, cv2.COLOR_RGB2BGR)

                per_output = np.ascontiguousarray(per_output, dtype=np.uint8)
                per_output = cv2.cvtColor(per_output, cv2.COLOR_RGB2BGR)

                save_image_name = f'image_{batch_idx}_{image_idx}.jpg'
                save_image_path = os.path.join(config.save_test_image_dir,
                                               save_image_name)
                cv2.imencode('.jpg', per_image)[1].tofile(save_image_path)

                save_output_name = f'output_{batch_idx}_{image_idx}.jpg'
                save_output_path = os.path.join(config.save_generate_image_dir,
                                                save_output_name)
                cv2.imencode('.jpg', per_output)[1].tofile(save_output_path)


    test_images_path_list = []
    for per_image_name in os.listdir(config.save_test_image_dir):
        per_image_path = os.path.join(config.save_test_image_dir,
                                      per_image_name)
        test_images_path_list.append(per_image_path)

    generate_images_path_list = []
    for per_image_name in os.listdir(config.save_generate_image_dir):
        per_image_path = os.path.join(config.save_generate_image_dir,
                                      per_image_name)
        generate_images_path_list.append(per_image_path)

    test_image_num = len(test_images_path_list)
    generate_image_num = len(generate_images_path_list)
    test_images_dataset = ImagePathDataset(test_images_path_list,
                                        transform=transforms.Compose([
                                            transforms.Resize([
                                                config.input_image_size,
                                                config.input_image_size
                                            ]),
                                            transforms.ToTensor(),
                                        ]))

    generate_images_dataset = ImagePathDataset(generate_images_path_list,
                                            transform=transforms.Compose([
                                                transforms.Resize([
                                                    config.input_image_size,
                                                    config.input_image_size
                                                ]),
                                                transforms.ToTensor(),
                                            ]))
    test_images_dataloader = DataLoader(test_images_dataset,
                                    batch_size=config.fid_model_batch_size,
                                    shuffle=False,
                                    drop_last=False,
                                    )
    generate_images_dataloader = DataLoader(generate_images_dataset,
                                        batch_size=config.fid_model_batch_size,
                                        shuffle=False,
                                        drop_last=False,
                                        )

    return test_images_dataloader, generate_images_dataloader, test_image_num, generate_image_num


def generate_diffusion_model_images_condition(test_loader, model, classifier, sampler, config):
    print('Generating......')
    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        batch_idx = 0
        model_on_cuda = next(model.parameters()).is_cuda
        for data in tqdm(test_loader, desc='generating 50 iterations'):
            batch_idx += 1
            # if batch_idx > 5:
            #     break
            images = data['image']
            if model_on_cuda:
                images = images.cuda()

            labels = None
            if 'label' in data.keys(
            ) and config.num_classes and config.use_condition_label:
                labels = data['label']
                if model_on_cuda:
                    labels = labels.cuda()

            # torch.cuda.synchronize()
            input_images, input_masks = None, None
            if config.use_input_images:
                input_images = images

           
            _, outputs = sampler(model,
                                 images.shape,
                                 class_label=labels,
                                 input_images=input_images,
                                 input_masks=input_masks,
                                 return_intermediates=True)

            # torch.cuda.synchronize()

            mean = np.expand_dims(np.expand_dims(config.mean, axis=0), axis=0)
            std = np.expand_dims(np.expand_dims(config.std, axis=0), axis=0)

            for image_idx, (per_image,
                            per_output) in enumerate(zip(images, outputs)): 
                per_image = per_image.cpu().numpy()
                per_image = per_image.transpose(1, 2, 0) # 3*32*32 --> 32*32*3
                per_image = (per_image * std + mean) * 255.

                per_output = per_output.transpose(1, 2, 0)
                per_output = (per_output * std + mean) * 255.

                per_image = np.ascontiguousarray(per_image, dtype=np.uint8)
                per_image = cv2.cvtColor(per_image, cv2.COLOR_RGB2BGR)

                per_output = np.ascontiguousarray(per_output, dtype=np.uint8)
                per_output = cv2.cvtColor(per_output, cv2.COLOR_RGB2BGR)

                save_image_name = f'image_{batch_idx}_{image_idx}.jpg'
                save_image_path = os.path.join(config.save_test_image_dir,
                                               save_image_name)
                cv2.imencode('.jpg', per_image)[1].tofile(save_image_path)

                save_output_name = f'output_{batch_idx}_{image_idx}.jpg'
                save_output_path = os.path.join(config.save_generate_image_dir,
                                                save_output_name)
                cv2.imencode('.jpg', per_output)[1].tofile(save_output_path)


    test_images_path_list = []
    for per_image_name in os.listdir(config.save_test_image_dir):
        per_image_path = os.path.join(config.save_test_image_dir,
                                      per_image_name)
        test_images_path_list.append(per_image_path)

    generate_images_path_list = []
    for per_image_name in os.listdir(config.save_generate_image_dir):
        per_image_path = os.path.join(config.save_generate_image_dir,
                                      per_image_name)
        generate_images_path_list.append(per_image_path)

    test_image_num = len(test_images_path_list)
    generate_image_num = len(generate_images_path_list)
    test_images_dataset = ImagePathDataset(test_images_path_list,
                                        transform=transforms.Compose([
                                            transforms.Resize([
                                                config.input_image_size,
                                                config.input_image_size
                                            ]),
                                            transforms.ToTensor(),
                                        ]))

    generate_images_dataset = ImagePathDataset(generate_images_path_list,
                                            transform=transforms.Compose([
                                                transforms.Resize([
                                                    config.input_image_size,
                                                    config.input_image_size
                                                ]),
                                                transforms.ToTensor(),
                                            ]))
    test_images_dataloader = DataLoader(test_images_dataset,
                                    batch_size=config.fid_model_batch_size,
                                    shuffle=False,
                                    drop_last=False,
                                    )
    generate_images_dataloader = DataLoader(generate_images_dataset,
                                        batch_size=config.fid_model_batch_size,
                                        shuffle=False,
                                        drop_last=False,
                                        )
    target_class_data = []
    for data, _ in generate_images_dataloader:
        with torch.no_grad():
            logits = classifier(data)
            predicts = torch.max(logits, dim=0)[0]
            index = torch.where(predicts==0)
            target_class_data.extend(data[index],dim=0)
    


    return test_images_dataloader, generate_images_dataloader, test_image_num, generate_image_num






def generate_diffusion_model_images_tensor(test_loader, model, sampler, config):
    print('****Generating images.......****')
    model.eval()
    with torch.no_grad():
        model_on_cuda = next(model.parameters()).is_cuda
        test_images_list = []
        generate_images_list = []
        batch_idx = 0
        for data in tqdm(test_loader):
            batch_idx += 1
            if batch_idx > 20:
                break
            images = data['image']
            if model_on_cuda:
                images = images.cuda()

            labels = None
            if 'label' in data.keys() and config.num_classes and config.use_condition_label:
                labels = data['label']
                if model_on_cuda:
                    labels = labels.cuda()

            torch.cuda.synchronize()
            input_images, input_masks = None, None
            if config.use_input_images:
                input_images = images

            _, outputs = sampler(model,
                                 images.shape,
                                 class_label=labels,
                                 input_images=input_images,
                                 input_masks=input_masks,
                                 return_intermediates=True)

            torch.cuda.synchronize()

            mean = np.expand_dims(np.expand_dims(config.mean, axis=0), axis=0)
            std = np.expand_dims(np.expand_dims(config.std, axis=0), axis=0)

            for per_image, per_output in zip(images, outputs):
                per_image = per_image.cpu().numpy()
                per_image = per_image.transpose(1, 2, 0)
                per_image = (per_image * std + mean) * 255.
                per_output = per_output.transpose(1, 2, 0)
                per_output = (per_output * std + mean) * 255.
                test_images_list.append(per_image)
                generate_images_list.append(per_output)

    test_images_tensor = torch.stack([transforms.ToTensor()(img) for img in test_images_list])
    generate_images_tensor = torch.stack([transforms.ToTensor()(img) for img in generate_images_list])
    test_images_dataset = TensorDataset(test_images_tensor)
    generate_images_dataset = TensorDataset(generate_images_tensor)
    test_images_dataloader = DataLoader(test_images_dataset,
                                        batch_size=config.fid_model_batch_size,
                                        shuffle=False,
                                        drop_last=False)
    generate_images_dataloader = DataLoader(generate_images_dataset,
                                            batch_size=config.fid_model_batch_size,
                                            shuffle=False,
                                            drop_last=False)
    return test_images_dataloader, generate_images_dataloader, len(test_images_list), len(generate_images_list)

def build_optimizer(config, model):
    optimizer_name = config.optimizer[0]
    optimizer_parameters = config.optimizer[1]
    assert optimizer_name in ['SGD', 'AdamW', 'Lion'], 'Unsupported optimizer!'

    lr = optimizer_parameters['lr']
    weight_decay = optimizer_parameters['weight_decay']
    # if global_weight_decay = True,no_weight_decay_layer_name_list can't be set.
    no_weight_decay_layer_name_list = []
    if 'no_weight_decay_layer_name_list' in optimizer_parameters.keys(
    ) and isinstance(optimizer_parameters['no_weight_decay_layer_name_list'],
                     list):
        no_weight_decay_layer_name_list = optimizer_parameters[
            'no_weight_decay_layer_name_list']

    # import pdb;pdb.set_trace()
    param_layer_name_list = []
    param_layer_weight_dict = {}
    param_layer_decay_dict, param_layer_lr_dict = {}, {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        param_layer_name_list.append(name)
        param_layer_weight_dict[name] = param

        if param.ndim == 1 or any(no_weight_decay_layer_name in name
                                    for no_weight_decay_layer_name in
                                    no_weight_decay_layer_name_list):
            param_layer_decay_dict[name] = 0.
        else:
            per_layer_weight_decay = weight_decay
            param_layer_decay_dict[name] = per_layer_weight_decay

        per_layer_lr = lr
        param_layer_lr_dict[name] = per_layer_lr

    assert len(param_layer_name_list) == len(
        param_layer_weight_dict) == len(param_layer_decay_dict) == len(
            param_layer_lr_dict)

    unique_decays = list(set(param_layer_decay_dict.values()))
    unique_lrs = list(set(param_layer_lr_dict.values()))

    lr_weight_decay_combination = []
    for per_decay in unique_decays:
        for per_lr in unique_lrs:
            lr_weight_decay_combination.append([per_decay, per_lr])

    model_params_weight_decay_list = []
    model_layer_weight_decay_list = []
    for per_decay, per_lr in lr_weight_decay_combination:
        per_decay_lr_param_list, per_decay_lr_name_list = [], []
        for per_layer_name in param_layer_name_list:
            per_layer_weight = param_layer_weight_dict[per_layer_name]
            per_layer_weight_decay = param_layer_decay_dict[per_layer_name]
            per_layer_lr = param_layer_lr_dict[per_layer_name]

            if per_layer_weight_decay == per_decay and per_layer_lr == per_lr:
                per_decay_lr_param_list.append(per_layer_weight)
                per_decay_lr_name_list.append(per_layer_name)

        assert len(per_decay_lr_param_list) == len(per_decay_lr_name_list)

        if len(per_decay_lr_param_list) > 0:
            model_params_weight_decay_list.append({
                'params': per_decay_lr_param_list,
                'weight_decay': per_decay,
                'lr': per_lr,
            })
            model_layer_weight_decay_list.append({
                'name': per_decay_lr_name_list,
                'weight_decay': per_decay,
                'lr': per_lr,
            })

    assert len(model_params_weight_decay_list) == len(
        model_layer_weight_decay_list)

    if optimizer_name == 'SGD':
        momentum = optimizer_parameters['momentum']
        nesterov = False if 'nesterov' not in optimizer_parameters.keys(
        ) else optimizer_parameters['nesterov']
        return torch.optim.SGD(
            model_params_weight_decay_list,
            lr=lr,
            momentum=momentum,
            nesterov=nesterov), model_layer_weight_decay_list
    elif optimizer_name == 'AdamW':
        beta1 = 0.9 if 'beta1' not in optimizer_parameters.keys(
        ) else optimizer_parameters['beta1']
        beta2 = 0.999 if 'beta2' not in optimizer_parameters.keys(
        ) else optimizer_parameters['beta2']
        return torch.optim.AdamW(model_params_weight_decay_list,
                                 lr=lr,
                                 betas=(beta1,
                                        beta2)), model_layer_weight_decay_list
    elif optimizer_name == 'Lion':
        beta1 = 0.9 if 'beta1' not in optimizer_parameters.keys(
        ) else optimizer_parameters['beta1']
        beta2 = 0.99 if 'beta2' not in optimizer_parameters.keys(
        ) else optimizer_parameters['beta2']
        return Lion(model_params_weight_decay_list,
                    lr=lr,
                    betas=(beta1, beta2)), model_layer_weight_decay_list

from PIL import Image

class ImagePathDataset(torch.utils.data.Dataset):

    def __init__(self, files, transform=None):
        self.files = files
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        return img


def load_state_dict(saved_model_path,
                    model,
                    excluded_layer_name=(),
                    loading_new_input_size_position_encoding_weight=False):
    '''
    saved_model_path: a saved model.state_dict() .pth file path
    model: a new defined model
    excluded_layer_name: layer names that doesn't want to load parameters
    loading_new_input_size_position_encoding_weight: default False, for vit net, loading a position encoding layer with new input size, set True
    only load layer parameters which has same layer name and same layer weight shape
    '''
    if not saved_model_path:
        print('No pretrained model file!')
        return

    saved_state_dict = torch.load(saved_model_path,
                                  map_location=torch.device('cpu'))

    not_loaded_save_state_dict = []
    filtered_state_dict = {}
    for name, weight in saved_state_dict.items():
        if name in model.state_dict() and not any(
                excluded_name in name for excluded_name in excluded_layer_name
        ) and weight.shape == model.state_dict()[name].shape:
            filtered_state_dict[name] = weight
        else:
            not_loaded_save_state_dict.append(name)

    position_encoding_already_loaded = False
    if 'position_encoding' in filtered_state_dict.keys():
        position_encoding_already_loaded = True

    # for vit net, loading a position encoding layer with new input size
    if loading_new_input_size_position_encoding_weight and not position_encoding_already_loaded:
        # assert position_encoding_layer name are unchanged for model and saved_model
        # assert class_token num are unchanged for model and saved_model
        # assert embedding_planes are unchanged for model and saved_model
        model_num_cls_token = model.cls_token.shape[1]
        model_embedding_planes = model.position_encoding.shape[2]
        model_encoding_shape = int(
            (model.position_encoding.shape[1] - model_num_cls_token)**0.5)
        encoding_layer_name, encoding_layer_weight = None, None
        for name, weight in saved_state_dict.items():
            if 'position_encoding' in name:
                encoding_layer_name = name
                encoding_layer_weight = weight
                break
        save_model_encoding_shape = int(
            (encoding_layer_weight.shape[1] - model_num_cls_token)**0.5)

        save_model_cls_token_weight = encoding_layer_weight[:, 0:
                                                            model_num_cls_token, :]
        save_model_position_weight = encoding_layer_weight[:,
                                                           model_num_cls_token:, :]
        save_model_position_weight = save_model_position_weight.reshape(
            -1, save_model_encoding_shape, save_model_encoding_shape,
            model_embedding_planes).permute(0, 3, 1, 2)
        save_model_position_weight = F.interpolate(save_model_position_weight,
                                                   size=(model_encoding_shape,
                                                         model_encoding_shape),
                                                   mode='bicubic',
                                                   align_corners=False)
        save_model_position_weight = save_model_position_weight.permute(
            0, 2, 3, 1).flatten(1, 2)
        model_encoding_layer_weight = torch.cat(
            (save_model_cls_token_weight, save_model_position_weight), dim=1)

        filtered_state_dict[encoding_layer_name] = model_encoding_layer_weight
        not_loaded_save_state_dict.remove('position_encoding')

    if len(filtered_state_dict) == 0:
        print('No pretrained parameters to load!')
    else:
        print(
            f'load/model weight nums:{len(filtered_state_dict)}/{len(model.state_dict())}'
        )
        print(f'not loaded save layer weight:\n{not_loaded_save_state_dict}')
        model.load_state_dict(filtered_state_dict, strict=False)

    return
import shutil

def deleteimg(config, total_loop, save_path):
    try:
        shutil.copytree(config.save_image_dir, save_path+f'/imgs/{total_loop}/')
        shutil.rmtree(config.save_image_dir)
        print(f'images removed')
        for dir in [config.save_image_dir, config.save_test_image_dir, config.save_generate_image_dir]:
            os.makedirs(dir, exist_ok=True)
    except Exception as e:
        print(f'error')

def intermidiate_test(model, original_trainloader_fortest, original_testloader, target_trainloader_fortest, target_testloader, config, train_criterion, trainer, sampler, fid_model, save_path, loop):
        print('\n********intermidiate test**********')
        print('-----test original performance-----')
        test_model = copy.deepcopy(model)
        # test_imagesloader, generate_imagesloader, test_image_num, generate_image_num = generate_diffusion_model_images(original_trainloader_fortest, test_model, sampler, config)
        # original_fid, _, _ = test(fid_model,test_imagesloader,generate_imagesloader,test_image_num, generate_image_num, config)
        # print(f'original fid is {original_fid}\n')
        # deleteimg(config, loop, save_path)
        # # test original train loss
        assert(all(torch.equal(p1,p2) for p1,p2 in zip(test_model.parameters(), model.parameters())))
        original_trainloss=test_loss(test_model, original_trainloader_fortest, train_criterion, trainer, torch.device('cuda'))
        print(f'original train loss is {original_trainloss}')
        # test original test loss
        assert(all(torch.equal(p1,p2) for p1,p2 in zip(test_model.parameters(), model.parameters())))
        original_testloss=test_loss(test_model, original_testloader, train_criterion, trainer, torch.device('cuda'))
        print(f'original test loss is {original_testloss}')

        print('-----test target performance-----')
        # test target train loss
        assert(all(torch.equal(p1,p2) for p1,p2 in zip(test_model.parameters(), model.parameters())))
        target_trainloss=test_loss(test_model, target_trainloader_fortest, train_criterion, trainer, torch.device('cuda'))
        print(f'target train loss is {target_trainloss}')
        # test target test loss
        assert(all(torch.equal(p1,p2) for p1,p2 in zip(test_model.parameters(), model.parameters())))
        target_testloss=test_loss(test_model, target_testloader, train_criterion, trainer, torch.device('cuda'))
        print(f'target test loss is {target_testloss}')
        print('********test finished**********')

        # wandb.log({'intermediate test/original fid':original_fid, 'intermediate test/original train loss':original_trainloss, 'intermediate test/original test loss':original_testloss, 'intermediate test/target train loss': target_trainloss, 'intermediate test/target test loss': target_testloss})
        wandb.log({ 'intermediate test/original train loss':original_trainloss, 'intermediate test/original test loss':original_testloss, 'intermediate test/target train loss': target_trainloss, 'intermediate test/target test loss': target_testloss})
        return original_trainloss, original_testloss, target_trainloss, target_testloss


def test_loss(model, target_testloader, criterion, trainer, device):
    totalloss = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for batch in target_testloader:
            inputs, targets = batch['image'], batch['label']
            inputs, targets = inputs.cuda(), targets.cuda()
            pred_noise_eva, noise_eva = trainer(model,inputs,class_label=targets)
            loss = criterion(pred_noise_eva, noise_eva)
            totalloss += loss.item() * inputs.shape[0]
            total += inputs.shape[0]
    return totalloss*1.0/total

def finetune_model(model,config, target_trainloader,trainer, criterion, optimizer,epochs):
    losses = AverageMeter()
    model.train()
    iters = len(target_trainloader.dataset) // config.batch_size
    iter_index = 1
    epoch = 0
    for epoch in tqdm(range(1,epochs+1), desc='epochs'):
        for batch in tqdm(target_trainloader):
            images, labels = batch['image'], batch['label']
            images = images.cuda()
            labels = None
            with autocast():
                pred_noise, noise = trainer(model,images,class_label=labels)
                loss = criterion(pred_noise, noise)
            if loss == 0. or torch.any(torch.isinf(loss)) or torch.any(
                torch.isnan(loss)):
                optimizer.zero_grad()
                continue
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.update(loss, images.size(0))
            iter_index += 1
            avg_loss = losses.avg
            avg_loss = avg_loss 
    return model

def final_testfinetune_aug(start, model, fid_model, sampler, config, target_trainloader, target_testloader, trainer, criterion, optimizer,epochs):
    train_losses = AverageMeter()
    model.train()
    iters = len(target_trainloader.dataset) // config.batch_size
    for epoch in tqdm(range(1,epochs+1), desc='epochs'):
        for batch in tqdm(target_trainloader):
            images, labels = batch['image'], batch['label']
            images = images.cuda()
            labels = None
            with autocast():
                pred_noise, noise = trainer(model,images,class_label=labels)
                loss = criterion(pred_noise, noise)
            if loss == 0. or torch.any(torch.isinf(loss)) or torch.any(
                torch.isnan(loss)):
                optimizer.zero_grad()
                continue
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_losses.update(loss, images.size(0))
            
        wandb.log({f'Truly finetune/{start} train loss': train_losses.avg})
        if (epoch % 10 == 0) or epoch == epochs :
            target_test_loss=test_loss(model, target_testloader, criterion, trainer, torch.device('cuda'))
            target_fid, _, _=test_finetune_fid(model, fid_model, target_testloader, sampler, config)
            wandb.log({f'Truly finetune/{start} test loss':target_test_loss,f'Truly finetune/{start} target fid':target_fid})
    return target_test_loss, target_fid

def final_testfinetune_lr(start, model, fid_model, sampler, config, target_trainloader, target_trainloaderfortest, target_testloader, trainer, criterion, scheduler, optimizer,epochs):
    # train_losses = AverageMeter()
    target_train_loss=test_loss(model, target_trainloaderfortest, criterion, trainer, torch.device('cuda'))
    # target_fid, _, _=test_finetune_fid(model, fid_model, target_trainloaderfortest, sampler, config)
    wandb.log({f'Truly finetune/{start} train loss':target_train_loss,})#f'Truly finetune/{start} target fid':target_fid})
    print('*****check train loss before finetune*****', target_train_loss)
    
    ## test test loss
    target_test_loss=test_loss(model, target_testloader, criterion, trainer, torch.device('cuda'))
    print('*****check test loss before finetune*****', target_test_loss)

    model.train()
    for epoch in tqdm(range(1,epochs+1), desc='epochs'):
        model.train()
        iter_index = 1
        description = "train loss={:.4f}"
        totalloss = 0
        total = 0
        print(f'current lr: {scheduler.current_lr:.6f}')
        with tqdm(target_trainloader) as batches:
         for batch in batches:
            images, labels = batch['image'], batch['label']
            images = images.cuda()
            labels = None
            # with autocast():
            pred_noise, noise = trainer(model,images,class_label=labels)
            loss = criterion(pred_noise, noise)
            if loss == 0. or torch.any(torch.isinf(loss)) or torch.any(
                torch.isnan(loss)):
                optimizer.zero_grad()
                continue
            loss.backward() 
            optimizer.step()
            optimizer.zero_grad()
            totalloss += loss*images.shape[0]
            total += images.shape[0]
            scheduler.step(optimizer, epoch)
            iter_index += 1
            batches.set_description(description.format(totalloss*1.0/total))
            
        wandb.log({f'Truly finetune/{start} train loss': totalloss*1.0/total})
        if (epoch % 1 == 0) or epoch == epochs :
            target_test_loss=test_loss(model, target_testloader, criterion, trainer, torch.device('cuda'))
            # target_fid, _, _=test_finetune_fid(model, fid_model, target_trainloaderfortest, sampler, config)
            wandb.log({f'Truly finetune/{start} test loss':target_test_loss,})#f'Truly finetune/{start} target fid':target_fid})
            print('*****check test loss*****', target_test_loss)
    return target_test_loss, model

def finetune_model_backup(model,config, target_trainloader,trainer, criterion, scheduler, optimizer,epochs):
    losses = AverageMeter()
    model.train()
    iters = len(target_trainloader.dataset) // config.batch_size
    iter_index = 1
    epoch = 0
    for epoch in range(1,epochs+1):
        for batch in target_trainloader:
            images, labels = batch['image'], batch['label']
            images = images.cuda()
            labels = None
            with autocast():
                pred_noise, noise = trainer(model,images,class_label=labels)
                loss = criterion(pred_noise, noise)
            if loss == 0. or torch.any(torch.isinf(loss)) or torch.any(
                torch.isnan(loss)):
                optimizer.zero_grad()
                continue
            config.scaler.scale(loss).backward() 
            if hasattr(config,
                        'clip_max_norm') and config.clip_max_norm > 0:
                import pdb;pdb.set_trace()
                config.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                config.clip_max_norm)
            # import pdb;pdb.set_trace()
            config.scaler.step(optimizer) 
            config.scaler.update()
            optimizer.zero_grad()
            losses.update(loss, images.size(0))
            scheduler.step(optimizer, iter_index / iters + (epoch - 1))
            iter_index += 1
            avg_loss = losses.avg
            avg_loss = avg_loss 
    return model
        
def test_finetune_fid(model, fid_model, target_testloader, sampler, config):
    model.eval()
    test_images_dataloader, generate_images_dataloader, test_image_num, generate_image_num = generate_diffusion_model_images(target_testloader, model, sampler, config)
    fid_value, is_score_mean, is_score_std=test(fid_model,test_images_dataloader, generate_images_dataloader,test_image_num, generate_image_num, config)
    return fid_value, is_score_mean, is_score_std

def test(fid_model,test_images_dataloader,generate_images_dataloader,test_image_num, generate_image_num, config):
    fid_model = fid_model.cuda()
    fid_value, is_score_mean, is_score_std = compute_diffusion_model_metric(
        test_images_dataloader, generate_images_dataloader, test_image_num,
        generate_image_num, fid_model, config)
    return fid_value, is_score_mean, is_score_std

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



def train_diffusion_model(batch, model, criterion, trainer, optimizer,
                          config):
    '''
    train diffusion model for one batch
    '''
    losses = AverageMeter()
    model.train()
    images, labels = batch['image'], batch['label']
    images = images.cuda()

    labels = None
    pred_noise, noise = trainer(model,
                                images,
                                class_label=labels)
    loss = criterion(pred_noise, noise)
    if loss == 0. or torch.any(torch.isinf(loss)) or torch.any(torch.isnan(loss)):
        optimizer.zero_grad()
    loss.backward() 
    optimizer.step()
    optimizer.zero_grad()
    losses.update(loss, images.size(0))
    avg_loss = losses.avg
    return avg_loss


def combined_train_diffusion_model(batchori, batchtar, model, criterion, trainer, optimizer,
                          config):
    '''
    train diffusion model for one batch
    '''
    losses = AverageMeter()

    model.train()
    images = batchori['image']
    images = images.cuda()
    pred_noise, noise = trainer(model,
                                images,
                                class_label=None)
    loss1 = criterion(pred_noise, noise)

    images = batchtar['image']
    images = images.cuda()
    pred_noise, noise = trainer(model,
                                images,
                                class_label=None)
    loss2 = criterion(pred_noise, noise)
    loss = 0.5*loss1 - 0.5*loss2
    config.scaler.scale(loss).backward() 
    if hasattr(config,
                'clip_max_norm') and config.clip_max_norm > 0:
        config.scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(),
                                        config.clip_max_norm)
    config.scaler.step(optimizer)
    config.scaler.update()
    optimizer.zero_grad()
    losses.update(loss, images.size(0))
    avg_loss = losses.avg
    avg_loss = avg_loss 
    return avg_loss
