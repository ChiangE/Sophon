import matplotlib.pyplot as plt
import os
import sys
import warnings
import learn2learn as l2l
from torch import nn, optim
import numpy as np
import random
import json
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
warnings.filterwarnings('ignore')
import wandb
from diffusion_model.models.diffusion_unet import DiffusionUNet

from diffusion_model.metrics.inception import InceptionV3
from datetime import datetime
import torchvision.transforms as transforms
import argparse
import math

def compute_beta_schedule(beta_schedule_mode, t, linear_beta_1, linear_beta_t,
                          cosine_s):
    assert beta_schedule_mode in [
        'linear',
        'cosine',
        'quad',
        'sqrt_linear',
        'const',
        'jsd',
        'sigmoid',
    ]

    if beta_schedule_mode == 'linear':
        betas = torch.linspace(linear_beta_1,
                               linear_beta_t,
                               t,
                               requires_grad=False,
                               dtype=torch.float64)
    elif beta_schedule_mode == 'cosine':
        x = torch.arange(t + 1, requires_grad=False, dtype=torch.float64)
        alphas_cumprod = torch.cos(
            ((x / t) + cosine_s) / (1 + cosine_s) * math.pi * 0.5)**2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = torch.clip(betas, 0, 0.999)
    elif beta_schedule_mode == 'quad':
        betas = (torch.linspace(linear_beta_1**0.5,
                                linear_beta_t**0.5,
                                t,
                                requires_grad=False,
                                dtype=torch.float64)**2)
    elif beta_schedule_mode == 'sqrt_linear':
        betas = torch.linspace(linear_beta_1,
                               linear_beta_t,
                               t,
                               requires_grad=False,
                               dtype=torch.float64)**0.5
    elif beta_schedule_mode == 'const':
        betas = linear_beta_t * torch.ones(
            t, requires_grad=False, dtype=torch.float64)
    elif beta_schedule_mode == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / torch.linspace(
            t, 1, t, requires_grad=False, dtype=torch.float64)
    elif beta_schedule_mode == 'sigmoid':
        betas = torch.linspace(-6,
                               6,
                               t,
                               requires_grad=False,
                               dtype=torch.float64)
        betas = torch.sigmoid(betas) * (linear_beta_t -
                                        linear_beta_1) + linear_beta_1

    return betas
def extract(v, t, x_shape):
    '''
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    '''
    device = t.device
    v = v.to(device)
    out = torch.gather(v, index=t, dim=0).float().to(device)
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))

class DDPMTester(nn.Module):

    def __init__(self,
                 beta_schedule_mode='linear',
                 linear_beta_1=1e-4,
                 linear_beta_t=0.02,
                 cosine_s=0.008,
                 t=1000):
        super(DDPMTester, self).__init__()
        assert beta_schedule_mode in [
            'linear',
            'cosine',
            'quad',
            'sqrt_linear',
            'const',
            'jsd',
            'sigmoid',
        ]

        self.t = t

        self.beta_schedule_mode = beta_schedule_mode
        self.linear_beta_1 = linear_beta_1
        self.linear_beta_t = linear_beta_t
        self.cosine_s = cosine_s

        self.betas = compute_beta_schedule(self.beta_schedule_mode, self.t,
                                           self.linear_beta_1,
                                           self.linear_beta_t, self.cosine_s)

        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. -
                                                        self.alphas_cumprod)

    # forward diffusion (using the nice property): q(x_t | x_0)
    def add_noise(self, x_start, t, noise):
        # import pdb;pdb.set_trace() 
        # 
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t,
                                        x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        x_noisy = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

        return x_noisy


    def denoise(self, model, noisy_x, n_steps):
        cur_x = noisy_x.clone()
        x_seq = [cur_x]

        for i in reversed(range(n_steps)):
            cur_x = self.p_sample(model, cur_x, i)
            x_seq.append(cur_x)
        return x_seq
        
    def p_sample(self, model, noisy_x, t):
        noisy_x, t = noisy_x.cuda(), t
        self.betas = self.betas.cuda()
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.cuda()

        t = t * torch.ones(noisy_x.shape[0], dtype=torch.int64).cuda()
        eps_theta = model(noisy_x.float(), t)
        
        # import pdb;pdb.set_trace()
        # print(t.shape)
        print('beta', self.betas[t].shape)
        print('sqrt', self.sqrt_one_minus_alphas_cumprod[t].shape)

        coeff = self.betas[t] / self.sqrt_one_minus_alphas_cumprod[t]
        mean = (1 / (1 - self.betas[t]).sqrt()).reshape(-1,1,1,1) * (noisy_x - (coeff.reshape(-1,1,1,1) * eps_theta))
        z = torch.randn_like(noisy_x)
        sigma_t = self.betas[t].sqrt().reshape(-1,1,1,1)
        
        if t[0] == 0:
            sample = mean
        else:
            sample = mean + (sigma_t * z)
        return sample
    
    def forward(self, model, x_start, t, class_label=None):
        device = x_start.device
        # import pdb;pdb.set_trace()
        # t = torch.randint(0, self.t, size=(x_start.shape[0], )).to(device) #x_start:[256,3,32,32] t:[256]
        # t = torch.randint(1, 2, size=(x_start.shape[0], )).to(device) #x_start:[256,3,32,32] t:[256]

        noise = torch.randn_like(x_start).to(device)
        x_noisy = self.add_noise(x_start, t, noise)
        # pred_noise = model(x_noisy, t, class_label)
        # return pred_noise, noise, x_noisy
        return noise, x_noisy

def args_parser():
    parser = argparse.ArgumentParser(description='train N shadow models')
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--bs', default=200, type=int)
    parser.add_argument('--ml_loop', default=1, type=int)
    parser.add_argument('--nl_loop', default=1, type=int)
    parser.add_argument('--total_loop', default=1000, type=int)
    parser.add_argument('--alpha', default=0.1, type=float, help='coefficient of maml lr')
    parser.add_argument('--beta', default=5.0, type=float, help='coefficient of natural lr')
    parser.add_argument('--test_iterval', default=10, type=int)
    parser.add_argument('--gpus', default='0,1,2,3', type=str)
    parser.add_argument('--target', default='celeba', type=str)
    parser.add_argument('--truly_finetune_epochs', default=20, type=int)
    parser.add_argument('--note', default='', type=str)
    parser.add_argument('--name', default='', type=str)
    parser.add_argument('--seed', default=99, type=int)
    parser.add_argument('--timestep', default=20, type=int)
    parser.add_argument('--fast_batches', default=50, type=int)
    parser.add_argument('--pretest', default=None, type=str)
    parser.add_argument('--naive', default='no', type=str)
    parser.add_argument('--finetuned', default='', type=str)
    parser.add_argument('--test_original', default='no', type=str)
    parser.add_argument('--ckpt', default='', type=str, help='the ckpt of train from scratch')

    args = parser.parse_args()
    return args
args = args_parser()
if args.gpus:
    gpu_list = args.gpus.split(',')
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(gpu_list)
    devices_id = [id for id in range(len(gpu_list))]
import torch
from torch.utils.data import DataLoader
import sys
sys.path.append('../')
from utils import (set_seed, build_optimizer, Scheduler, build_training_mode, load_state_dict,final_testfinetune_lr, test_loss )
import torch
import copy
import os
from datetime import datetime
from tqdm import tqdm

def final_testfinetune_lr(start, model, fid_model, sampler, config, target_trainloader, target_trainloaderfortest, target_testloader, trainer, criterion, scheduler, optimizer,epochs):
    # train_losses = AverageMeter()
    target_train_loss=test_loss(model, target_trainloaderfortest, criterion, trainer, torch.device('cuda'))
    
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
        if (epoch % 1 == 0) or epoch == epochs :
            target_test_loss=test_loss(model, target_testloader, criterion, trainer, torch.device('cuda'))
            # target_fid, _, _=test_finetune_fid(model, fid_model, target_trainloaderfortest, sampler, config)
            print('*****check test loss*****', target_test_loss)
    return target_test_loss, model


def generateimg(args, name, model, trainer, testloader):
    if args.test_original == 'yes':
        selected_indexes = [5,9,14,65,47]#[5,9,14,18,47]
    else:
        selected_indexes = [9, 12, 15, 16, 17]
    model.eval()
    model = model.cuda()
    now = datetime.now()
    save_dir = f'./generated/{name}_{args.timestep}_{args.test_original}/'  + f'{now.month}_{now.day}_{now.hour}_{now.minute}_{now.second}/'
    os.makedirs(save_dir, exist_ok=True)
    if args.finetuned:
        with open(save_dir+"ckpt.json", "w") as file:
            json.dump((args.finetuned), file, indent=4)
        
    with torch.no_grad():
        for batch in testloader:
            images, labels = torch.squeeze(batch['image'][[selected_indexes],:,:,:]), batch['label'][[selected_indexes]]
            # images, labels = torch.squeeze(batch['image'][selected_indexes]), batch['label'][[selected_indexes]]
            x_original = images.cuda()
            labels = None
            # with autocast():
            t = args.timestep * torch.ones(x_original.shape[0], dtype=torch.int64).cuda()
            # pred_noise, noises, noisy_x = trainer(model,x_original,class_label=labels)
            noises, noisy_x = trainer(model, x_original, t, class_label=labels)
            x_denoise = trainer.denoise(model, noisy_x, args.timestep)
            print('The number of steps', len(x_denoise))
            print('The shape of a batch of denoised images', x_denoise[0].shape)
            
            for step in [1,5,10,15,20]:
                for i, (x, x_noise, x_ori) in enumerate(zip(x_denoise[step], noisy_x, x_original)):
                        x = torch.clamp(x, 0, 1) 
                        image_denoise = x.detach().cpu().numpy()
                        plt.imshow(np.transpose(image_denoise, (1, 2, 0))) 
                        plt.axis('off')  
                        plt.savefig(save_dir+f'/denoised_image_{i}_step{step}.png', bbox_inches='tight', pad_inches=0.0)

                        if step == 20 & i == 0:
                            image_original = x_ori.detach().cpu().numpy()
                            plt.imshow(np.transpose(image_original, (1, 2, 0)))  
                            plt.axis('off')  
                            plt.savefig(save_dir+f'/original_image_{i}.png', bbox_inches='tight', pad_inches=0.0)

                            image_noisy = x_noise.detach().cpu().numpy()
                            plt.imshow(np.transpose(image_noisy, (1, 2, 0))) 
                            plt.axis('off') 
                            plt.savefig(save_dir+f'/noisy_image_{i}.png', bbox_inches='tight', pad_inches=0.0)
                print(f"Saving generated images to {save_dir} !")
            exit()
    exit()


def main(
        args,
        ways=10,
        shots=24,
        meta_lr=0.0001,
        fast_lr=0.1,
        meta_batch_size=32,
        adaptation_steps=50,
        num_iterations=60000,
        cuda=True,
        seed=42,
):
    import socket
    hostname = socket.gethostname()
    print("Hostname:", hostname)
    ip_address = socket.gethostbyname(hostname)
    # print("IP Address:", ip_address)
    args.from_machine = ip_address
    shots = int(args.bs * 0.9 / ways)
    if args.seed:
        seed = args.seed
    else:
        seed = random.randint(0,99)
    set_seed(seed)
    import socket
    hostname = socket.gethostname()
    print("Hostname:", hostname)
    ip_address = socket.gethostbyname(hostname)
    # print("IP Address:", ip_address)
    args.from_machine = ip_address
    name = f'{args.target}_alpha{args.alpha}_beta{args.beta}_fastbatches{args.fast_batches}_ml{args.ml_loop}_nl{args.nl_loop}' if not args.name else args.name
    # wandb.init(
    # project="generate image",  
    # entity="pangpang",
    # config = args,
    # notes=args.note,
    # name = name
    # )   
    # wandb.config.update(args)
    # wandb.log({'seed':seed})
    torch.cuda.empty_cache()
    args.work_dir = f'./res/zeros_{args.target}'
    sys.path.append(args.work_dir)
    now = datetime.now()
    save_path = args.work_dir + '/' + f'{now.year}_{now.month}_{now.day}_{now.hour}_{now.minute}_{now.second}/'
    # import pdb;pdb.set_trace()
    os.makedirs(save_path, exist_ok=True)
    # wandb.log({'save path': save_path})

    if args.target == 'celeba':
        from train_config_celeba_denoise import config
    elif args.target == 'ffhq':
        from train_config_ffhq_denoise import config

    checkpoint_dir = os.path.join(args.work_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    original_trainloader_fortest = DataLoader(config.original_trainset,
                              batch_size=2000,
                              shuffle=False,
                              pin_memory=True,
                              drop_last=True,
                              collate_fn=config.train_collater,
                              )
    original_testloader = DataLoader(config.original_testset,
                              batch_size=2000,
                              shuffle=False,
                              pin_memory=True,
                              drop_last=True,
                              collate_fn=config.train_collater,
                              )
    
    target_trainloader = DataLoader(config.target_trainset,
                              batch_size=args.bs,
                              shuffle=True,
                              pin_memory=True,
                              drop_last=True,
                              collate_fn=config.train_collater,
                              )
    target_trainloader_fortest = DataLoader(config.target_trainset,
                              batch_size=2000,
                              shuffle=False,
                              pin_memory=True,
                              drop_last=True,
                              collate_fn=config.train_collater,
                              )
    target_testloader = DataLoader(config.target_testset,
                              batch_size=100,
                              shuffle=False,
                              pin_memory=True,
                              drop_last=True,
                              collate_fn=config.train_collater,
                              )

    trained_path = './pretrained/checkpoints_cifar100/loss0.026.pth'
    model = copy.deepcopy(config.model).cuda()
    model, config.ema_model, config.scaler = build_training_mode(config, model)
    finetune_optimizer, model_layer_weight_decay_list = build_optimizer(config, model)
    sampler = config.sampler
    load_state_dict(trained_path, model)
    model = nn.DataParallel(model)
    trainer = DDPMTester()
    train_criterion = config.train_criterion.cuda()
    from multistep_train_config_celeba_denoise import config
    realtrainer = config.trainer.cuda()
    if args.pretest:
        if args.pretest == 'scratch':
            print('===============Test train from scratch==============')
            test_model3 = copy.deepcopy(config.model).cuda()
            print(all(torch.equal(p1,p2) for p1,p2 in zip(test_model3.parameters(), model.parameters())))
            # test_model3 = nn.DataParallel(test_model3)
            test_model3.load_state_dict(torch.load(args.ckpt)['model'])
            scheduler = Scheduler(config, finetune_optimizer)
            for param_group in finetune_optimizer.param_groups:
                param_group['lr'] *= 10
            # loss, test_model3 = final_testfinetune_lr('finetune v.s. train from scratch', test_model3, None, sampler, config, target_trainloader, target_trainloader_fortest, target_testloader, realtrainer, train_criterion, scheduler, finetune_optimizer, args.truly_finetune_epochs)
            # torch.save({'model':test_model3.state_dict(),},save_path+'/'+'start_scratch.pt')
            # print(f"saving model to {save_path+'/'+'start_scratch.pt'}")
            generateimg( args, 'scratch', test_model3, trainer, target_testloader)
        
        elif args.pretest == 'pretrained':
            print('===============Test finetune==============')
            test_model4 = copy.deepcopy(config.model).cuda()#test train from scratch
            trained_model_path = './pretrained/checkpoints_cifar100/loss0.026.pth'
            load_state_dict(trained_model_path, test_model4)
            print(all(torch.equal(p1,p2) for p1,p2 in zip(test_model4.parameters(), model.parameters())))
            scheduler = Scheduler(config, finetune_optimizer)
            # loss, test_model4 = final_testfinetune_lr('finetune v.s. train from scratch', test_model4, None, sampler, config, target_trainloader, target_trainloader_fortest, target_testloader, realtrainer, train_criterion, scheduler, finetune_optimizer, args.truly_finetune_epochs)
            # torch.save({'model':test_model4.state_dict(),},save_path+'/'+'start_pretrained.pt')
            # print(f"saving model to {save_path+'/'+'start_pretrained.pt'}")
            if args.test_original == 'yes':
                generateimg(args, 'pretrained', test_model4, trainer, original_testloader)
            else:
                generateimg(args, 'pretrained', test_model4, trainer, target_testloader)
        else:
            assert(0)
        print('Pretest finished!')
        exit()
    return model, config, target_trainloader, target_trainloader_fortest, target_testloader, original_testloader, trainer



if __name__ == '__main__':
    args = args_parser()
    # import os;os.environ["WANDB_MODE"] = "offline" 
    if args.pretest:
        args.name = args.pretest
    model, config, target_trainloader, target_trainloader_fortest, target_testloader, original_testloder, trainer = main(args)
    device = torch.device('cuda')
    print('*****************Final truly finetune test********************')
    print('-----test finetune -----')
    test_model = copy.deepcopy(model.module)

    if args.finetuned:

            test_model = copy.deepcopy(model)
            test_model.load_state_dict(torch.load(args.finetuned)['model'])
        

        # wandb.log({"finetuned model dir": args.finetuned})
    config.epochs = args.truly_finetune_epochs
    finetune_optimizer, model_layer_weight_decay_list = build_optimizer(config, test_model)
    scheduler = Scheduler(config, finetune_optimizer)

    if args.test_original == 'yes':
        targettestloss = generateimg(args, 'sophon', test_model, trainer, original_testloder)
    
    else:
        targettestloss = generateimg(args, 'sophon', test_model, trainer, target_testloader)

    # wandb.log({"Finetune target test loss": targettestloss})
    
   
