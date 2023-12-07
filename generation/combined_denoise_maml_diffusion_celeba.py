import os
import sys
import warnings
import learn2learn as l2l
from torch import nn, optim
import numpy as np
import random
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
warnings.filterwarnings('ignore')
import wandb
from diffusion_model.models.diffusion_unet import DiffusionUNet

from diffusion_model.metrics.inception import InceptionV3
from datetime import datetime
import torchvision.transforms as transforms
import argparse
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
    parser.add_argument('--target', default='CIFAR10', type=str)
    parser.add_argument('--truly_finetune_epochs', default=20, type=int)
    parser.add_argument('--note', default='', type=str)
    parser.add_argument('--name', default='', type=str)
    parser.add_argument('--seed', default=99, type=int)
    parser.add_argument('--pretest', default=None, type=str)
    parser.add_argument('--naive', default='no', type=str)

    args = parser.parse_args()
    return args
args = args_parser()
if args.gpus:
    gpu_list = args.gpus.split(',')
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(gpu_list)
    devices_id = [id for id in range(len(gpu_list))]
import time
from torch.cuda.amp import autocast
import torch
from torch.utils.data import DataLoader
import sys
sys.path.append('../')
from utils import (get_logger, set_seed, worker_seed_init_fn,ImagePathDataset,generate_diffusion_model_images,generate_diffusion_model_images_tensor,generate_diffusion_model_images,
                         build_optimizer, Scheduler, build_training_mode, compute_diffusion_model_metric, load_state_dict, test_loss, test_finetune_fid,
                         AverageMeter,test, train_diffusion_model,finetune_model, check_gradients, final_testfinetune_lr, final_testfinetune_aug, intermidiate_test, combined_train_diffusion_model )
from tqdm import tqdm
import torch
import copy
import os
import shutil

def fast_adapt(batch1, batch2, learner, loss, shots, ways,trainer, device, loop):
    # Adapt the model
    data, labels = batch1['image'], batch1['label']
    data, labels = data.to(device), labels.to(device)
    adaptation_indices = np.zeros(data.size(0), dtype=bool)
    adaptation_indices[np.arange(shots*ways)] = True
    evaluation_indices = torch.from_numpy(~adaptation_indices)
    adaptation_indices = torch.from_numpy(adaptation_indices)
    adaptation_data, adaptation_labels = data[adaptation_indices], labels[adaptation_indices]
    evaluation_data, evaluation_labels = data[evaluation_indices], labels[evaluation_indices]
    pred_noise, noise = trainer(learner,adaptation_data,class_label=None)
    adaptation_error = loss(pred_noise, noise)
    learner.adapt(adaptation_error) 
    pred_noise_eva, noise_eva = trainer(learner,evaluation_data,class_label=None)
    loss1 = loss(pred_noise_eva, noise_eva)


    
    data, labels = batch2['image'], batch2['label']
    data, labels = data.to(device), labels.to(device)
    pred_noise, noise = trainer(learner,adaptation_data,class_label=None)
    loss2 = loss(pred_noise, noise)

    print('Query set loss', round(-loss1.item(),2))
    wandb.log({"Query set loss": -loss1.item()}, step=loop)
    print(f'Original train loss {loss2}') 
    wandb.log({"Original train loss": loss2}, step=loop)
    return 0.5*loss2 - 0.5*loss1

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
    name = f'alpha{args.alpha}_beta{args.beta}_ml{args.ml_loop}_nl{args.nl_loop}' if not args.name else args.name
    wandb.init(
    # dir = 'maml_log',
    project="diffusion_celeba_denoise",  
    entity="pangpang",
    config = args,
    notes=args.note,
    name = name
    )   
    wandb.config.update(args)
    wandb.log({'seed':seed})
    torch.cuda.empty_cache()
    args.work_dir = './res/'
    sys.path.append(args.work_dir)
    now = datetime.now()
    save_path = args.work_dir + '/' + f'{now.year}_{now.month}_{now.day}_{now.hour}_{now.minute}_{now.second}/'
    # import pdb;pdb.set_trace()
    os.makedirs(save_path, exist_ok=True)
    wandb.log({'save path': save_path})
    from train_config_celeba_denoise import config
    checkpoint_dir = os.path.join(args.work_dir, 'checkpoints')
    resume_model = os.path.join(checkpoint_dir, 'latest.pth')
    os.makedirs(checkpoint_dir, exist_ok=True)
    original_trainloader = DataLoader(config.original_trainset,
                              batch_size=args.bs,
                              shuffle=True,
                              pin_memory=True,
                              drop_last=True,
                              collate_fn=config.train_collater,
                              )
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
                              batch_size=2000,
                              shuffle=False,
                              pin_memory=True,
                              drop_last=True,
                              collate_fn=config.train_collater,
                              )
    original_iter = iter(original_trainloader)
    target_iter = iter(target_trainloader)


    trained_path = './pretrained/checkpoints_cifar100/loss0.026.pth'
    model = copy.deepcopy(config.model).cuda()
    model, config.ema_model, config.scaler = build_training_mode(config, model)
    finetune_optimizer, model_layer_weight_decay_list = build_optimizer(config, model)
    sampler = config.sampler
    load_state_dict(trained_path, model)
    model = nn.DataParallel(model)
    train_criterion = config.train_criterion.cuda()
    trainer = config.trainer.cuda()
    fid_model = config.fid_model

    config.save_test_image_dir = os.path.join(config.save_image_dir,
                                              'test_images')
    config.save_generate_image_dir = os.path.join(config.save_image_dir,
                                                  'generate_images')
    for dir in [config.save_image_dir, config.save_test_image_dir, config.save_generate_image_dir]:
        os.makedirs(dir, exist_ok=True)

    maml = l2l.algorithms.MAML(model, lr=0.0001, first_order=True)
    maml_opt = optim.Adam(maml.parameters(), args.alpha*args.lr)
    natural_optimizer, _ = build_optimizer(config, model)
    for param_group in natural_optimizer.param_groups:
        param_group['lr'] = args.beta*args.lr


    if args.pretest:
        if args.pretest == 'scratch':
            print('===============Test train from scratch==============')
            test_model3 = copy.deepcopy(config.model).cuda()#test train from scratch  ###wocccc单纯这句话并不是在做初始化，效果会是取出来了上面装载过trained参数的那个model，太tmtmtm坑了. 只有最上面那个model用copy.deepcopy之后这里才是想要的初始化效果。又是tensor之间的地址链接！！！！
            print(all(torch.equal(p1,p2) for p1,p2 in zip(test_model3.parameters(), model.parameters())))
            test_model3, config.ema_model, config.scaler = build_training_mode(config, test_model3)
            config.epochs = args.truly_finetune_epochs
            finetune_optimizer, model_layer_weight_decay_list = build_optimizer(config, test_model3)
            scheduler = Scheduler(config, finetune_optimizer)
            loss = final_testfinetune_lr('finetune v.s. train from scratch', test_model3, fid_model, sampler, config, target_trainloader, target_trainloader_fortest, target_testloader, trainer, train_criterion, scheduler, finetune_optimizer, args.truly_finetune_epochs)
        elif args.pretest == 'pretrained':
            print('===============Test finetune==============')
            test_model4 = copy.deepcopy(config.model).cuda()#test train from scratch
            trained_model_path = './pretrained/checkpoints_cifar100/loss0.026.pth'
            load_state_dict(trained_model_path, test_model4)
            print(all(torch.equal(p1,p2) for p1,p2 in zip(test_model4.parameters(), model.parameters())))
            config.epochs = args.truly_finetune_epochs
            finetune_optimizer, model_layer_weight_decay_list = build_optimizer(config, test_model4)
            scheduler = Scheduler(config, finetune_optimizer)
            loss = final_testfinetune_lr('finetune v.s. train from scratch', test_model4, fid_model, sampler, config, target_trainloader, target_trainloader_fortest, target_testloader, trainer, train_criterion, scheduler, finetune_optimizer, args.truly_finetune_epochs)
        else:
            assert(0)
        print('Pretest finished!')
        exit()
    maml_loop = 0
    for i in range(args.total_loop):
        torch.cuda.empty_cache()
        print('\n\n')
        print(f'=============================TOTAL train loop:{i}===============================')
        for ml in range(args.ml_loop):
                print(f'---------Train MAML {ml}----------')
                maml_opt.zero_grad()
                maml_loop += 1
                try:
                    batch_original = next(original_iter)
                except StopIteration:
                    original_iter = iter(original_trainloader)
                    batch_original = next(original_iter)
                try:
                    batch_target = next(target_iter)
                except StopIteration:
                    target_iter = iter(target_trainloader)
                    batch_target = next(target_iter)

                learner = maml.clone()
                # import pdb;pdb.set_trace()
                loss = fast_adapt(batch_target, batch_original,
                                            learner,
                                            train_criterion,
                                            shots,
                                            ways,
                                            trainer,
                                            torch.device('cuda'),
                                            i)
                loss.backward()
                nn.utils.clip_grad_norm_(maml.module.parameters(), max_norm=0.5, norm_type=2)
                print('grad',check_gradients(maml.module))
                maml_opt.step()
                print('\n')        
        print(f'=============================Loop:{i} finished=============================') 
        
        if ((i+1) %args.test_iterval == 0) or (i+1 == args.total_loop):
            original_trainloss, original_testloss, target_trainloss, target_testloss = intermidiate_test(model, original_trainloader_fortest, original_testloader, target_trainloader_fortest, target_testloader, config, train_criterion, trainer, sampler, fid_model, save_path, i+1)
            name = f'epoch_{i}_oritrain_{original_trainloss}_oritest_{original_testloss}_tartrain_{target_trainloss}_tartest_{target_testloss}.pt'
            torch.save({'model':model.state_dict(),},save_path+'/'+name)

        # if ((i+1) %50 == 0) or i+1 == args.total_loop:
        #     print('-----test finetune -----')
        #     test_model = copy.deepcopy(model.module)
        #     config.epochs = args.truly_finetune_epochs
        #     finetune_optimizer, model_layer_weight_decay_list = build_optimizer(config, test_model)
        #     scheduler = Scheduler(config, finetune_optimizer)
        #     targettestloss = final_testfinetune_lr(f'Number {i} loop', test_model, fid_model, sampler, config, target_trainloader, target_trainloader_fortest, target_testloader, trainer, train_criterion, scheduler, finetune_optimizer, args.truly_finetune_epochs)
        #     wandb.log({"Finetune target test loss": targettestloss})
        #     name = f'loop_{i}_targettestloss_{targettestloss}.pt'
        #     torch.save({'model':model.state_dict(),},save_path+'/'+name)
    return model, config, fid_model, target_trainloader, target_trainloader_fortest, target_testloader, trainer, train_criterion, sampler



if __name__ == '__main__':
    args = args_parser()
    # import os;os.environ["WANDB_MODE"] = "offline" 
    model, config, fid_model, target_trainloader, target_trainloader_fortest, target_testloader, trainer, train_criterion, sampler = main(args)
    device = torch.device('cuda')
    print('*****************Final truly finetune test********************')
    print('-----test finetune -----')
    test_model = copy.deepcopy(model.module)
    config.epochs = args.truly_finetune_epochs
    finetune_optimizer, model_layer_weight_decay_list = build_optimizer(config, test_model)
    scheduler = Scheduler(config, finetune_optimizer)
    targettestloss = final_testfinetune_lr(f'final ours', test_model, fid_model, sampler, config, target_trainloader, target_trainloader_fortest, target_testloader, trainer, train_criterion, scheduler, finetune_optimizer, args.truly_finetune_epochs)
    wandb.log({"Finetune target test loss": targettestloss})
    
    
