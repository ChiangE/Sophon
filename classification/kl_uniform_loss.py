
import random
import numpy as np
from torch.utils.data import DataLoader
import os
from datetime import datetime
import argparse
import json
import sys
import torch.nn.functional as F
sys.path.append('../')
def args_parser():
    parser = argparse.ArgumentParser(description='train N shadow models')
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--bs', default=150, type=int)
    parser.add_argument('--ml_loop', default=1, type=int)
    parser.add_argument('--nl_loop', default=1, type=int)
    parser.add_argument('--total_loop', default=1000, type=int)
    parser.add_argument('--alpha', default=3.0, type=float, help='coefficient of maml lr')
    parser.add_argument('--beta', default=5.0, type=float, help='coefficient of natural lr')
    parser.add_argument('--test_iterval', default=10, type=int)
    parser.add_argument('--arch', default='', type=str)
    parser.add_argument('--gpus', default='0,1', type=str)
    parser.add_argument('--dataset', default='CIFAR10', type=str, choices=['CIFAR10', 'MNIST', 'SVHN', 'STL', 'CINIC'])
    parser.add_argument('--finetune_epochs', default=1, type=int)
    parser.add_argument('--truly_finetune_epochs', default=20, type=int)
    parser.add_argument('--finetune_lr', default=0.0001, type=float)
    parser.add_argument('--fast_lr', default=0.0001, type=float)
    parser.add_argument('--root', default='results', type=str) 
    parser.add_argument('--notes', default=None, type=str)
    parser.add_argument('--seed', default=99, type=int)
    parser.add_argument('--partial', default='no', type=str, help='whether only use last ten batch to maml')
    parser.add_argument('--adaptation_steps', default=50, type=int)
    args = parser.parse_args()
    return args
args = args_parser()
if args.gpus:
    gpu_list = args.gpus.split(',')
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(gpu_list)
    devices_id = [id for id in range(len(gpu_list))]
from utils import save_bn, load_bn, check_gradients, accuracy, get_pretrained_model, test_original, test, initialize00, set_seed, save_data, get_finetuned_model
from tqdm import tqdm
import torch
from torch import nn, optim
from utils import get_dataset
import wandb
import learn2learn as l2l
import copy
import timm

def fast_adapt_multibatch(batches, learner, loss, shots, ways, device):
    # Adapt the model
    test_loss = 0
    test_accuracy = 0
    total_test = 0
    for index,batch in enumerate(batches):
        data, labels = batch
        data, labels = data.to(device), labels.to(device)
        adaptation_indices = np.zeros(data.size(0), dtype=bool)
        # adaptation_indices[np.arange(shots*ways)] = True
        adaptation_indices[np.random.choice(np.arange(data.size(0)), shots*ways, replace=False)] = True
        evaluation_indices = torch.from_numpy(~adaptation_indices)
        adaptation_indices = torch.from_numpy(adaptation_indices)
        adaptation_data, adaptation_labels = data[adaptation_indices], labels[adaptation_indices]
        evaluation_data, evaluation_labels = data[evaluation_indices], labels[evaluation_indices]
        current_test = evaluation_data.shape[0]
        # print(current_test)
        total_test += current_test
        adaptation_error = loss(learner(adaptation_data), adaptation_labels)
        if index == 0:
            current_grads = learner.adapt(adaptation_error,None) 
        else:
            last_grads = current_grads
            current_grads = learner.adapt(adaptation_error,last_grads) 
    # Evaluate the adapted model
        predictions = learner(evaluation_data)
        normalized_preds = torch.nn.functional.softmax(predictions, dim=1).cuda()
        target_preds = 0.1 * torch.ones((predictions.shape[0], predictions.shape[1])).cuda()
        evaluation_error = F.kl_div(torch.log(normalized_preds), target_preds, reduction='batchmean')
        evaluation_accuracy = accuracy(predictions, evaluation_labels)
        test_loss += evaluation_error*current_test
        test_accuracy += evaluation_accuracy*current_test
    return test_loss*1.0/total_test, test_accuracy*1.0/total_test 


def partial_fast_adapt_multibatch(batches, learner, loss, shots, ways, device):
    # Adapt the model
    test_loss = 0
    test_accuracy = 0
    total_test = 0
    for index,batch in enumerate(batches):
        if index <=40:
            data, labels = batch
            data, labels = data.to(device), labels.to(device)
            adaptation_indices = np.zeros(data.size(0), dtype=bool)
            adaptation_indices[np.arange(shots*ways)] = True
            adaptation_indices = torch.from_numpy(adaptation_indices)
            adaptation_data, adaptation_labels = data[adaptation_indices], labels[adaptation_indices]
            # print(current_test)
            adaptation_error = loss(learner(adaptation_data), adaptation_labels)
            if index == 0:
                current_grads = learner.adapt(adaptation_error,None) 
            else:
                last_grads = current_grads
                current_grads = learner.adapt(adaptation_error,last_grads) 
        
        else:
            data, labels = batch
            data, labels = data.to(device), labels.to(device)
            adaptation_indices = np.zeros(data.size(0), dtype=bool)
            adaptation_indices[np.arange(shots*ways)] = True
            evaluation_indices = torch.from_numpy(~adaptation_indices)
            adaptation_indices = torch.from_numpy(adaptation_indices)
            adaptation_data, adaptation_labels = data[adaptation_indices], labels[adaptation_indices]
            evaluation_data, evaluation_labels = data[evaluation_indices], labels[evaluation_indices]
            current_test = evaluation_data.shape[0]
            # print(current_test)
            total_test += current_test
            adaptation_error = loss(learner(adaptation_data), adaptation_labels)
            if index == 0:
                current_grads = learner.adapt(adaptation_error,None) 
            else:
                last_grads = current_grads
                current_grads = learner.adapt(adaptation_error,last_grads) 
        # Evaluate the adapted model
            predictions = learner(evaluation_data)
            evaluation_error = loss(predictions, evaluation_labels)

            evaluation_accuracy = accuracy(predictions, evaluation_labels)
            test_loss += evaluation_error*current_test
            test_accuracy += evaluation_accuracy*current_test
    return test_loss*1.0/total_test, test_accuracy*1.0/total_test 

def test_finetune(model, trainset, testset, epochs, lr):
    model = nn.DataParallel(model)
    trainloader = DataLoader(trainset, batch_size=256, shuffle=True, num_workers=4,drop_last=True)
    testloader = DataLoader(testset, batch_size=256, shuffle=False, num_workers=4,drop_last=True)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    model.train()
    for ep in tqdm(range(epochs)):
        for inputs, targets in tqdm(trainloader):
            inputs, targets = inputs.cuda(), targets.cuda() 
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model.eval()
    acc, test_loss = test(model, testloader, torch.device('cuda'))
    return round(acc,2), round(test_loss,2)

def test_finetune_final(mode, model, trainset, testset, epochs, lr):
    model = nn.DataParallel(model)
    trainloader = DataLoader(trainset, batch_size=256, shuffle=True, num_workers=4,drop_last=True)
    testloader = DataLoader(testset, batch_size=256, shuffle=False, num_workers=4,drop_last=True)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    model.train()
    # epochs = 1
    for ep in tqdm(range(epochs)):
        model.train()
        for inputs, targets in tqdm(trainloader):
            inputs, targets = inputs.cuda(), targets.cuda()  
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # scheduler.step()
        test_acc, test_loss = test(model, testloader, torch.device('cuda'))
        wandb.log({f'{mode}: test accuracy':test_acc, f'{mode}: test loss':test_loss,})
    return round(test_acc,2), round(test_loss,2)

def save_args_to_file(args, file_path):
    with open(file_path, "w") as file:
        json.dump(vars(args), file, indent=4)

def main(
        args,
        ways=10,
        shots=24,
        adaptation_steps=100,
        cuda=True,
):  
    seed = args.seed if args.seed else random.randint(0,99)
    set_seed(seed)
    import socket
    hostname = socket.gethostname()
    print("Hostname:", hostname)
    ip_address = socket.gethostbyname(hostname)
    # print("IP Address:", ip_address)
    args.from_machine = ip_address
    # args.arch = 'caformer'

    wandb.init(
    project="Sophon classification",  
    entity="Sophon",
    config = args,
    name = f"{args.arch}_alpha{args.alpha}_beta{args.beta}_ml{args.ml_loop}_nl{args.nl_loop}_batches{args.adaptation_steps}" ,
    notes= args.notes,         
  
)   
    wandb.config.update(args)
    shots = int(args.bs * 0.9 / ways)
    print(f'shots is {shots}')
    device = torch.device('cpu')
    if cuda and torch.cuda.device_count():
        # torch.cuda.manual_seed(seed)
        device = torch.device('cuda')
    wandb.log({'seed':seed})
    save_path = args.root +'/'+args.arch+'_'+ args.dataset + '/'
    adaptation_steps = args.adaptation_steps
    now = datetime.now()
    save_path = save_path + '/' + f'{now.month}_{now.day}_{now.hour}_{now.minute}_{now.second}/'
    os.makedirs(save_path, exist_ok=True)
    wandb.log({'save path': save_path})
    save_args_to_file(args, save_path+"args.json")
    trainset_ori, testset_ori = get_dataset('ImageNet', '../../../datasets/', subset='imagenette', args=args)
    original_trainloader = DataLoader(trainset_ori, batch_size=args.bs, shuffle=True, num_workers=0)
    original_testloader = DataLoader(testset_ori, batch_size=args.bs, shuffle=False, num_workers=0)
    trainset_tar, testset_tar = get_dataset(args.dataset, '../../../datasets', args=args)
    target_trainloader = DataLoader(trainset_tar, batch_size=args.bs, shuffle=True, num_workers=0,drop_last=True)
    target_testloader = DataLoader(testset_tar, batch_size=args.bs, shuffle=False, num_workers=0,drop_last=True)
    original_iter = iter(original_trainloader)
    target_iter = iter(target_trainloader)


    queryset_loss = []
    queryset_acc = []
    originaltest_loss = []
    originaltrain_loss = []
    originaltest_acc = []
    finetuned_target_testacc = []
    finetuned_target_testloss = []
    final_original_testacc = []
    final_finetuned_testacc = []
    final_finetuned_testloss = []
    total_loop_index = []
    ml_index = []
    nl_index = []

    # Create model
    model = get_pretrained_model(args)
    model = nn.DataParallel(model)
    test_original(model, original_testloader, device)
    model0 = copy.deepcopy(model)
    means_original , vars_original = save_bn(model0)
    maml = l2l.algorithms.MAML(model, lr=args.fast_lr, first_order=True)
    maml_opt = optim.Adam(maml.parameters(), args.alpha*args.lr)
    criterion = nn.CrossEntropyLoss(reduction='mean')
    natural_optimizer = optim.Adam(maml.parameters(), args.beta*args.lr)
    maml_loop = 0
    natural_loop = 0 
    total_loop = args.total_loop 
    ### test the finetune
    # print('*********Test normal finetune***********')
    # test_model = copy.deepcopy(model.module)
    # acc, test_loss = test_finetune(test_model, trainset_tar, testset_tar, args.finetune_epochs, args.finetune_lr)
    # print(f'normal finetune outcome: test accuracy is {acc}, test loss is {test_loss}')
    # wandb.log({"Finetune outcome-test accuracy":acc, "Finetune outcome-test loss":test_loss})
    best = -1
    ### train maml
    for i in range(1,total_loop+1):
        print('\n\n')
        print(f'============================================================')
        print(f'TOTAL train loop:{i}')
        backup = copy.deepcopy(model)
        total_loop_index.append(i)
        for ml in range(args.ml_loop):
                print(f'---------Train MAML {ml}----------')
                maml_loop += 1
                ml_index.append(maml_loop)
                maml_opt.zero_grad()
                batches = []
                for _ in range(adaptation_steps):
                    try:
                        batch = next(target_iter)
                        batches.append(batch)
                    except StopIteration:
                        target_iter = iter(target_trainloader)
                        batch = next(target_iter)
                learner = maml.clone()
                means, vars  = save_bn(model)
                if args.partial == 'no':
                    evaluation_error, evaluation_accuracy = fast_adapt_multibatch(batches,
                                                                    learner,
                                                                    criterion,
                                                                    shots,
                                                                    ways,
                                                                    device)
                elif args.partial == 'yes':
                    evaluation_error, evaluation_accuracy = partial_fast_adapt_multibatch(batches,
                                                                    learner,
                                                                    criterion,
                                                                    shots,
                                                                    ways,
                                                                    device)       
                model.module.zero_grad()
                # evaluation_error = -evaluation_error
                evaluation_error.backward()
                nn.utils.clip_grad_norm_(maml.module.parameters(), max_norm=0.5, norm_type=2)
                avg_gradients = check_gradients(maml.module)
                # print(avg_gradients)
                # Print some metrics
                print('Query set loss', round(evaluation_error.item(),2))
                print('Query set accuracy', round(100*evaluation_accuracy.item(),2), '%')
                maml_opt.step()
                wandb.log({"Query set loss": evaluation_error.item(), "Query set accuracy": 100*evaluation_accuracy.item(), "Gradients after maml loop": round(avg_gradients,2)})
                queryset_loss.append(-evaluation_error)
                queryset_acc.append(100*evaluation_accuracy.item())
                model = load_bn(model, means, vars)
        for nl in  range(args.nl_loop):
            natural_loop += 1
            nl_index.append(natural_loop)
            print('\n')
            print(f'---------Train Original {nl}----------')
            torch.cuda.empty_cache()
            try:
                batch = next(original_iter)
            except StopIteration:
                original_iter = iter(original_trainloader)
                batch = next(original_iter)
            inputs, targets = batch
            inputs, targets = inputs.cuda(), targets.cuda()       
            # print(inputs.shape)
            natural_optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets) 
            loss.backward()
            avg_gradients = check_gradients(model)
            # print('check gradients!!!!!!!!!')
            # print(avg_gradients)
            print('Original train loss', round(loss.item(),2))
            originaltrain_loss.append(round(loss.item(),2))
            natural_optimizer.step()
            acc, loss = test_original(model, original_testloader, device)
            wandb.log({"Original test acc": acc, "Original test loss": loss, "Gradients after natural loop":avg_gradients})
            originaltest_loss.append(loss)
            originaltest_acc.append(acc)

        if acc <=80:
            model = copy.deepcopy(backup) #if acc boom; reroll to backup saved in last outerloop 
            break
        print('==========================================================') 

        if (i+1) %args.test_iterval == 0:
            print('*************test finetune outcome**************')
            ## test finetune outcome
            originalacc = acc
            test_model = copy.deepcopy(model.module)
            finetuneacc, finetunetest_loss = test_finetune(test_model, trainset_tar, testset_tar, args.finetune_epochs, args.finetune_lr)
            print(f'finetune outcome: test accuracy is{finetuneacc}, test loss is{finetunetest_loss}')  
            wandb.log({"Finetune outcome-test accuracy":finetuneacc, "Finetune outcome-test loss":finetunetest_loss})
            finetuned_target_testacc.append(finetuneacc)
            finetuned_target_testloss.append(finetunetest_loss)

            name = f'loop{i}_ori{round(originalacc,2)}_ft{round(finetuneacc,2)}_qloss{evaluation_error}.pt'
            torch.save({
            'model':model.state_dict(),
            'maml_lr': args.lr*args.alpha,
            'nt_lr': args.lr*args.beta,
            'lr': args.lr,
            'nl_loop': args.nl_loop,
            'ml_loop': args.ml_loop,
            'total_loop': args.total_loop,
            'batch_size': args.bs},save_path+'/'+name)
            # gain = originalacc-finetuneacc
            # if gain > best:
            #     best = gain
            #     torch.save({'model':model.state_dict()},save_path+'/'+f'best_{gain}_ori_{originalacc}_tar_{finetuneacc}.pt')
                
            print('************************************************')



## test the original accuracy
    print('===============Test original==============')
    model = load_bn(model, means, vars)
    test_acc,_ = test_original(model, original_testloader, device)
    final_original_testacc.append(test_acc)
## test finetune outcome
    print(f'**************Finally test truly finetune ({args.truly_finetune_epochs} epochs)***************')
    test_model2 = copy.deepcopy(model.module)
    finetune_test_acc, finetune_test_loss = test_finetune_final('our finetuned/not init fc',test_model2, trainset_tar, testset_tar, args.truly_finetune_epochs, args.finetune_lr)
    print(f'Finally finetune outcome: test accuracy is{finetune_test_acc}, test loss is{finetune_test_loss}')
    final_finetuned_testacc.append(finetune_test_acc)
    final_finetuned_testloss.append(finetune_test_loss)
## save model
    name = f'{round(test_acc,2)}_{round(finetune_test_acc,2)}_{round(finetune_test_loss,2)}.pt'
    torch.save({
        'model':model.state_dict(),
        'maml_lr': args.lr*args.alpha,
        'nt_lr': args.lr*args.beta,
        'lr': args.lr,
        'nl_loop': args.nl_loop,
        'ml_loop': args.ml_loop,
        'total_loop': args.total_loop,
        'batch_size': args.bs},save_path+'/'+name)
    print(f'Saving to {save_path}/{name}......')
    wandb.log({'Checkpoints': save_path+'/'+name})

    save_data(save_path, queryset_loss, queryset_acc, originaltest_loss, originaltrain_loss, originaltest_acc, finetuned_target_testacc, finetuned_target_testloss, final_original_testacc, final_finetuned_testacc, final_finetuned_testloss, total_loop_index, ml_index, nl_index)
    return save_path+'/'+name

if __name__ == '__main__':
    ckpt = main(args)
    
