# Sophon - Non-finetunable-learning



## Table of Contents

+ [Install](https://github.com/Sophon-NonFinetunableLearning/Sophon/blob/main/Readme.md#install)

+ [Usage](https://github.com/Sophon-NonFinetunableLearning/Sophon/blob/main/Readme.md#usage)

+ [Note](https://github.com/Sophon-NonFinetunableLearning/Sophon/blob/main/Readme.md#note)

  

## Install

You can build the required environment  by running:

```bash
conda env create -f environment.yml
```



## Usage

The whole project is devided into two parts: 	

+ classification : codes for reproducing our classification-related experiments
+ generation : codes for reproducing our generation-related experiments



### Classification task

Workspace is ``./classification``, thus

```bash
cd classification
```

#### Train Sophoned model

For inverse cross-entropy sophon, run:

```bash
python inverse_loss.py --alpha 3 --beta 5 --dataset CIFAR10 --arch res50
```

The output ckpt will be saved to `results/inverse_loss/[args.arch]_[args.dataset]/[current_time]/`

For kl divergence from uniform distribution sophon, run:

```bash
python kl_uniform_loss.py.py --alpha 1 --beta 1 --nl 5 --dataset CIFAR10 --arch res50
```

The choices of ``args.dataset`` are ``[CIFAR10, CINIC, SVHN, STL, MNIST]``

The choices of ``args.arch`` are ``[caformer, res50, res34, res18, vgg]``

The output ckpt will be saved to `results/kl_loss/[args.arch]_[args.dataset]/[current_time]/`



#### Test finetune

For test a target ckpt's finetune outcome directly:

```bash
# for finetuned ckpt
python finetune_test.py --start sophon --path path_to_ckpt

# for normal pretrained
python finetune_test.py --start normal

# for train from scratch
python finetune_test.py --start scratch
```



### Generation task

Workspace is ``./generation``, thus:

```bash
cd generation
```

#### Train Sophoned model

For mean squred loss sophon, run:

```bash
python ./mean_squared_loss.py --alpha 1.0 --beta 5.0 --bs 100 --fast_batches 50 --ml 1 --nl 1
```

The output ckpts will be saved to: `./res/mean_squared_loss_celeba/[current time]/`

For denial service loss sophon, run:

```bash
python denial_service_loss.py --alpha 0.05 --beta 2 --nl 10 --total 200
```

The output ckpts will be saved to: `./res/denial_service_loss_celeba/[current time]/`



#### Test finetune

For test a target ckpt's finetune outcome directly:

##### For any sophoned or other processed ckpt

run:

 ```bash
 # for mean squared loss test
 python mean_squared_loss.py --finetune path_to_ckpt 
 
 # for denial service loss test
 python denial_service_loss.py --finetune path_to_ckpt  
 ```

##### For two baselines: normal pretrained ckpt or train from scratch

run:

```bash
# for normal pretrained baseline
python mean_squared_loss.py --pretest scratch

# for train from scratch baseline
python mean_squared_loss.py --pretest pretrained
```










