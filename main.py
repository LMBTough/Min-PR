import argparse
from models.resnet import resnet50
from datasets import load_cifar10, load_cifar100, load_caltech_101
import torch
import torch.nn as nn
import numpy as np
import os
import torchvision
import pickle as pkl
from foolbox.attacks import LinfPGD, L2PGD, L2CarliniWagnerAttack,fast_gradient_method
from foolbox import PyTorchModel
from tqdm import tqdm
from sigmoid_method import exp as sigmoid_exp
from combine_method import exp as combine_exp
from fvw_method import exp as fvw_exp
from utils import setup_seed, AverageMeter
setup_seed(3407)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

args = argparse.ArgumentParser()
args.add_argument("--dataset", type=str, default="cifar10")
args.add_argument("--method", type=str, default="sigmoid-1",
                  choices=["sigmoid-1", "sigmoid-2", "taylor", "total*delta", "total"])
args.add_argument("--attack", type=str, default="linfpgd",
                  choices=["linfpgd", "l2pgd", "l2cwa", "fgsm"])
args.add_argument("--pic_num", type=int, default=10)
args.add_argument("--alpha", type=float, default=0.001)
args.add_argument("--lbda", type=float, default=0.01)
args.add_argument("--attack_step", type=int, default=40)
args.add_argument("--attack_step_size", type=float, default=0.01 / 0.3)
args.add_argument("--update_step", type=int, default=5)
args.add_argument("--add_step", type=int, default=2)
args.add_argument("--minus_step", type=int, default=2)
args.add_argument("--use_label", action="store_true")
args.add_argument("--fvw", action="store_true")


if __name__ == "__main__":
    args = args.parse_args()
    if args.dataset == "cifar10":
        train_dataloader, test_dataloader, data_min, data_max = load_cifar10()
        model = resnet50(args.dataset)
    elif args.dataset == "cifar100":
        train_dataloader, test_dataloader, data_min, data_max = load_cifar100()
        model = resnet50(args.dataset)
    elif args.dataset == "caltech101":
        train_dataloader, test_dataloader, data_min, data_max = load_caltech_101()
        model = torchvision.models.resnet50()
        model.fc = nn.Linear(2048, 101)
        model.load_state_dict(torch.load(
            "./weights/caltech101_resnet50_weights.pth"))
        model.to(device)
        model.eval()
    for param in model.parameters():
        param.requires_grad = False
    if args.attack == "linfpgd":
        attack = LinfPGD(steps=args.attack_step,
                         rel_stepsize=args.attack_step_size)
    elif args.attack == "l2pgd":
        attack = L2PGD(steps=args.attack_step,
                       rel_stepsize=args.attack_step_size)
    elif args.attack == "l2cwa":
        attack = L2CarliniWagnerAttack(
            steps=args.attack_step, stepsize=args.attack_step_size)
    elif args.attack == "fgsm":
        attack = fast_gradient_method.LinfFastGradientAttack()
    pt_model = PyTorchModel(model, bounds=(data_min-1e-8, data_max+1e-8))
    if args.method == "sigmoid-1":
        def exp(x, label, delta_x, y): return sigmoid_exp(model, x, delta_x, y,
                                                          steps=args.update_step, alpha=args.alpha, lbda=args.lbda, kind=1)
    elif args.method == "sigmoid-2":
        def exp(x, label, delta_x, y): return sigmoid_exp(model, x, delta_x, y,
                                                          steps=args.update_step, alpha=args.alpha, lbda=args.lbda, kind=2)
    elif args.method == "taylor" or args.method == "total*delta" or args.method == "total":
        if args.fvw:
            def exp(x, label, delta_x, y): return fvw_exp(model, x, label, delta_x, y, add_steps=args.add_step,
                                                          minus_steps=args.minus_step, alpha=args.alpha, method=args.method, use_label=args.use_label)
        else:
            def exp(x, label, delta_x, y): return combine_exp(model, x, label, delta_x, y, add_steps=args.add_step,
                                                              minus_steps=args.minus_step, alpha=args.alpha, lambda_r=args.lbda, method=args.method, use_label=args.use_label)

    count = 0
    pbar = tqdm(total=args.pic_num)
    avg = AverageMeter()
    for x, label in test_dataloader:
        x, label = x.to(device), label.to(device)
        pred = model(x)
        correct = pred.argmax(-1) == label
        x = x[correct]
        label = label[correct]
        _, adv_data, success = attack(pt_model, x, label, epsilons=0.5)
        adv_pred = model(adv_data)
        success_x = x[success]
        success_adv_data = adv_data[success]
        success_label = label[success]
        success_y = adv_pred[success]
        for x, label, delta_x, y in zip(success_x, success_label, success_adv_data-success_x, success_y):
            delta_x_norm = torch.norm(delta_x)
            norm, _ = exp(x=x, delta_x=delta_x, y=y, label=label)
            avg.calculate(norm, delta_x_norm)
            pbar.update(1)
            count += 1
            if count == args.pic_num:
                break
        if count == args.pic_num:
            break
    # print(avg.avg)
    log_file = open(f"logs/{args.dataset}_{args.method}_{args.attack}_use_label_{args.use_label}_fvw_{args.fvw}.txt", "a")
    log_file.write(str(avg.avg))
    log_file.close()
