import argparse
from models.resnet import resnet50
from datasets import load_cifar10, load_cifar100, load_imagenet
import torch
import numpy as np
from foolbox.attacks import LinfPGD, L2PGD, fast_gradient_method
from foolbox import PyTorchModel
from tqdm import tqdm
from utils import setup_seed
import os
setup_seed(3407)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

args = argparse.ArgumentParser()
args.add_argument("--dataset", type=str, default="cifar10",
                  choices=["cifar10", "cifar100", "imagenet"])
args.add_argument("--attack", type=str, default="linfpgd",
                  choices=["linfpgd", "l2pgd", "fgsm"])
args.add_argument("--pic_num", type=int, default=1000)
args.add_argument("--epsilons", type=float, default=0.3)

if __name__ == "__main__":
    args = args.parse_args()
    if args.dataset == "cifar10":
        num_classes = 10
        train_dataloader, test_dataloader, data_min, data_max = load_cifar10()
    elif args.dataset == "cifar100":
        num_classes = 100
        train_dataloader, test_dataloader, data_min, data_max = load_cifar100()
    elif args.dataset == "imagenet":
        num_classes = 1000
        train_dataloader, test_dataloader, data_min, data_max = load_imagenet()
    model = resnet50(args.dataset)
    for param in model.parameters():
        param.requires_grad = False
    if args.attack == "linfpgd":
        attack = LinfPGD()
    elif args.attack == "l2pgd":
        attack = L2PGD()
    elif args.attack == "fgsm":
        attack = fast_gradient_method.LinfFastGradientAttack()
    pt_model = PyTorchModel(model, bounds=(data_min-1e-8, data_max+1e-8))
    result_dict = {
        "success_x": [],
        "success_adv_data": [],
        "success_label": [],
        "success_y": []
    }
    count = 0
    pbar = tqdm(total=args.pic_num)
    for x, label in test_dataloader:
        x, label = x.to(device), label.to(device)
        pred = model(x)
        correct = pred.argmax(-1) == label
        x = x[correct]
        label = label[correct]
        if args.attack != "advGan":
            _, adv_data, success = attack(
                pt_model, x, label, epsilons=args.epsilons)
            adv_pred = model(adv_data)
        else:
            adv_data = attack.generate(x)
            adv_pred = model(adv_data)
            success = adv_pred.argmax(-1) != label
        success_x = x[success]
        success_adv_data = adv_data[success]
        success_label = label[success]
        success_y = adv_pred[success]
        result_dict["success_x"].append(success_x.cpu().detach().numpy())
        result_dict["success_adv_data"].append(
            success_adv_data.cpu().detach().numpy())
        result_dict["success_label"].append(
            success_label.cpu().detach().numpy())
        result_dict["success_y"].append(success_y.cpu().detach().numpy())
        pbar.update(len(success_x))
        count += len(success_x)
        if count >= args.pic_num:
            break
    pbar.close()
    result_dict["success_x"] = np.concatenate(
        result_dict["success_x"], axis=0)[:args.pic_num]
    result_dict["success_adv_data"] = np.concatenate(
        result_dict["success_adv_data"], axis=0)[:args.pic_num]
    result_dict["success_label"] = np.concatenate(
        result_dict["success_label"], axis=0)[:args.pic_num]
    result_dict["success_y"] = np.concatenate(
        result_dict["success_y"], axis=0)[:args.pic_num]
    os.makedirs("attack_result", exist_ok=True)
    np.savez(
        f"attack_result/{args.dataset}_{args.attack}_{args.epsilons}.npz", **result_dict)
