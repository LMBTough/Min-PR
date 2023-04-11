import argparse
from models.resnet import resnet50
from datasets import load_cifar10, load_cifar100, load_imagenet
import torch
import numpy as np
from foolbox.attacks import LinfPGD, L2PGD, fast_gradient_method
from foolbox import PyTorchModel
from tqdm import tqdm
from sigmoid_method import exp as sigmoid_exp
from combine_method import exp as combine_exp
from utils import setup_seed, AverageMeter
setup_seed(3407)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

args = argparse.ArgumentParser()
args.add_argument("--dataset", type=str, default="cifar10")
args.add_argument("--method", type=str, default="sigmoid-1",
                  choices=["sigmoid-1", "sigmoid-2", "taylor", "total*delta", "total", "OBD"])
args.add_argument("--attack", type=str, default="linfpgd",
                  choices=["linfpgd", "l2pgd","fgsm"])
args.add_argument("--pic_num", type=int, default=10)
args.add_argument("--alpha", type=float, default=0.001)
args.add_argument("--lbda", type=float, default=0.01)
args.add_argument("--attack_step", type=int, default=40)
args.add_argument("--attack_step_size", type=float, default=0.01 / 0.3)
args.add_argument("--update_step", type=int, default=5)
args.add_argument("--add_step", type=int, default=2)
args.add_argument("--minus_step", type=int, default=2)
args.add_argument("--use_label", action="store_true")
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
    pt_model = PyTorchModel(model, bounds=(data_min-1e-8, data_max+1e-8))
    if args.method == "sigmoid-1":
        def exp(x, label, delta_x, y): return sigmoid_exp(model, x, delta_x, y,
                                                          steps=args.update_step, alpha=args.alpha, lbda=args.lbda, kind=1)
    elif args.method == "sigmoid-2":
        def exp(x, label, delta_x, y): return sigmoid_exp(model, x, delta_x, y,
                                                          steps=args.update_step, alpha=args.alpha, lbda=args.lbda, kind=2)
    elif args.method == "taylor" or args.method == "total*delta" or args.method == "total" or args.method == "OBD":
        def exp(x, label, delta_x, y): return combine_exp(model, x, label, delta_x, y, add_steps=args.add_step,
                                                          minus_steps=args.minus_step, alpha=args.alpha, lambda_r=args.lbda, method=args.method, use_label=args.use_label)

    count = 0
    affected = 0
    total = 0
    pbar = tqdm(total=args.pic_num)
    data = np.load(f"attack_result/{args.dataset}_{args.attack}_0.3.npz")
    success_x = data["success_x"]
    success_adv_data = data["success_adv_data"]
    success_label = data["success_label"]
    success_y = data["success_y"]
    avg = AverageMeter()
    if args.dataset == "cifar10" or args.dataset == "cifar100":
        batch_size = 128
    elif args.dataset == "imagenet":
        batch_size = 64
    for i in range(0, len(success_x), batch_size):
        batch_x = success_x[i:i+batch_size]
        batch_label = success_label[i:i+batch_size]
        batch_adv_data = success_adv_data[i:i+batch_size]
        batch_y = success_y[i:i+batch_size]

        batch_x = torch.from_numpy(batch_x).float().to(device)
        batch_label = torch.from_numpy(batch_label).long().to(device)
        batch_adv_data = torch.from_numpy(batch_adv_data).float().to(device)
        batch_y = torch.from_numpy(batch_y).float().to(device)

        for x, label, delta_x, y in zip(batch_x, batch_label, batch_adv_data-batch_x, batch_y):
            delta_x_norm = torch.norm(delta_x)
            try:
                norm, delta_cg = exp(x=x, delta_x=delta_x, y=y, label=label)
                affected += (delta_cg.cpu().detach().numpy() != 0).sum()
                total += delta_cg.cpu().detach().numpy().size
                avg.calculate(norm, delta_x_norm)
            except:
                continue
            pbar.update(1)
            count += 1
            if count == args.pic_num:
                break
        if count == args.pic_num:
            break
    print(f"reduced {avg.avg}")
    print(f"affected rate {affected / total}")
