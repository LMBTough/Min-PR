import torch
import numpy as np
import torch.nn.functional as F


def update_total(total, grads_before, grads, alpha, op):
    if total is None:
        if op == "add":
            total = (alpha * torch.sign(grads_before)) * \
                (grads_before + grads) / 2
        else:
            total = -(alpha * torch.sign(grads_before)) * \
                (grads_before + grads) / 2
    else:
        if op == "add":
            total += (alpha * torch.sign(grads_before)) * \
                (grads_before + grads) / 2
        else:
            total -= (alpha * torch.sign(grads_before)) * \
                (grads_before + grads) / 2
    return total


def caculate_total(model, x, delta_x, y, steps=5, alpha=0.0025, op="add", num_classes=10):
    total = None
    delta_x = delta_x.clone()
    # loss_func = torch.nn.CrossEntropyLoss(reduction='sum')
    x.requires_grad_(False)
    delta_x.requires_grad_(True)
    model.zero_grad()
    total_all_classes = list()
    class_weights = list()
    for class_ in range(num_classes):
        start_loss = None
        end_loss = None
        grads_before = None
        total = None
        grads = None
        delta_x_cls = torch.nn.Parameter(data=delta_x)
        for step in range(steps + 1):
            outputs = model(x + delta_x_cls)
            # loss = loss_func(outputs, torch.LongTensor([class_]).to(outputs.device))
            if op == "minus":
                loss = -outputs[:,class_]
                loss.backward(retain_graph=True)
                grad = delta_x_cls.grad
                delta_x_cls = delta_x_cls - alpha * torch.sign(grad)
            elif op == "add":
                loss = -outputs[:,class_]
                loss.backward(retain_graph=True)
                grad = delta_x_cls.grad
                delta_x_cls = delta_x_cls + alpha * torch.sign(grad)
            if step == 0:
                start_loss = loss.item()
                grads_before = grad
            else:
                grads = grad
                total = update_total(total, grads_before, grads, alpha, op)
                grads_before = grad
                end_loss = loss.item()
            delta_x_cls = delta_x_cls.detach().requires_grad_(True)
            model.zero_grad()
        class_weight = end_loss - start_loss
        total_all_classes.append(total.cpu().detach().numpy())
        class_weights.append(class_weight)
    # class_weights = np.abs(np.array(class_weights))
    # total = np.array(total_all_classes)
    # class_weights_ = np.array([-1/(num_classes-1)]*num_classes)
    # class_weights_[y.argmax(-1)] = 1
    # class_weights = class_weights_ / class_weights * F.softmax(y.squeeze(),dim=-1).cpu().detach().numpy()
    # print(class_weights)
    # total = np.average(total_all_classes,weights=class_weights,axis=0)
    total = np.average(total_all_classes,axis=0)
    total = torch.from_numpy(total).to(outputs.device)
    return total


def caculate_combine(total, delta, use_total=True, use_delta=True):
    if use_total and use_delta:
        combine = torch.abs(torch.abs(total) * delta)
    elif use_total:
        combine = torch.abs(total)
    elif use_delta:  # taylor
        combine = torch.abs(delta)
    combine = combine.unsqueeze(0).cpu().detach().numpy()
    combine_flatten = combine.flatten()
    return combine, combine_flatten


def get_result(model, x, pos, combine, combine_flatten, delta):
    threshold = np.sort(combine_flatten)[pos]
    delta_ = delta.clone()
    delta_[combine < threshold] = 0
    result = model(x+delta_).argmax(-1)
    return result, torch.norm(delta_).item(), delta_


def binary_search(model, x, delta, y, r, combine, combine_flatten, search_times=10):
    l = 0
    r = r
    pos = int((l + r) / 2)
    y_label = y.argmax(-1)
    for _ in range(search_times):
        if l == r:
            break
        result, norm_delta, delta_ = get_result(
            model, x, pos, combine, combine_flatten, delta)
        if result == y_label:
            l = pos
            pos = int((pos + r) / 2)
            flag = pos
            flag_norm = norm_delta
            flag_delta = delta_
        else:
            r = pos
            pos = int((pos + l) / 2)
    return flag, flag_norm, flag_delta


def exp(model, x, delta_x, y, add_steps=5, minus_steps=0, alpha=0.0025, method="total*delta"):
    if add_steps != 0:
        total_add = caculate_total(
            model, x.unsqueeze(0), delta_x.unsqueeze(0), y.unsqueeze(0), steps=add_steps, op="add", alpha=alpha, num_classes=y.shape[-1])
    if minus_steps != 0:
        total_minus = caculate_total(
            model, x.unsqueeze(0), delta_x.unsqueeze(0), y.unsqueeze(0), steps=minus_steps, op="minus", alpha=alpha, num_classes=y.shape[-1])
    if add_steps != 0 and minus_steps != 0:
        total = total_add - total_minus
    elif add_steps != 0:
        total = total_add
    elif minus_steps != 0:
        total = total_minus
    delta_x = delta_x.unsqueeze(0)
    if method == "total*delta":
        combine, combine_flatten = caculate_combine(
            total, delta_x, use_total=True, use_delta=True)
    elif method == "total":
        combine, combine_flatten = caculate_combine(
            total, delta_x, use_total=True, use_delta=False)
    elif method == "taylor":
        combine, combine_flatten = caculate_combine(
            total, delta_x, use_total=False, use_delta=True)
    _, flag_norm, flag_delta = binary_search(model, x.unsqueeze(0), delta_x, y, len(
        combine_flatten) - 1, combine, combine_flatten, search_times=10)
    return flag_norm,flag_delta
