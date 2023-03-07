import torch
import numpy as np
import torch.nn.functional as F


# def update_total(total, grads_before, grads, alpha, op):
#     if total is None:
#         if op == "add":
#             total = (alpha * torch.sign(grads_before)) * \
#                 (grads_before + grads) / 2
#         else:
#             total = -(alpha * torch.sign(grads_before)) * \
#                 (grads_before + grads) / 2
#     else:
#         if op == "add":
#             total += (alpha * torch.sign(grads_before)) * \
#                 (grads_before + grads) / 2
#         else:
#             total -= (alpha * torch.sign(grads_before)) * \
#                 (grads_before + grads) / 2
#     return total


def caculate_total(model, x, label, delta_x, y, steps=5, alpha=0.0025, op="add", num_classes=10):
    total = None
    delta_x = delta_x.clone()
    loss_func = torch.nn.CrossEntropyLoss(reduction='sum')
    x.requires_grad_(False)
    delta_x.requires_grad_(True)
    model.zero_grad()
    delta_x = torch.nn.Parameter(data=delta_x).requires_grad_(True)
    for _ in range(steps):
        outputs = model(x + delta_x)
        if op == "minus":
            # loss = -outputs[:, label]
            loss = loss_func(outputs, y.argmax(-1).squeeze(0))
            loss.backward(retain_graph=True)
            grad = delta_x.grad
            delta_x = delta_x - alpha * torch.sign(grad)
        elif op == "add":
            # loss = -outputs[:, label]
            loss = loss_func(outputs, y.argmax(-1).squeeze(0))
            loss.backward(retain_graph=True)
            grad = delta_x.grad
            delta_x = delta_x + alpha * torch.sign(grad)
        if total is None:
            total = ((alpha * torch.sign(grad)) * grad).detach()
        else:
            total += ((alpha * torch.sign(grad)) * grad).detach()
        delta_x = delta_x.detach().requires_grad_(True)
        model.zero_grad()
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


def exp(model, x, label, delta_x, y, add_steps=5, minus_steps=0, alpha=0.0025, method="total*delta"):
    if add_steps != 0:
        total_add = caculate_total(
            model, x.unsqueeze(0), label.argmax(-1), delta_x.unsqueeze(0), y.unsqueeze(0), steps=add_steps, op="add", alpha=alpha, num_classes=y.shape[-1])
    if minus_steps != 0:
        total_minus = caculate_total(
            model, x.unsqueeze(0), label.argmax(-1), delta_x.unsqueeze(0), y.unsqueeze(0), steps=minus_steps, op="minus", alpha=alpha, num_classes=y.shape[-1])
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
    return flag_norm, flag_delta
