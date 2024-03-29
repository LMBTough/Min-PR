import torch
import numpy as np


def caculate_total(model, x, label, delta_x, y, steps=5, alpha=0.0025, lambda_r=0.01, op="add", use_label=False, add_mask=False):
    total = None
    delta_x = delta_x.clone()
    delta_x = torch.nn.Parameter(data=delta_x)
    loss_func = torch.nn.CrossEntropyLoss(reduction='sum')
    x.requires_grad_(False)
    delta_x.requires_grad_(True)
    model.zero_grad()
    start_loss = None
    end_loss = None
    for _ in range(steps):
        outputs = model(x+delta_x)
        model.zero_grad()
        if use_label:
            loss = loss_func(outputs, label)
        else:
            loss = loss_func(outputs, y)
        regularization_loss = torch.norm(delta_x)
        if op == "minus":
            loss = loss + lambda_r * regularization_loss
            loss.backward(retain_graph=True)
            grad = delta_x.grad
            if add_mask:
                mask = ~((torch.sign(grad) >= 0) ^ (delta_x >= 0))
                delta_x = delta_x - alpha * torch.sign(grad) * mask
            else:
                delta_x = delta_x - alpha * torch.sign(grad)
        elif op == "add":
            loss = loss - lambda_r * regularization_loss
            loss.backward(retain_graph=True)
            grad = delta_x.grad
            if add_mask:
                mask = (torch.sign(grad) >= 0) ^ (delta_x >= 0)
                delta_x = delta_x + alpha * torch.sign(grad) * mask
            else:
                delta_x = delta_x + alpha * torch.sign(grad)
        delta_x = delta_x.detach().requires_grad_(True)
        if total == None:
            total = ((alpha * torch.sign(grad)) * grad).detach()
        else:
            total += ((alpha * torch.sign(grad)) * grad).detach()
        model.zero_grad()
        if start_loss == None:
            start_loss = loss.item()
        end_loss = loss.item()
    if op == "minus":
        return total, start_loss - end_loss
    elif op == "add":
        return total, end_loss - start_loss


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


def binary_search(model, x, label, delta, y, r, combine, combine_flatten, search_times=10):
    l = 0
    r = r
    pos = int((l + r) / 2)
    # y_label = y.argmax(-1)
    # label_label = label.argmax(-1)
    label_label = label
    for _ in range(search_times):
        if l == r:
            break
        result, norm_delta, delta_ = get_result(
            model, x, pos, combine, combine_flatten, delta)
        # print(result,pos,label_label)
        if result != label_label:
            l = pos
            pos = int((pos + r) / 2)
            flag = pos
            flag_norm = norm_delta
            flag_delta = delta_
        else:
            r = pos
            pos = int((pos + l) / 2)
    return flag, flag_norm, flag_delta


def compute_hessian(model, x, label, delta_x, y, use_label=False):
    total = None
    delta_x = delta_x.clone()
    delta_x = torch.nn.Parameter(data=delta_x)
    loss_func = torch.nn.CrossEntropyLoss(reduction='sum')
    x.requires_grad_(False)
    delta_x.requires_grad_(True)
    model.zero_grad()
    outputs = model(x+delta_x)
    if use_label:
        loss = loss_func(outputs, label)
    else:
        loss = loss_func(outputs, y)
    grads = torch.autograd.grad(
        loss, delta_x, retain_graph=True, create_graph=True)[0]
    hessian = torch.autograd.grad(grads, delta_x, retain_graph=True,
                                  create_graph=True, grad_outputs=torch.ones_like(grads))[0]
    return hessian


def exp(model, x, label, delta_x, y, add_steps=5, minus_steps=0, alpha=0.0025, lambda_r=0.01, method="total*delta", use_label=False):
    if method == "taylor":
        total = None
    elif method == "OBD":
        total = compute_hessian(model, x.unsqueeze(0), label.unsqueeze(
            0), delta_x.unsqueeze(0), y.argmax(-1).unsqueeze(0), use_label=use_label)
    else:
        if add_steps != 0:
            total_add, add_weight = caculate_total(
                model, x.unsqueeze(0), label.unsqueeze(0), delta_x.unsqueeze(0), y.argmax(-1).unsqueeze(0), steps=add_steps, op="add", alpha=alpha, lambda_r=lambda_r, use_label=use_label)
        if minus_steps != 0:
            total_minus, minus_weight = caculate_total(
                model, x.unsqueeze(0), label.unsqueeze(0), delta_x.unsqueeze(0), y.argmax(-1).unsqueeze(0), steps=minus_steps, op="minus", alpha=alpha, lambda_r=lambda_r, use_label=use_label)
        if add_steps != 0 and minus_steps != 0:
            total = total_add / add_weight + total_minus / minus_weight
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
    elif method == "OBD":
        combine, combine_flatten = caculate_combine(
            total, delta_x, use_total=True, use_delta=True)
    _, flag_norm, flag_delta = binary_search(model, x.unsqueeze(0), label, delta_x, y, len(
        combine_flatten) - 1, combine, combine_flatten, search_times=10)
    return flag_norm, flag_delta
