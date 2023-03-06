import torch
import torch.nn as nn
from models.signet import SigNet
from torch import optim


def caculate_sigmoid(model, x, delta_x, y, steps=5, alpha=0.0025, lbda=0.01, kind=2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_func = nn.CrossEntropyLoss()
    model.zero_grad()
    sig_net = SigNet(kind)
    sig_net.to(device)
    optimizer = optim.Adam(params=sig_net.parameters(), lr=alpha)
    sigmoid_outputs = list()
    for _ in range(steps):
        if kind == 1:
            sigmoid_output = sig_net(delta_x)
        elif kind == 2:
            sigmoid_output = sig_net(torch.cat([x, delta_x], dim=1))
        sigmoid_outputs.append(sigmoid_output)
        outputs = model(x+delta_x * sigmoid_output)
        if outputs.argmax(-1) != y:
            break
        model.zero_grad()
        sig_net.zero_grad()
        loss = loss_func(outputs, y)
        loss = loss + lbda * torch.norm(sigmoid_output)
        loss.backward()
        optimizer.step()
    sigmoid_output = sigmoid_outputs[-2]
    return sigmoid_output

def exp(model, x, delta_x, y, steps=5, alpha=0.0025, lbda=0.01, kind=2):
    sigmoid_output = caculate_sigmoid(model, x.unsqueeze(0), delta_x.unsqueeze(0), y.argmax(-1).unsqueeze(0), steps, alpha, lbda, kind)
    return torch.norm(delta_x * sigmoid_output).item()