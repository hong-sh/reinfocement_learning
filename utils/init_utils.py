import numpy as np
import torch
 

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

def weight_init(m):
    classname = m.__class__.__name__

    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        orthogonal(m.weight.data)
    
        if m.bias is not None:
            m.bias.data.fill_(0)

def orthogonal(tensor, gain = 1):
    if tensor.ndimension() < 2:
        raise ValueError("Only tensors with 2 or more dimensions are suppored")
    
    rows = tensor.size(0)
    cols = tensor[0].numel()

    flattened = torch.Tensor(rows, cols).normal_(0, 1)

    if rows < cols:
        flattened.t_()


    q, r = torch.qr(flattened)
    d = torch.diag(r, 0)
    ph = d.sign()

    q *= ph.expand_as(q)

    if rows < cols:
        q.t_()

    tensor.view_as(q).copy_(q)
    tensor.mul_(gain)
    return tensor