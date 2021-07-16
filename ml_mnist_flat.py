from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
import torch
import copy

class MNISTBlock(nn.Module):
    def __init__(self, width, scaling=1.0,  use_bias = True):
        super(MNISTBlock, self).__init__()

        self.scaling = scaling
        self.linear = nn.Linear(width, width, bias = use_bias)
        nn.init.xavier_normal_(self.linear.weight)
        self.shortcut = nn.Sequential()

    def forward(self, x):
        out = self.linear(x)
        out = float(self.scaling) * F.relu(out)
        out += self.shortcut(x)

        return out

### mnist net  ###
class MLFlatNetMNIST(nn.Module):
    linear: Linear
    def __init__(self, BasicBlock, num_layers=512, scaling=1, nclasses=10, width=10, use_bias=True):
        super(MLFlatNetMNIST, self).__init__()

        self.num_blocks = num_layers
        layers = []
        self.linear_in = nn.Linear(784, 10, use_bias)
        for i in range(self.num_blocks): # num_layers is n^m, the +1 is for nice matching
            layers.append(MNISTBlock(width=10, scaling=scaling, use_bias=use_bias))

        self.blocks = nn.Sequential(*layers)
        self.linear_out = nn.Linear(10, nclasses, use_bias)

    def forward(self, x):
        out = self.linear_in(x)
        out = self.blocks(out)
        out = F.relu(self.linear_out(out))
        return out

    def restrict_weights(self, net_H: MLFlatNetMNIST, scaling=1.0):
        with torch.no_grad():

            net_H.linear_in.weight.data.copy_(torch.nn.parameter.Parameter(self.linear_in.weight.data))
            net_H.linear_in.bias.data.copy_(torch.nn.parameter.Parameter(self.linear_in.bias.data))

            for id_H, block_H in enumerate(net_H.blocks):
                block_h = self.blocks[2 * id_H]
                block_H.linear.weight.data.copy_(scaling * torch.nn.parameter.Parameter(block_h.linear.weight.data))
                block_H.linear.bias.data.copy_(scaling * torch.nn.parameter.Parameter(block_h.linear.bias.data))

            net_H.linear_out.weight.data.copy_(torch.nn.parameter.Parameter(self.linear_out.weight.data))
            net_H.linear_out.bias.data.copy_(torch.nn.parameter.Parameter(self.linear_out.bias.data))


    def restrict_gradient(self, net_H: MLFlatNetMNIST, scaling=1.0):
        with torch.no_grad():
            # I_h^H * g_h
            I_g_h = []

            I_g_h.append(self.linear_in.weight.grad.clone().detach())
            I_g_h.append(self.linear_in.bias.grad.clone().detach())

            for id_H, block_H in enumerate(net_H.blocks):
                block_h1 = self.blocks[2 * id_H]
                I_g_h_linear = nn.Parameter(scaling*block_h1.linear.weight.grad).clone().detach()
                I_g_h.append(I_g_h_linear)

                I_g_h_linear_bias = nn.Parameter(scaling*block_h1.linear.bias.grad).clone().detach()
                I_g_h.append(I_g_h_linear_bias)

            I_g_h.append(self.linear_out.weight.grad)
            I_g_h.append(self.linear_out.bias.grad)

        return I_g_h

    def form_v(self, net_H: MLFlatNetMNIST):
        # v = g_H(x_h)- I[g_h(x_h)], I_g_h = I[g_h(x_h)]
        # add 1st and last layer ?
        I_g_h = self.restrict_gradient(net_H)
        last_idx = len(I_g_h) -1
        v = []

        v.append(torch.add(net_H.linear_in.weight.grad, -1.0 * I_g_h[0]))
        v.append(torch.add(net_H.linear_in.bias.grad, -1.0 * I_g_h[1]))

        for id_H, block_H in enumerate(net_H.blocks):
            v.append(torch.add(block_H.linear.weight.grad, -1 * I_g_h[(2 + (2 * id_H))]))
            v.append(torch.add(block_H.linear.bias.grad, -1 * I_g_h[(2 + (2 * id_H) + 1)]))

        v.append(torch.add(net_H.linear_out.weight.grad, -1.0 * I_g_h[last_idx-1]))
        v.append(torch.add(net_H.linear_out.bias.grad, -1.0 * I_g_h[last_idx]))

        return v

    def prolong_step_to_gradient(self, prev_w_H, net_H: MLFlatNetMNIST):
        # I_H^h (x_H^2 - x_H^1)
        # note: we assemble e in the gradient of  net_h
        sign = -1.0
        self.zero_grad()  # set to zero, then  we need to add only and we don't need grad(f(\theta^2_h)) anymore
        last_idx = len(prev_w_H) -1

        self.linear_in.weight.grad.copy_(torch.add(sign * net_H.linear_in.weight.data, sign * -1 * prev_w_H[0].data))
        self.linear_in.bias.grad.copy_(torch.add(sign * net_H.linear_in.bias.data, sign * -1 * prev_w_H[1].data))

        for id_H,  block_H in enumerate(net_H.blocks, 0):
            idx = 2*(id_H+1)

            block_h1 = self.blocks[2*id_H]
            block_h1.linear.weight.grad.add_(torch.add(sign * block_H.linear.weight.data, sign * -1.0 * prev_w_H[idx].data))
            block_h1.linear.bias.grad.add_(torch.add(sign * block_H.linear.bias.data, sign * -1.0 * prev_w_H[idx + 1].data))

            if id_H < net_H.num_blocks - 1:
                block_h2 = self.blocks[id_H * 2 + 1]
                block_H2 = net_H.blocks[id_H + 1]

                block_h2.linear.weight.grad.add_(0.5 * torch.add(sign * block_H.linear.weight.data, sign * -1 * prev_w_H[idx].data))
                block_h2.linear.weight.grad.add_(0.5 * torch.add(sign * block_H2.linear.weight.data, sign * -1 * prev_w_H[idx + 2].data))

                block_h2.linear.bias.grad.add_(0.5 * torch.add(sign * block_H.linear.bias.data, sign * -1 * prev_w_H[idx + 1].data))
                block_h2.linear.bias.grad.add_(0.5 * torch.add(sign * block_H2.linear.bias.data, sign * -1 * prev_w_H[idx + 3].data))

        self.linear_out.weight.grad.copy_(torch.add(sign * net_H.linear_out.weight.data, sign * -1 * prev_w_H[last_idx-1].data))
        self.linear_out.bias.grad.copy_(torch.add(sign * net_H.linear_out.bias.data, sign * -1 * prev_w_H[last_idx].data))

    def subtract_v_from_grad_f_H(self, v):
        sign  = -1.0
        last_idx = len(v) -1
        self.linear_in.weight.grad.add_(sign * v[0])
        self.linear_in.bias.grad.add_(sign * v[1])
        # correct indices !
        for id_H, block_H in enumerate(self.blocks):
            block_H.linear.weight.grad.add_(sign * v[2 * (id_H + 1)])
            block_H.linear.bias.grad.add_(sign * v[2 * (id_H + 1) + 1])

        self.linear_out.weight.grad.add_(sign * v[last_idx-1])
        self.linear_out.bias.grad.add_(sign * v[last_idx])


    def save_weights_to_tensor(self):
        t = []
        t.append(copy.deepcopy(self.linear_in.weight.data))
        t.append(copy.deepcopy(self.linear_in.bias.data))

        for block in self.blocks:
            t.append(copy.deepcopy(block.linear.weight.data))
            t.append(copy.deepcopy(block.linear.bias.data))

        t.append(copy.deepcopy(self.linear_out.weight.data))
        t.append(copy.deepcopy(self.linear_out.bias.data))

        return t

    def _gather_flat_grad(self):
        views = []
        for p in self.parameters():
            if p.grad is None:
                view = p.data.new(p.data.numel()).zero_()
            elif p.grad.data.is_sparse:
                view = p.grad.data.to_dense().view(-1)
            else:
                view = p.grad.data.view(-1)
            views.append(view)
        return torch.cat(views, 0)

    def compute_m(self, grad_x, p):
        m = 0
        last_idx = len(grad_x) -1
        for i in range(last_idx):

            for param in self.parameters():
                m += torch.sum(param.data * param.grad)
        return m

    def save_grad_to_tensor(self):
        t = []

        t.append(copy.deepcopy(self.linear_in.weight.grad))
        t.append(copy.deepcopy(self.linear_in.bias.grad))

        for block in self.blocks:
            t.append(copy.deepcopy(block.linear.weight.grad))
            t.append(copy.deepcopy(block.linear.bias.grad))

        t.append(copy.deepcopy(self.linear_out.weight.grad))
        t.append(copy.deepcopy(self.linear_out.bias.grad))

        return t

    def copy_tensor_to_weights(self, t):
        last_idx = len(t) -1

        self.linear_in.weight.data = copy.deepcopy(t[0])
        self.linear_in.bias.data = copy.deepcopy(t[1])

        for id, block in enumerate(self.blocks):
            block.linear.weight.data = copy.deepcopy(t[2 * (id + 1)])
            block.linear.bias.data = copy.deepcopy(t[2 * (id + 1) + 1])

        self.linear_out.weight.data = copy.deepcopy(t[last_idx-1])
        self.linear_out.bias.data = copy.deepcopy(t[last_idx])