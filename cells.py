import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional, init

import numpy as np

from torch.nn.modules import  Linear
from torch.nn.modules.utils import _pair
import torch.nn.functional as F
from expRNN.exprnn import modrelu
from expRNN.initialization import henaff_init, cayley_init, random_orthogonal_init
from expRNN.exp_numpy import expm, expm_frechet
import snorm


class RNNCell1(nn.Module):
    def __init__(self, inp_size, hid_size, nonlin, bias=True, cuda=False, r_initializer=henaff_init,
                 i_initializer=nn.init.xavier_normal_):
        super(RNNCell1, self).__init__()
        self.cudafy = cuda
        self.hidden_size = hid_size
        # Add Non linearity
        if nonlin == 'relu':
            self.nonlinearity = nn.ReLU()
        if nonlin == 'modrelu':
            self.nonlinearity = modrelu(hid_size)
        elif nonlin == 'tanh':
            self.nonlinearity = nn.Tanh()
        elif nonlin == 'sigmoid':
            self.nonlinearity = nn.Sigmoid()
        else:
            self.nonlinearity = None

        # Create linear layer to act on input X
        self.U = nn.Linear(inp_size, hid_size, bias=bias)
        self.i_initializer = i_initializer

        self.V = nn.Linear(hid_size, hid_size, bias=False)

        self.r_initializer = r_initializer
        self.reset_parameters()

    def reset_parameters(self):
        self.i_initializer(self.U.weight.data)
        if self.r_initializer == random_orthogonal_init or \
                self.r_initializer == henaff_init or \
                self.r_initializer == cayley_init:
            self.V.weight.data = self._B(
                torch.as_tensor(self.r_initializer(self.hidden_size)))
        else:
            print('other')
            self.r_initializer(self.V.weight.data)

    def _A(self, gradients=False):
        A = self.V.weight.data
        if not gradients:
            A = A.data
        A = A.triu(diagonal=1)
        return A - A.t()

    def _B(self, gradients=False):
        return expm(self._A())
    
    def _norm(self):
        norm = snorm.spectral_norm(self.V.weight.data)
        return norm
    def get_alpha(self):
        return self.alpha.clone().detach().cpu().numpy()

    def forward(self, x, hidden=None):
        if hidden is None:
            hidden = x.new_zeros(x.shape[0], self.hidden_size, requires_grad=True, device= x.device)

            self.first_hidden = hidden

        h = self.U(x) + self.V(hidden)
        if self.nonlinearity:
            h =  self.nonlinearity(h)
        return h

class RNNCell2(nn.Module):
    def __init__(self, inp_size, hid_size, slen, nonlin, bias=True, cuda=False, r_initializer=henaff_init,
                 i_initializer=nn.init.xavier_normal_):
        super(RNNCell2, self).__init__()
        self.cudafy = cuda
        self.hidden_size = hid_size
        # Add Non linearity
        if nonlin == 'relu':
            self.nonlinearity = nn.ReLU()
        if nonlin == 'modrelu':
            self.nonlinearity = modrelu(hid_size)
        elif nonlin == 'tanh':
            self.nonlinearity = nn.Tanh()
        elif nonlin == 'sigmoid':
            self.nonlinearity = nn.Sigmoid()
        else:
            self.nonlinearity = None

        # Create linear layer to act on input X
        self.U = nn.Linear(inp_size, hid_size, bias=bias)
        self.i_initializer = i_initializer

        self.V = nn.Linear(hid_size, hid_size, bias=False)
        self.alpha = nn.Parameter(torch.rand(slen))
        self.r_initializer = r_initializer
        self.reset_parameters()


    def reset_parameters(self):
        self.i_initializer(self.U.weight.data)
        if self.r_initializer == random_orthogonal_init or \
                self.r_initializer == henaff_init or \
                self.r_initializer == cayley_init:
            self.V.weight.data = self._B(
                torch.as_tensor(self.r_initializer(self.hidden_size)))
        else:
            print('other')
            self.r_initializer(self.V.weight.data)

    def _A(self, gradients=False):
        A = self.V.weight.data
        if not gradients:
            A = A.data
        A = A.triu(diagonal=1)
        return A - A.t()

    def _B(self, gradients=False):
        return expm(self._A()) 
    def get_alpha(self):
        return self.alpha.clone().detach().cpu().numpy()
    def _norm(self):
        norm = snorm.spectral_norm(self.V.weight.data)

        return norm


    def forward(self, x, hidden=None, i=1):
        if hidden is None:
            hidden = x.new_zeros(x.shape[0], self.hidden_size, requires_grad=True, device= x.device)

        h = self.U(x) + self.V(hidden)
        if self.nonlinearity:
            h = (1 - torch.exp(-self.alpha[i] * h)) * self.nonlinearity(h)
        return h

 
class RNNCellLT(nn.Module):
    def __init__(self, inp_size, hid_size, nonlin, bias=True, cuda=False, r_initializer=henaff_init,
                 i_initializer=nn.init.xavier_normal_):
        super(RNNCellLT, self).__init__()
        self.cudafy = cuda
        self.hidden_size = hid_size
        # Add Non linearity
        if nonlin == 'relu':
            self.nonlinearity = nn.ReLU()
        if nonlin == 'modrelu':
            self.nonlinearity = modrelu(hid_size)
        elif nonlin == 'tanh':
            self.nonlinearity = nn.Tanh()
        elif nonlin == 'sigmoid':
            self.nonlinearity = nn.Sigmoid()
        else:
            self.nonlinearity = None

        # Create linear layer to act on input X
        self.U = nn.Linear(inp_size, hid_size, bias=bias)
        self.i_initializer = i_initializer

        self.V = torch.nn.utils.spectral_norm(nn.Linear(hid_size, hid_size, bias=False))

        self.r_initializer = r_initializer
        #self.reset_parameters()


    def reset_parameters(self):
        self.i_initializer(self.U.weight.data)
        if self.r_initializer == random_orthogonal_init or \
                self.r_initializer == henaff_init or \
                self.r_initializer == cayley_init:
            self.V.weight.data = self._B(
                torch.as_tensor(self.r_initializer(self.hidden_size)))
        else:
            print('other')
            self.r_initializer(self.V.weight.data)

    def _A(self, gradients=False):
        A = self.V.weight.data
        if not gradients:
            A = A.data
        A = A.triu(diagonal=1)
        return A - A.t()

    def _B(self, gradients=False):
        return expm(self._A())


    def forward(self, x, hidden=None):
        if hidden is None:
            hidden = x.new_zeros(x.shape[0], self.hidden_size, requires_grad=True)
            self.first_hidden = hidden

        h = self.nonlinearity(self.U(x)) + self.V(hidden)
        return h



class EURNNCell(nn.Module):
    """An EURNN cell."""

    def __init__(self, input_size, hidden_size, capacity):

        super(EURNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.capacity = capacity
        self.U = nn.Parameter(
            torch.FloatTensor(input_size, hidden_size))
        self.thetaA = nn.Parameter(
            torch.FloatTensor(hidden_size/2, capacity/2))
        self.thetaB = nn.Parameter(
            torch.FloatTensor(hidden_size/2-1, capacity/2))
        self.bias = nn.Parameter(
            torch.FloatTensor(hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize parameters
        """
        init.uniform(self.thetaA, a=-0.1, b=0.1)
        init.uniform(self.thetaB, a=-0.1, b=0.1)
        init.uniform(self.U, a=-0.1, b=0.1)
        init.constant(self.bias.data, val=0)

    def _EUNN(self, hx, thetaA, thetaB):

        L = self.capacity
        N = self.hidden_size

        sinA = torch.sin(self.thetaA)
        cosA = torch.cos(self.thetaA)
        sinB = torch.sin(self.thetaB)
        cosB = torch.cos(self.thetaB)

        I = Variable(torch.ones((L/2, 1)))
        O = Variable(torch.zeros((L/2, 1)))

        diagA = torch.stack((cosA, cosA), 2)
        offA = torch.stack((-sinA, sinA), 2)
        diagB = torch.stack((cosB, cosB), 2)
        offB = torch.stack((-sinB, sinB), 2)

        diagA = diagA.view(L/2, N)
        offA = offA.view(L/2, N)
        diagB = diagB.view(L/2, N-2)
        offB = offB.view(L/2, N-2)

        diagB = torch.cat((I, diagB, I), 1)
        offB = torch.cat((O, offB, O), 1)

        batch_size = hx.size()[0]
        x = hx
        for i in range(L/2):
            # A
            y = x.view(batch_size, N/2, 2)
            y = torch.stack((y[:,:,1], y[:,:,0]), 2)
            y = y.view(batch_size, N)

            x = torch.mul(x, diagA[i].expand_as(x))
            y = torch.mul(y, offA[i].expand_as(x))

            x = x + y

            # B
            x_top = x[:,0]
            x_mid = x[:,1:-1].contiguous()
            x_bot = x[:,-1]
            y = x_mid.view(batch_size, N/2-1, 2)
            y = torch.stack((y[:, :, 1], y[:, :, 0]), 1)
            y = y.view(batch_size, N-2)
            x_top = torch.unsqueeze(x_top, 1)
            x_bot = torch.unsqueeze(x_bot, 1)
            # print x_top.size(), y.size(), x_bot.size()
            y = torch.cat((x_top, y, x_bot), 1)

            x = x * diagB[i].expand(batch_size, N)
            y = y * offB[i].expand(batch_size, N)

            x = x + y
        return x

    def _modReLU(self, h, bias):
        """
        sign(z)*relu(z)
        """
        batch_size = h.size(0)
        sign = torch.sign(h)
        bias_batch = (bias.unsqueeze(0)
                      .expand(batch_size, *bias.size()))
        return sign * functional.relu(torch.abs(h) + bias_batch)

    def forward(self, input_, hx, i):
        """
        Args:
            input_: A (batch, input_size) tensor containing input
                features.
            hx: initial hidden, where the size of the state is
                (batch, hidden_size).
        Returns:
            newh: Tensors containing the next hidden state.
        """
        batch_size = hx.size(0)
        Ux = torch.mm(input_, self.U)
        hx = Ux + hx
        newh = self._EUNN(hx=hx, thetaA=self.thetaA, thetaB=self.thetaB)
        newh = self._modReLU(newh, self.bias)
        return newh

    def __repr__(self):
        s = '{name}({input_size}, {hidden_size})'
        return s.format(name=self.__class__.__name__, **self.__dict__)
