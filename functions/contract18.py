# functions/add.py
import torch
from torch.autograd import Function, Variable
from _ext import gpu18
import pdb
from torch.nn.modules.module import Module

class Contract18Function(Function):
    def __init__(self, num_contractions=18):
        self.num_contractions = num_contractions

    def forward(self, activations, adjacency):
        self.save_for_backward(adjacency)

        nChannels = activations.shape[-1]
        N = activations.shape[0]

        output = torch.zeros(N, N, nChannels * self.num_contractions).float().cuda()
        gpu18.RiskContraction18_forward(activations, adjacency, output, N, nChannels)
        return output

    def backward(self, grad_output):
        #print(type(grad_output))
        #pdb.set_trace()
        adjacency = self.saved_tensors[0]
        #print(adjacency.shape, grad_output.shape)
        nChannels = grad_output.shape[-1] // self.num_contractions
        assert(nChannels * self.num_contractions == grad_output.shape[-1])
        N = adjacency.shape[0]
        #N = grad_output.shape[-2]

        grad_input = torch.zeros(N, N, N, nChannels).float().cuda()
        gpu18.RiskContraction18_backward(grad_input, adjacency, grad_output, N, nChannels)
        return grad_input, None

class Contract18Module(Module):
    def forward(self, activations, adjacency):
        return Contract18Function()(activations, adjacency)

"""
    def forward(self, activations, adjacency):
        self.save_for_backward(adjacency)

        nChannels = activations.shape[0]
        N = activations.shape[1]

        output = torch.zeros(nChannels * self.num_contractions, N, N).float().cuda()
        gpu18.RiskContraction18_forward(activations, adjacency, output, N, nChannels)
        return output

    def backward(self, grad_output):
        print(type(grad_output))
        adjacency = self.saved_tensors

        #nChannels = grad_output.data.shape[0] / self.num_contractions
        #assert(nChannels * self.num_contractions == grad_output.data.shape[0])
        nChannels = grad_output.shape[0] / self.num_contractions
        assert(nChannels * self.num_contractions == grad_output.shape[0])
        
        N = grad_output.data.shape[1]

        grad_input = torch.zeros(nChannels, N, N, N).float().cuda()
        gpu18.RiskContraction18_backward(grad_input, adjacency, grad_output, N, nChannels)
        return grad_input
"""
