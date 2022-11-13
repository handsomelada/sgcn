# The based unit of graph convolutional networks.

import torch
import torch.nn as nn

class ConvTemporalGraphical(nn.Module):
    r"""
    The basic module for applying a graph convolution.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 bias=True):
        super().__init__()

        self.kernel_size = kernel_size

        self.conv = nn.Conv1d(
            in_channels,
            out_channels * kernel_size,
            kernel_size=1,
            stride=1,
            dilation=1,
            bias=bias)

    def forward(self, x, A):
        x = self.conv(x)
        # x = torch.einsum('ncv,kvw->ncw', (x, A))
        A = torch.sum(A, dim=0)
        x = torch.matmul(x, A)
        return x.contiguous(), A
