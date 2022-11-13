import torch
import torch.nn as nn
import torch.nn.functional as F
from model.utils.tgcn import ConvTemporalGraphical
from model.utils.graph import Graph

class Model(nn.Module):
    r"""
    Spatial graph convolutional networks.
    """

    def __init__(self, in_channels, num_class, num_joints, graph_args,
                 edge_importance_weighting, **kwargs):
        super().__init__()

        self.num_joints = num_joints
        # load graph
        self.graph = Graph(**graph_args)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        # build networks
        spatial_kernel_size = A.size(0)

        self.data_bn = nn.BatchNorm1d(in_channels)
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
        self.st_gcn_networks = nn.ModuleList((
            st_gcn(in_channels, 64, spatial_kernel_size, 1, residual=False, **kwargs0),
            st_gcn(128, 64, spatial_kernel_size, 1, **kwargs),
            st_gcn(128, 64, spatial_kernel_size, 1, **kwargs),
            st_gcn(128, 64, spatial_kernel_size, 1, **kwargs),
        ))

        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

        # fcn for prediction
        self.fcn = nn.Conv1d(128, num_class, kernel_size=1)

    def forward(self, x):

        # data normalization
        x = x.permute(0, 2, 1).contiguous()
        x = self.data_bn(x)

        # forwad
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        # global pooling
        x = F.avg_pool1d(x, self.num_joints)
        # a = x.size()

        # prediction
        x = self.fcn(x)
        # a = x.size()
        return x

class st_gcn(nn.Module):
    r"""
    Applies a spatial graph convolution over an input graph sequence.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 residual=True):
        super().__init__()

        # assert kernel_size % 2 == 1

        self.gcn = ConvTemporalGraphical(in_channels, out_channels,
                                         kernel_size)
        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    out_channels*kernel_size,
                    kernel_size=1,
                    stride=1),
                nn.BatchNorm1d(out_channels*kernel_size),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):
        # print('1:',x.shape)
        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = x + res

        return self.relu(x), A