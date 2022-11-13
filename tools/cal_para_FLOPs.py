from thop import clever_format
from thop import profile
import sys,os
sys.path.append(os.getcwd())
from configs.sgcn_4_layer import test_args
from model.st_gcn import Model
import torch

if __name__ == '__main__':
    device = test_args['device']
    annt_path = test_args['testset_path']
    log_dir = test_args['log_dir']
    weights_path = test_args['weights_path']
    in_channels = test_args['in_channels']
    num_class = test_args['num_class']
    edge_importance_weighting = test_args['edge_importance_weighting']
    graph_args = test_args['graph_args']
    batch_size = test_args['batch_size']


    model = Model(in_channels, num_class, 14, graph_args, edge_importance_weighting)

    inputs = torch.randn(1, 14, 3)
    flops, params = profile(model, inputs=(inputs, ))
    print('flops 1:', flops)
    print('params 1:', params)
    flops, params = clever_format([flops, params], "%.3f")
    print('flops 2:', flops)
    print('params 2:', params)





