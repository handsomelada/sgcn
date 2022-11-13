import torch
import torch.nn
import onnx
from model.st_gcn import Model
from configs.sgcn_4_layer import args

if __name__ == '__main__':
    weights = '../weights/sgcn_test_kuozeng_withear_dis_4layer_spatial.pt'
    num_joints = args['num_joints'] # 关键点个数
    device = torch.device("cuda")
    pretrained_dict = torch.load(weights, map_location=lambda storage, loc: storage.cuda(device))
    model = Model(args['in_channels'], args['num_class'], args['num_joints'], args['graph_args'], args['edge_importance_weighting'])
    model.load_state_dict(pretrained_dict, strict=False)
    # model = model.half()
    model.to(device)
    model.eval()

    x = torch.randn(1, num_joints, 3)
    # x = x.half()
    x = x.to(device)

    with torch.no_grad():
        torch.onnx.export(
            model,
            x,
            "sgcn_test_kuozeng_withear_dis_4layer_spatial.onnx",
            opset_version=13,
            input_names=['input'],
            output_names=['output']
        )
    print('Export onnx file finished.')

