import torch
import json
import torch.backends.cudnn as cudnn
import random
import numpy as np
import logging
import time
import os
from torch.utils.data import Dataset
from model.st_gcn import Model
from data.transform import *
from configs.sgcn_4_layer import test_args
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def val_logging(file_path):
    logging.basicConfig(filename=file_path, level=logging.DEBUG,
                        format="%(asctime)s %(filename)s %(levelname)s %(message)s",
                        datefmt="%a %d %b %Y %H:%M:%S")
    logging.debug('debug')
    logging.info('info')
    logging.warning('warning')
    logging.error('Error')
    logging.critical('critical')


# 1.load data

class ValDataset(Dataset):
    def __init__(self, annot_file, transform=None, target_transform=None):
        self.annot_file = annot_file
        self.transform = transform
        self.target_transform = target_transform
        with open(self.annot_file, encoding='utf-8') as annot:
            self.result = json.load(annot)

    def __len__(self):
        return len(self.result)

    def __getitem__(self, item):
        person = self.result[item]
        # bbox = np.array(person['bbox'])
        keypoints = np.array(person['keypoints'])
        action = np.array([person['action_category']])
        # translate(keypoints, bbox)

        return keypoints, action


def init_seed(seed=1):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_val(eval_args):

    # 1. load args
    device = eval_args['device']
    annt_path = eval_args['valset_path']
    weights_path = eval_args['weights_path']
    in_channels = eval_args['in_channels']
    num_class = eval_args['num_class']
    edge_importance_weighting = eval_args['edge_importance_weighting']
    graph_args = eval_args['graph_args']
    batch_size = eval_args['batch_size']
    valset = eval_args['valDataLoader']


    # 2.load model
    devices = torch.device("cuda")
    pretrained_dict = torch.load(weights_path, map_location=lambda storage, loc: storage.cuda(devices))
    model = Model(in_channels, num_class, 14, graph_args, edge_importance_weighting)
    model.load_state_dict(pretrained_dict, strict=False)
    model.to(devices)
    model.eval()

    # 3.start eval
    num_samples = len(valset) * batch_size
    num_class_samples_gt = [0, 0, 0, 0]
    num_class_samples_pred = [0, 0, 0, 0]
    # print('num_samples: ', num_samples)
    tp = 0
    for step, (keypoints, label) in enumerate(valset):
        with torch.no_grad():
            keypoints = keypoints.float().cuda(device)
            label = label.cuda(device)
        # forward
            output = model(keypoints)
        pred_label = output.argmax(dim=1)

        for i, action in enumerate(label):
            if  action == pred_label[i]:
                tp += 1
                num_class_samples_pred[int(pred_label[i])] += 1
            num_class_samples_gt[int(label[i])] += 1

    acc = tp/num_samples
    return acc

if __name__ == '__main__':
    # init_seed()
    # args define
    device = test_args['device']
    annt_path = test_args['testset_path']
    log_dir = test_args['log_dir']
    weights_path = test_args['weights_path']
    in_channels = test_args['in_channels']
    num_class = test_args['num_class']
    edge_importance_weighting = test_args['edge_importance_weighting']
    graph_args = test_args['graph_args']
    batch_size = test_args['batch_size']


    # logging
    log_name = 'SGCN_test_' + graph_args['strategy'] + '.log'
    log_path = os.path.join(log_dir, log_name)
    val_logging(log_path)

    # 1.load data
    Valdata = ValDataset(annt_path)
    valset = torch.utils.data.DataLoader(
        dataset=Valdata,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        worker_init_fn=init_seed
    )

    # 2.load model
    devices = torch.device("cuda")
    pretrained_dict = torch.load(weights_path, map_location=lambda storage, loc: storage.cuda(devices))
    model = Model(in_channels, num_class, 14, graph_args, edge_importance_weighting)
    model.load_state_dict(pretrained_dict, strict=False)
    model.to(devices)
    model.eval()

    # 5.start eval

    start = time.time()
    num_samples = len(valset) * batch_size
    num_class_samples_gt = [0, 0, 0, 0]
    num_class_samples_pred = [0, 0, 0, 0]
    print('num_samples: ', num_samples)
    tp = 0
    for step, (keypoints, label) in enumerate(valset):
        with torch.no_grad():
            keypoints = keypoints.float().cuda(device)
            label = label.cuda(device)
        # forward
            output = model(keypoints)
        pred_label = output.argmax(dim=1)

        for i, action in enumerate(label):
            if  action == pred_label[i]:
                tp += 1
                num_class_samples_pred[int(pred_label[i])] += 1
            num_class_samples_gt[int(label[i])] += 1


    # cal acc
    class_acc = np.array(num_class_samples_pred)/np.array(num_class_samples_gt)
    print(class_acc)
    print('num_class_samples_gt: ', num_class_samples_gt)
    print('num_class_samples_pred: ', num_class_samples_pred)
    print('class_acc: ', np.round(class_acc, 2))
    acc = tp/num_samples
    print('acc: ', round(acc, 2))

    logging.info(f"\tNumber of gound truth: {num_class_samples_gt}")
    logging.info(f"\tNumber of predict: {num_class_samples_pred}")
    logging.info(f"\tClass acc: {np.round(class_acc, 2)}")
    logging.info(f"\tMean acc: {acc}")


