import torch
import json
import torch.nn as nn
import random
import numpy as np
import logging
import time
import sys, os
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch import optim
from torch.utils.data import Dataset
from model.st_gcn import Model
from configs.sgcn_4_layer import train_args
from data.transform import *
from action_test import ValDataset, train_val
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def train_logging(file_path):
    logging.basicConfig(filename=file_path, level=logging.DEBUG,
                        format="%(asctime)s %(filename)s %(levelname)s %(message)s",
                        datefmt="%a %d %b %Y %H:%M:%S")
    logging.debug('debug')
    logging.info('info')
    logging.warning('warning')
    logging.error('Error')
    logging.critical('critical')

# 1.load data
class TrainDataset(Dataset):
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
        action = np.array([person['action_category']])
        keypoints = np.array(person['keypoints'])
        # print(keypoints)
        # a = random.random()
        # if a >= 0.5 and action != 3:
        #     poseInbbox_flip(keypoints, bbox, 'horizontal')
        # keypoints = rand_scale(keypoints, bbox, 0.2)
        return keypoints, action


def init_seed(seed=1999):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    # args define
    init_seed()

    in_channels = train_args['in_channels']
    num_class = train_args['num_class']
    num_joints = train_args['num_joints']
    edge_importance_weighting = train_args['edge_importance_weighting']
    graph_args = train_args['graph_args']
    save_best = train_args['save_best']

    device = train_args['device']
    trainset_path = train_args['trainset_path']
    log_dir = train_args['log_dir']
    valset_path = train_args['valset_path']
    weights_dir = train_args['weights_dir']
    total_epoch = train_args['total_epoch']
    batch_size = train_args['batch_size']
    lr = train_args['lr']
    steps = train_args['steps']

    # logging and visualize curves
    log_name = 'SGCN_train_' + graph_args['strategy'] + '.log'
    log_path = os.path.join(log_dir, log_name)
    train_logging(log_path)

    writer = SummaryWriter("training curves")

    # 1.load training and validation data
    traindata = TrainDataset(trainset_path)
    # mydata = MyDataset(annt_path, transforms.Compose([transforms.pose_flip('horizontal'),
    #                                                   transforms.ToTensor()
    #                                                   ]))
    print('Size of trainset: ', len(traindata))
    trainset = torch.utils.data.DataLoader(
        dataset=traindata,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        worker_init_fn=init_seed
    )
    if valset_path:
        valdata = ValDataset(valset_path)
        print('Size of validatioon set: ', len(valdata))
        valset = torch.utils.data.DataLoader(
            dataset=valdata,
            batch_size=batch_size,
            shuffle=False,
            num_workers=8,
            worker_init_fn=init_seed
        )
        train_args['valDataLoader'] = valset

    # 2.load model
    model = Model(in_channels, num_class, num_joints, graph_args, edge_importance_weighting)
    model.to(device)

    # 3.define loss
    CELloss = nn.CrossEntropyLoss().to(device)

    # 4.define optimizer
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=0.0004)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, steps, gamma=0.1, last_epoch=-1)

    # 5.start train
    # tqdm setting
    pbar = tqdm(range(total_epoch))
    # pbar = tqdm(range(total_epoch), ncols=100)

    model.train()
    train_record = dict()
    least_loss = 1000.
    least_acc = 0.
    train_info={'loss':100, 'lr':0.1}

    for epo, element in enumerate(pbar):
        total_loss = []
        start_time = time.time()
        for step, (keypoints, label) in enumerate(trainset):
            with torch.no_grad():
                keypoints = keypoints.float().to(device)
                label = label.to(device)
            # forward
            output = model(keypoints)
            loss = CELloss(output, label)
            
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss.append(loss.data.item())
        scheduler.step()
        mean_epoch_loss = np.mean(total_loss)

        end_time = time.time()
        iter_time = round(end_time - start_time, 2)
        # save current weights
        weights_name = 'SGCN_' + graph_args['strategy'] + '_' + str(epo) + 'epoch' + '.pt'
        current_weights_path = os.path.join(weights_dir, weights_name)
        train_args['weights_path'] = current_weights_path
        state_dict = model.state_dict()
        torch.save(state_dict, current_weights_path)
    
        # 6. start validation
        if valset_path:
            acc = train_val(train_args)
            pbar.set_description(f"Epoch {epo}/{total_epoch}")
            pbar.set_postfix(
                cost_time = str(iter_time) + 's',
                lr = scheduler.get_last_lr(),
                loss = mean_epoch_loss, 
                acc = round(acc, 3)
                )
            logging.info(
            f"\tEpoch: {epo + 1}/{total_epoch}\tcost: {end_time - start_time:.4f}\tloss: {mean_epoch_loss:.4f}\tacc: {acc:.4f}") 
            writer.add_scalar("loss", mean_epoch_loss, epo)
            writer.add_scalar("acc",acc,epo)
            if acc > least_acc:
                least_acc = acc
                weights_name = 'SGCN_' + graph_args['strategy'] + '_best_' + 'epoch' + '.pt'
                best_weights_path = os.path.join(weights_dir, weights_name)
                # save best weights
                torch.save(state_dict, best_weights_path)
                logging.info(f'save epoch:{epo+1}')
        else:
            pbar.set_description(f"Epoch {epo}/{total_epoch}")
            pbar.set_postfix(
                cost_time = str(iter_time) + 's',
                lr = scheduler.get_last_lr(),
                loss = mean_epoch_loss, 
                ) 
            logging.info(
            f"\tEpoch: {epo + 1}/{total_epoch}\tcost: {end_time - start_time:.4f}\tloss: {mean_epoch_loss:.4f}") 
            writer.add_scalar("loss", mean_epoch_loss, epo)
            if mean_epoch_loss < least_loss:
                least_loss = mean_epoch_loss
                weights_name = 'SGCN_' + graph_args['strategy'] + '_best_' + 'epoch' + '.pt'
                best_weights_path = os.path.join(weights_dir, weights_name)
                # save best weights
                torch.save(state_dict, best_weights_path)
                logging.info(f'save epoch:{epo+1}')
    
        # Save best checkpoints and delete others.
        if save_best == True:
            os.remove(current_weights_path)
    
    writer.close()


    

            








