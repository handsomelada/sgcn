import json
import os
import copy
import numpy as np
import random
from data.transform import *

def merge(json_dir, out_path):
    """
        json_dir: must be a directory
        out_path: example: xxxx.json
    """
    json_list = os.listdir(json_dir)
    merge_json = []
    for i, js in enumerate(json_list):
        json_path = os.path.join(json_dir, js)
        with open(json_path, 'r', encoding='utf-8') as annot:
            result = json.load(annot)
        for j in result:
            merge_json.append(j)
    with open(out_path, 'w', encoding='utf-8') as output:
        json.dump(merge_json, output, indent=4, separators=(',', ': '))
    print(merge_json)

def merge_2(dir_path, out_path):
    """
        json_dir: must be a directory
        out_path: example: xxxx.json
    """
    dir_list = os.listdir(dir_path)
    merge_json = []
    for k in dir_list:
        json_path = os.path.join(dir_path, k)
        json_list = os.listdir(json_path)
        for i, js in enumerate(json_list):
            jp = os.path.join(json_path, js)
            with open(jp, 'r', encoding='utf-8') as annot:
                result = json.load(annot)
            for j in result:
                merge_json.append(j)
    print('len of merge json', len(merge_json))
    with open(out_path, 'w', encoding='utf-8') as output:
        json.dump(merge_json, output, indent=4, separators=(',', ': '))


def add_conf(json_path, out_path = None):
    """
        function: Add visible flag
        json_path: example: xxxx.json
        out_path: example: xxxx.json
    """
    with open(json_path, 'r') as annot:
        result = json.load(annot)
    for j in result:
        keypoints = j.get('keypoints')
        for i, k in enumerate(keypoints):
            keypoints[i] = float(k)
        keypoints_list = []
        k = 0
        for i in range(int(len(keypoints) / 2)):
            keypoints_list.append(keypoints[k:k+2])
            if float(keypoints_list[i][0]) == 0 and float(keypoints_list[i][1]) == 0:
                keypoints_list[i].append(0)
            else:
                keypoints_list[i].append(1)
            k += 2
        j['keypoints'] = keypoints_list

    with open(out_path, 'w', encoding='utf-8') as output:
        json.dump(result, output, indent=4, separators=(',', ': '))

def add_conf2(json_path, out_path = None):
    """
        function: Add visible flag
        json_path: example: xxxx.json
        out_path: example: xxxx.json
    """
    with open(json_path, 'r') as annot:
        result = json.load(annot)
    for j in result:
        keypoints = j.get('keypoints')
        for i, k in enumerate(keypoints):
            keypoints[i] = float(k)
        keypoints_list = []
        k = 0
        for i in range(17):
            keypoints_list.append(keypoints[k:k+2])
            if float(keypoints_list[i][0]) == 0 and float(keypoints_list[i][1]) == 0:
                keypoints_list[i].append(0)
            else:
                keypoints_list[i].append(1)
            k += 3
        j['keypoints'] = keypoints_list

    with open(out_path, 'w', encoding='utf-8') as output:
        json.dump(result, output, indent=4, separators=(',', ': '))

def gen_dataset(full_label_path, ratio, trainset_path, valset_path):
    with open(full_label_path, 'r') as annot:
        result = json.load(annot)
    random.shuffle(result)
    num_samples = len(result)
    num_train = int(ratio * num_samples)
    train_set = result[:num_train]
    val_set = result[num_train:]
    print('len trainset-----', len(train_set))
    print('len val_set-----', len(val_set))
    with open(trainset_path, 'w', encoding='utf-8') as output:
        json.dump(train_set, output, indent=4, separators=(',', ': '))
    with open(valset_path, 'w', encoding='utf-8') as output:
        json.dump(val_set, output, indent=4, separators=(',', ': '))

def augmentation(full_label_path):
    with open(full_label_path, 'r') as annot:
        result = json.load(annot)

def delete_head(json_path, out_path):
    with open(json_path, 'r') as annot:
        result = json.load(annot)
    new_data = copy.deepcopy(result)
    for j in new_data:
        del j['keypoints'][0:5]
    with open(out_path, 'w', encoding='utf-8') as output:
        json.dump(new_data, output, indent=4, separators=(',', ': '))

def delete_eye_nose(json_path, out_path):
    with open(json_path, 'r') as annot:
        result = json.load(annot)
    new_data = copy.deepcopy(result)
    for j in new_data:
        del j['keypoints'][0:3]
    with open(out_path, 'w', encoding='utf-8') as output:
        json.dump(new_data, output, indent=4, separators=(',', ': '))

def kuozeng(json_path, out_path=None):
    with open(json_path, 'r') as annot:
        result = json.load(annot)
    kuozen_list = []
    new_data = copy.deepcopy(result)
    for j in new_data:
        if j['action_category'] != 3:
            kuozen_list.append(j)
            kpts = j['keypoints']
            bbox = j['bbox']
            copy_j = copy.deepcopy(j)
            copy_jj = copy.deepcopy(j)

            # print('before',copy_j)
            kk = np.array(copy_j['keypoints'])
            copy_kpts = poseInbbox_flip(kk, bbox, 'horizontal')
            copy_j['keypoints'] = copy_kpts.tolist()
            # print('after', copy_j)
            kuozen_list.append(copy_j)

            print('before', copy_jj)
            kkk = np.array(copy_jj['keypoints'])
            copy_kptss = poseInimg_flip(kkk, 'horizontal', img_shape=[720, 1280])
            copy_jj['keypoints'] = copy_kptss.tolist()
            print('after', copy_jj)
            kuozen_list.append(copy_jj)

        if  j['action_category'] == 3:
            kuozen_list.append(j)

    print('kuozen_list len: ',len(kuozen_list))
    with open(out_path, 'w', encoding='utf-8') as output:
        json.dump(kuozen_list, output, indent=4, separators=(',', ': '))



if __name__ == '__main__':
    json_dir = '/root/autodl-tmp/wmh/dataset/escalator/action_label'
    out_path = '/root/autodl-tmp/wmh/dataset/escalator/action_classification_merge.json'
    new_out_path = '/root/autodl-tmp/wmh/dataset/escalator/new_action_classification_merge.json'
    trainset_path = '/root/autodl-tmp/wmh/dataset/escalator/action_train.json'
    valset_path = '/root/autodl-tmp/wmh/dataset/escalator/action_val.json'

    # delete_eye_nose(new_out_path, '/root/autodl-tmp/wmh/dataset/escalator/no_eye_ear_pose.json')
    # kuozeng('/root/autodl-tmp/wmh/dataset/escalator/no_eye_ear_pose.json',
            # '/root/autodl-tmp/wmh/dataset/escalator/no_eye_ear_kuozeng-17884.json')

    gen_dataset('/root/autodl-tmp/wmh/dataset/escalator/no_eye_kuozeng-17884.json',
                0.8,
                '/root/autodl-tmp/wmh/dataset/escalator/no_eye_kuozeng-17884_train.json',
                '/root/autodl-tmp/wmh/dataset/escalator/no_eye_kuozeng-17884_val.json')

    # merge(json_dir, out_path)
    # add_conf(out_path, new_out_path)
    # gen_dataset(new_out_path, 0.8, trainset_path, valset_path)
    # merge_2(json_dir, out_path)
    # delete_head(new_out_path, '/root/autodl-tmp/wmh/dataset/escalator/train_without_head_pose.json')

    # calculate number of each class
    # num_normal = 0
    # with open('/root/autodl-tmp/wmh/dataset/escalator/train_without_head_pose.json', 'r') as annot:
    #     result = json.load(annot)
    # new_data = copy.deepcopy(result)
    # random.shuffle(new_data)
    # action_list = []
    # new_action_count = []
    # print('len of dataset: ', len(new_data))
    #
    # formal_list = []
    # flag = True
    # for i, elem in enumerate(new_data):
    #     if flag == True:
    #         formal_list.append(elem)
    #     elif elem['action_category'] != 3:
    #         formal_list.append(elem)
    #     if elem['action_category'] == 3:
    #         num_normal += 1
    #     if num_normal == 2000:
    #         flag = False
    #
    # for j in formal_list:
    #     action_list.append(j['action_category'])
    # # print(action_list)
    # new_action_count.append(action_list.count(0))
    # new_action_count.append(action_list.count(1))
    # new_action_count.append(action_list.count(2))
    # new_action_count.append(action_list.count(3))
    # print(new_action_count)
    #
    # with open('/root/autodl-tmp/wmh/dataset/escalator/train_with_2000normal.json', 'w', encoding='utf-8') as output:
    #     json.dump(formal_list, output, indent=4, separators=(',', ': '))




