import json
import argparse
import copy
import sys,os
import numpy as np
import random
sys.path.append(os.getcwd())

from data.transform import *

def data_augment(trainset_path, out_path):
	with open(trainset_path, 'r') as annot:
		result = json.load(annot)
	print('Length of origin trainset : ',len(result))
	kuozen_list = []
	new_data = copy.deepcopy(result)
	for j in new_data:
		if j['action_category'] != 3:
			kuozen_list.append(j)
			kpts = j['keypoints']
			bbox = j['bbox']
			copy_j = copy.deepcopy(j)
			copy_jj = copy.deepcopy(j)

			kk = np.array(copy_j['keypoints'])
			copy_kpts = poseInbbox_flip(kk, bbox, 'horizontal')
			copy_j['keypoints'] = copy_kpts.tolist()
			kuozen_list.append(copy_j)

			kkk = np.array(copy_jj['keypoints'])
			copy_kptss = poseInimg_flip(kkk, 'horizontal', img_shape=[720, 1280])
			copy_jj['keypoints'] = copy_kptss.tolist()
			kuozen_list.append(copy_jj)

		if  j['action_category'] == 3:
			kuozen_list.append(j)
	print('Length of augmented trainset : ',len(kuozen_list))
	with open(out_path, 'w', encoding='utf-8') as output:
		json.dump(kuozen_list, output, indent=4, separators=(',', ': '))

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	# data augmentation args
	parser.add_argument('--trainset_path', default='/home/wmh/wmh/sgcn/data/escalator_dataset/labels/train.json', type=str)
	parser.add_argument('--augset_path', default='/home/wmh/wmh/sgcn/data/escalator_dataset/labels/aug_trainset.json', type=str)
	
	opt = parser.parse_args()
	data_augment(opt.trainset_path, opt.augset_path)

	