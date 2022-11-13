import json
import argparse
import os
import copy
import numpy as np
import random


def merge(json_dir, out_dir, split_ratio=[0.8, 0.2], num_keyponits=14):
	"""
		把所有单个的动作标注json文件放到一个目录下
		args:
		json_dir: action annotation directory, conatins .json file
		out_path: example: xxxx.json
	"""
	# merge all json files
	json_list = os.listdir(json_dir)
	merge_json = []
	for i, js in enumerate(json_list):
		json_path = os.path.join(json_dir, js)
		with open(json_path, 'r', encoding='utf-8') as annot:
			result = json.load(annot)
		for j in result:
			merge_json.append(j)
	# print('merge_json: ',merge_json)

	# add visiable flag for the lacking of vis labels
	for j in merge_json:
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

		# delete head keypoints or eyes and nose
		if num_keyponits == 12:
			del j['keypoints'][0:5]
		elif num_keyponits == 14:
			del j['keypoints'][0:3]
		else:
			pass
	save_dir = os.path.join(out_dir, 'labels')
	if not os.path.exists(save_dir):
		os.mkdir(save_dir)

	trainset_path = os.path.join(save_dir, 'train.json')
	valset_path = os.path.join(save_dir, 'val.json')
	testset_path = os.path.join(save_dir, 'test.json')
	random.shuffle(merge_json)
	if len(split_ratio) == 2:
		num_samples = len(merge_json)
		num_train = int(split_ratio[0] * num_samples)
		train_set = merge_json[:num_train]
		test_set = merge_json[num_train:]
		print('Length of total dataset:', len(merge_json))
		print('Length of trainset:', len(train_set))
		print('Length of testset:', len(test_set))
		with open(trainset_path, 'w', encoding='utf-8') as output:
			json.dump(train_set, output, indent=4, separators=(',', ': '))
		with open(testset_path, 'w', encoding='utf-8') as output:
			json.dump(test_set, output, indent=4, separators=(',', ': '))
	elif len(split_ratio) == 3:
		num_samples = len(merge_json)
		num_train = int(split_ratio[0] * num_samples)
		num_test = int(split_ratio[1] * num_samples)
		train_set = merge_json[:num_train]
		val_set = merge_json[num_train : num_train + num_test]
		test_set = merge_json[num_train + num_test:]
		print('Length of total dataset:', len(merge_json))
		print('Length of train set:', len(train_set))
		print('Length of validation set:', len(val_set))
		print('Length of test set:', len(test_set))
		with open(trainset_path, 'w', encoding='utf-8') as output:
			json.dump(train_set, output, indent=4, separators=(',', ': '))
		with open(valset_path, 'w', encoding='utf-8') as output:
			json.dump(test_set, output, indent=4, separators=(',', ': '))
		with open(testset_path, 'w', encoding='utf-8') as output:
			json.dump(test_set, output, indent=4, separators=(',', ': '))
	elif len(split_ratio)== 1:
		print('Length of total dataset:', len(merge_json))
		print('Length of train set:', len(merge_json))
		with open(trainset_path, 'w', encoding='utf-8') as output:
			json.dump(merge_json, output, indent=4, separators=(',', ': '))



if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	# dataset args
	parser.add_argument('--json_dir', default='/root/autodl-tmp/wmh/dataset/escalator/action_label/1', type=str)
	parser.add_argument('--out_dir', default='/root/autodl-tmp/wmh/sgcn/data/escalator_dataset/', type=str)
	parser.add_argument('--ratio', default=[0.8, 0.1, 0.1], type=list)
	parser.add_argument('--num_keyponits', default=14, type=int)

	opt = parser.parse_args()
	merge(opt.json_dir, opt.out_dir, split_ratio=opt.ratio, num_keyponits=opt.num_keyponits)




	