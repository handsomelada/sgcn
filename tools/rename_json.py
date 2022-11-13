import os

if __name__ == '__main__':
    dirs = '/root/autodl-tmp/wmh/dataset/escalator/action_label/2'
    new_dirs = '/root/autodl-tmp/wmh/dataset/escalator/action_label/1'
    lists = os.listdir(dirs)
    for i, j_file in enumerate(lists):
        ori_path = os.path.join(dirs, j_file)
        new_name = j_file.split('.')[0] + '_futi' + '.json'
        new_path = os.path.join(new_dirs, new_name)
        os.rename(ori_path, new_path)