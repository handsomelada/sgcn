# SGCN单帧动作分类开发文档
&emsp;&emsp;此仓库修改了ST-GCN的模型代码，删除了时序卷积，只在空间维度采用图卷积


## 一、算法训练
### 1.1 代码克隆
运行如下命令克隆训练代码，并安装训练代码的环境依赖：
```bash
# 1.获取训练代码（SGCN代码、训练、数据处理与相关优化脚本）
git clone https://github.com/handsomelada/sgcn.git

# 2.安装环境依赖
cd sgcn & pip install -r requirements.txt
```

### 1.2 数据准备
按照如下步骤进行：
1. 合并采集数据

&emsp;&emsp;将动作分类标注数据的json文件全部放在escalator_dataset/annotations文件夹下。
```bash
...
escalator_dataset
  |--annotations
  |		|--00001.json
  |		|--00007.json
  |		|--00013.json
  |		|--...
...
```
2. 处理数据并划分数据集
```bash

# 运行如下命令，将在escalator_dataset/下生成labels文件夹，里面保存了生成的训练、验证以及测试文件
python tools/generate_annotation.py \
	--json_dir escalator_dataset/annotations \
	--out_dir escalator_dataset/labels \
    --ratio [0.8, 0.2] \
    --num_keyponits 14

## 参数含义
## --json_dir	    json标注文件所在文件夹
## --out_dir	    训练、验证与测试标签保存的文件夹
## --ratio          数据集划分比例：
##                      1.若数据集太小，只需要训练集，设置[1]
##                      2.若划分为训练集与测试集，设置[0.8, 0.2]
##                      3.若划分为训练集、验证集与测试集，设置[0.8, 0.1，0.1]
## --num_keyponits  输入网络的COCO关键点个数，目前设置为12、14、17：
##                      1. 12对应除去头部五个点后剩余的关键点
##                      2. 14对应除去双眼和鼻子三个点后剩余的关键点
##                      3. 17对应COCO完整的关键点
```

3. 数据扩增

&emsp;&emsp;训练集可使用水平镜像反转、缩放、平移等数据扩增手段扩增数据，在本实验中发现镜像反转与缩放等扩增手段有明显的效果提升，因此只采用数据的镜像反转与缩放，扩增步骤如下所示：

```bash
# 运行如下命令，将在escalator_dataset/labels下生成扩增后的数据文件
python tools/data_augmentation.py \
	--trainset_path escalator_dataset/labels/train.py \
	--augset_path escalator_dataset/labels/aug_train.py \

## 参数含义
## --trainset_path	    原始训练标注文件路径
## --augset_path	    扩增后训练标注文件路径
```

### 1.3 模型训练
&emsp;&emsp;在终端运行train.py，命令行如下所示：
```bash
python train.py
```
&emsp;&emsp;训练参数由参数配置文件sgcn_4_layer.py配置，当需要修改训练参数时，直接修改sgcn_4_layer.py即可，然后直接在终端运行python train.py。
参数说明：
```python
train_args = dict(
    # 模型参数
    in_channels=3,  # 输入通道
    num_class=4,    # 动作类别
    num_joints=14,  # 关键点个数
    graph_args={'layout': 'noeye_ear_coco', 'strategy': 'spatial'}, #关键点配置与分配策略参数
    edge_importance_weighting=True, # 图边权重

    # 训练参数
    device='cuda:0', # 指定训练设备
    trainset_path='./dataset/escalator/action_label/aug_trainset.json', # 训练集路径
    valset_path='./dataset/escalator/action_label/val.json', # 验证集路径
    log_dir='log/', # 训练日志保存路径
    weights_dir='weights/', #训 练权重保存路径
    lr=0.1, # 初始学习率
    batch_size=32, # 数据批次大小
    total_epoch=250, #训练轮次
    steps=[50, 100, 150, 200], # 多阶段学习率衰减步数
    save_best=True, # 是否只保存最优模型
)
```
&emsp;&emsp;使用Tensorboard可查看训练精度与损失曲线。

### 1.4 模型测试
&emsp;&emsp;在终端运行action_test.py，命令行如下所示：
```bash
python action_test.py
```
&emsp;&emsp;测试参数由参数配置文件sgcn_4_layer.py配置，当需要修改测试参数时，直接修改sgcn_4_layer.py即可，然后直接在终端运行python eval.py。
参数说明：
```python
test_args = dict(
    # 模型参数
    in_channels=3,  # 输入通道
    num_class=4,    # 动作类别
    num_joints=14,  # 关键点个数
    graph_args={'layout': 'noeye_ear_coco', 'strategy': 'distance'}, #关键点配置与分配策略参数
    edge_importance_weighting=True, # 图边权重

    # 测试参数
    device='cuda:0', # 指定训练设备
    testset_path='/root/autodl-tmp/wmh/dataset/escalator/action_label/test.json', # 测试集路径
    log_dir='log/',  # 测试日志保存路径
    weights_path='weights/SGCN_distance_best_epoch.pt', # 训练权重加载存路径
    batch_size=32, # 数据批次大小
)
```
### 1.5 新增动作数据（类别）训练指南
新增的动作如果是：
1. 简单动作且环境无关的，可以直接采集数据标注训练；
2. 复杂动作，需要根据动作序列来判断，那么ST-GCN是不适合用于实时部署使用的，因为加上多帧信息与时序卷积后运行时间会大大加长，而且显存消耗会增大，因为它原来就是拿来做视频动作分类的，不是在线的框架；
3. 环境相关动作，比如抽烟，打手机，需要结合骨架之外的信息，那S-GCN、ST-GCN均不能达到要求，因为其输入是人体关键点信息。


### 1.6 动作分类实验结果
* 总样本量：17884，训练集：14307个动作样本，测试集:3577个动作样本
* 数据集划分比例：训练：测试 = 8:2 
* 物体标签分配：['lying', 'squatting', 'stoop', 'normal']-->['摔倒', '蹲坐', '弯腰', '正常（其他）']

训练结果如下：
<p align="center"><font face="黑体" size=3.>表3-1 SGCN验证集结果</font></p>
<div align="center">

 |   模型   |   分区策略   | 平均精度 |  摔倒  |  蹲坐  |  弯腰  | 正常（其他） |
|:------:|:--------:|:----:|:----:|:----:|:----:|:------:|
 | 3层SGCN | Spatial  | 0.96 | 0.92 | 0.96 | 0.85 |  0.99  |
| 3层SGCN | Distance | 0.96 | 0.92 | 0.95 | 0.85 |  0.99  |
| 4层SGCN | Spatial  | 0.97 | 0.95 | 0.97 | 0.88 |  0.98  |
| 4层SGCN | Distance  | 0.96 | 0.94 | 0.96 | 0.85 |  0.99  |

</div>

## 二、代码接口说明
&emsp;&emsp;1. detect.py为pytorch接口文件,参数说明及使用方法如下：
```python
# ----------------------------------- #
#   class Detect():
#        1.__init__(self, weights, class_thr, classmap_path,
#        device, num_joints=12, keypoints_thr=0.5, num_invalid=4)：初始化
#            weights: 动作分类模型权重文件路径
#            class_thr: person bbox 阈值，小于此阈值不进行动作分类
#            classmap_path: 动作分类class map文件路径
#            device: 计算设备
#            num_joints: 送入网络关键点个数，此处删除头部关键点为17-3=14
#            keypoints_thr: yolopose输出的关键点置信度，小于此值将被置为无效关键点
#            num_invalid：无效关键点个数，当一个对象中无效关键点个数大于此值时将输出'unsure_pose'
#        2.load_model(self): 加载动作分类模型
#        3.get_skeleton(self, detection):  输入：单帧yolopose输出  输出：单帧person列表，包含每个人keypoints,bbox
#            detection：yolopose输入，输入为(1, number_person, 57)
#               1: 单帧
#               number_person: 一帧中人数
#               57: 6+51 (6:bbox+conf+class)(51: 17 keypoints * 3)
#        4.inference_single_frame(self, det): 输入：单帧yolopose输出  输出：单帧action_result列表，包含每个人bbox坐标,动作类别，置信度
#        5.inference_single_object(self, keypoint): 输入：单个对象  输出：对象动作类别，置信度
# ----------------------------------- #

# ----------------------------------- #
#   接口使用方法：
#       1.实例化类，并初始化加载模型，例：
#       detector = Detect(action_weights, class_thr, classmap_path,
#       device, num_joints=14, keypoints_thr=0.5, num_invalid=4)
#
#       2.使用inference_single_frame方法获得动作识别模型的输出
#       例：results = detector.inference(det)
#
#       3.或者使用inference_single_object方法获得动作识别模型的输出
#       例： action_label, action_confidence = detector.inference(kpts)
# ----------------------------------- #
```

&emsp;&emsp;2. TensorRT与DeepStream推理接口说明：deepstream目录下action_TRT.py、deepstream_action.py为接口文件，附带demo测试，具体的使用方法参考代码相应类的注释。