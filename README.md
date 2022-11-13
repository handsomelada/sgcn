# 动作分类开发文档

<div align="right">
    王明晖 <br/>
    2022.11.13
</div>

## 一、算法调研
### 1.1 需求分析
* 需求描述

&emsp;&emsp;（1）识别乘梯人员在运行的扶梯或人行步道出入口逗留、拥堵、踩踏、蹲坐、弯腰的报警及联动停梯功能，如发生连续 5s 或以上拥堵或踩踏应立即联动停梯。

&emsp;&emsp;（2）识别乘梯人员在扶梯或人行步道全区域的跌倒报警及联动停梯功能，当发生乘梯人员在梯路上跌倒应立即联动停梯。

* 需求分析

&emsp;&emsp;算法需要识别出扶梯及人行步道环境下的人体特定动作，包括：逗留、拥堵、踩踏、蹲坐、弯腰与跌倒动作，其中逗留、拥堵、踩踏属于人体目标间关系分析，不属于单人动作分类内容，因此不在本算法开发内容内。
	
&emsp;&emsp;综上，本算法对应的目标需求为：蹲坐、弯腰、跌倒与其他（正常）类别的动作识别。

* 动作特点分析

&emsp;&emsp;从执行动作的时空特征分析，人体动作大致可分为复杂动作与简单动作，其中复杂动作需要根据时序与空间信息来判断动作类别，如：跑步、挥手等；简单动作只需要根据空间信息来判断，如躺、坐、蹲等。

&emsp;&emsp;从执行动作的环境交互特征分析，人体动作大致可分为环境相关与环境无关动作，其中环境相关动作需要根据与人体交互的物体来判断，如：打电话，抽烟等；环境无关动作不需要根据与人体交互的物体来判断，如：躺、坐、蹲等。

&emsp;&emsp;因此，需求中的蹲坐、弯腰、跌倒均属于简单动作与环境无关动作。

### 1.2 算法调研
&emsp;&emsp;根据1.1的需求分析，可确定算法的特点为：（1）利用空间信息，无需时序信息；（2）无需利用人体交互物体信息。

&emsp;&emsp;经算法调研，现有的图片动作识别主要方法有：detection-based与skeleton-based，基于检测的方法其输入为人体RGB图像，输出为动作类别，其算法特点是端到端输出动作类别，对于整体任务而言计算速度快，但是容易受到人体附着物以及背景的干扰；基于姿态的方法其输入为人体关键点坐标，输出为动作类别，其算法特点是模型尺寸以及计算开销较小，而且不会受到人体附着物以及背景的干扰，识别精度高。考虑到项目中区域入侵需要用到人体关键点信息，因此直接采用skeleton-based方法是较优方案。

&emsp;&emsp;在skeleton-based方法中又可分为传统方法、基于CNN与基于GCN的方法，传统方法手工提取人体运动学特征，计算开销低，但是算法泛化性能差，精度低；基于CNN的方法与基于GCN的方法，从输入特征图的尺寸以及模型尺寸定性对比，基于GCN方法的特征图尺寸以及模型的尺寸均比CNN的方法低（14*3 + small GCN model size VS 64*64 + middle CNN model），因此选择GCN算法为动作识别任务的目标算法。

## 二、算法原理
### 2.1 算法名称
&emsp;&emsp;SGCN（Spatial Graph Convolution Networks空间图卷积）, 图神经网络(Graph Neural networks, GCNs)是一种将卷积神经网络(convolutional Neural networks, CNNs)推广到任意结构的图的方法。本算法通过对人体关键点进行图网络建模，自动捕捉2D人体关键点的空间特性。
### 2.2 算法原理
&emsp;&emsp;首先输入单帧人体骨骼节点的2维坐标，根据给定的关节连接信息和顺序可以构造一个以关节坐标为图节点，人体结构的连接为边的空间图来作为SGCN的输入。对输入数据进行多层图卷积运算，在图上生成更高层次的特征图。然后输入到SoftMax分类器进行动作分类(Action Classification)。整个模型采用端到端反向传播的方式进行训练。
<div align=center>
<img src="readme_files/图1 人体骨架图.png" width=320>
<center>图1 人体骨架图 </center> 
</div>
&emsp;&emsp;如图1所示，该模型建立在一系列骨架图之上，其中每个节点(node)对应人体的一个关节，边缘(edge)即符合人体关节点自然连通性的空间边缘。

&emsp;&emsp;SGCN的空间构型的层次特性消除了手工设计运动学特征的需要，这让模型具有了更强的动作特征表达能力，而且也使它更容易推广到不同的环境中。在通用的GCN公式的基础上，新设计的图卷积核的策略被应用到SGCN中。
### 2.3 骨架数据前处理
&emsp;&emsp;根据姿态估计模型输出的关键点置信度，我们将关键点置信度小于阈值（keypoints_thr）的关键点置为无效关键点，将无效关键点个数大于阈值（num_invalid）的个体设置为无效姿态，并输出“unsure pose”，即不确定的姿态，降低误识别率。

### 2.4 骨架图的构建
&emsp;&emsp;SGCN的输入是包含2D 坐标的特征向量以及相应关节点的置信度，通过构造无向图G=(V，E)来表征人体骨架图，如图2所示。SGCN根据需求动作的特点，优化了COCO人体关键点的数量和顺序，将头部的三个关键点（鼻子，左眼、右眼）删除，进一步提高了动作的识别精度和模型的推理速度。
<div align=center>
<img src="readme_files/图2 优化后的骨架图.png" width=320>
<center>图2 优化后的骨架图 </center> 
</div>

### 2.5 空间构型划分策略
&emsp;&emsp;一种基于对关节点分组的分类的策略，可分为单标签分区策略、距离分区策略、空间配置分区策略。

* 单标签分区策略：最简单的分区策略是有子集，即整个邻居集本身。在此策略中，每个相邻节点上的特征向量将具有相同权重向量的内积，如图3的所有点与相邻节点（绿色）。
<div align=center>
<img src="readme_files/图3 单标签分区策略示意图.png" width=320>
<center>图3 单标签分区策略示意图 </center> 
</div>
* 距离分区策略：根据节点到根节点的距离来划分邻居集。在这项工作中，因为我们设置 D = 1，所以邻居集将被分成两个子集，其中 d = 0 指的是根节点本身（图4绿色点），其余的邻居节点（图4黄色点）在 d = 1 子集中，。因此有两个不同的权重向量，它们能够对局部微分属性进行建模，例如关节之间的相对平移。
<div align=center>
<img src="readme_files/图4 距离分区策略示意图.png" width=320>
<center>图4 距离分区策略示意图 </center> 
</div>
* 空间配置分区策略：由于人体骨架在空间上是局部的，我们可以人为的定义3个子集类别，且将骨架中所有关节的平均坐标视为其重心。第一类是根结点本身（图5绿色点），第二类是邻居节点到人体重心（图5黑色十字）的距离小于根结点到重心的距离的节点（图5红色点），第三类表示其他情况（图5黄色点）。
<div align=center>
<img src="readme_files/图5 空间分区策略示意图.png" width=320>
<center>图5 空间分区策略示意图 </center> 
</div>

### 2.6 可学习的边重要性权重
&emsp;&emsp;因为人体在运动时，某几个关节经常时成团运动（如手腕和肘），并且可能出现在身体的各个部分，因此这些关节的建模应包含有不同的重要性。为此，SGCN为每层添加了一个可学习的掩膜，它基于骨骼图中边的信息学习到的重要性权重来衡量该节点特征对其相邻节点的贡献度。即：SGCN为由人体关键点构成的图中的每个边都赋予了一个衡量这条边所连接的两个节点间相互影响大小的值，而这个值是通过图的边信息训练学习得到的。

### 2.7 图卷积
&emsp;&emsp;单个框架内关节的体内连接由邻接矩阵A和表示自连接的单位矩阵I表示。在单帧情况下，具有多个子集分区策略的SGCN可以通过以下公式实现：
$$f_{out}=∑_jΛ_j^{-1/2}  A_j Λ_j^{-1/2} f_{in} W_j$$
&emsp;&emsp;其中$Λ_j^{-1/2}=∑_kA_j^{ik} +α$，A为邻接矩阵，由多个子图的邻接矩阵堆叠而成，W为权重矩阵，图卷积是通过执行标准2D卷积来实现的，特征图可表示为(P, C)维的张量，并将得到的张量与规范化邻接矩阵$Λ_j^{-1/2} A_j Λ_j^{-1/2}$在第二维度上相乘得到。

### 2.8 模型结构
<div align=center>
<img src="readme_files/图6  ONNX格式导出的模型结构图1.png" width=320>
<img src="readme_files/图6  ONNX格式导出的模型结构图2.png" width=320>
<center>图6 ONNX格式导出的模型结构图 </center> 
</div>

<div align=center>
<img src="readme_files/图7 单层SGCN卷积示意图.png" width=320>
<center>图7 单层SGCN卷积示意图 </center> 
</div>

&emsp;&emsp;经计算，四层SGCN + Distance分区策略的模型总参数量为1.404M，FLOPs为100.874K。

说明：
1. 模型结构输入的1×3×14指的是输入batch ×关键点二维坐标与置信度×关键点个数；
2. reducesum输入的data是邻接矩阵，其数据结构为以人体关键点与连接的肢体组成的图的邻接矩阵；
3. 一层的SGCN如图7所示; 
4.  matmul和add算子对应的是矩阵相乘和矩阵相加，这一部分是邻接矩阵和上一层的卷积输出相乘相加的结果。

### 2.9 模型输出后处理
&emsp;&emsp;我们将模型输出动作对应置信度小于阈值的动作对象设置为 “unsure action”，即为不确定动作，降低误识别率。

## 三、算法训练
### 3.1 代码克隆
运行如下命令克隆训练代码，并安装训练代码的环境依赖：
```bash
# 1.获取训练代码（SGCN代码、训练、数据处理与相关优化脚本）
git clone https://github.com/handsomelada/sgcn.git

# 2.安装环境依赖
cd sgcn & pip install -r requirements.txt
```

### 3.2 数据准备
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

### 3.3 模型训练
&emsp;&emsp;在终端运行train.py，命令行如下所示：
```bash
python train.py
```
&emsp;&emsp;训练参数由参数配置文件sgcn_3_layer.py配置，当需要修改训练参数时，直接修改sgcn_3_layer.py即可，然后直接在终端运行python train.py。
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
&emsp;&emsp;单卡3090训练四层SGCN、250个Epoch的时长在半个小时左右，使用Tensorboard可查看训练精度与损失曲线。

### 3.4 模型测试
&emsp;&emsp;在终端运行eval.py，命令行如下所示：
```bash
python eval.py
```
&emsp;&emsp;测试参数由参数配置文件sgcn_3_layer.py配置，当需要修改测试参数时，直接修改sgcn_3_layer.py即可，然后直接在终端运行python eval.py。
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
### 3.5 新增动作数据（类别）训练指南
新增的动作如果是：
1. 简单动作且环境无关的，可以直接采集数据标注训练；
2. 复杂动作，需要根据动作序列来判断，那么ST-GCN是不适合用于实时部署使用的，因为加上多帧信息与时序卷积后运行时间会大大加长，而且显存消耗会增大，因为它原来就是拿来做视频动作分类的，不是在线的框架；
3. 环境相关动作，比如抽烟，打手机，需要结合骨架之外的信息，那S-GCN、ST-GCN均不能达到要求，因为其输入是人体关键点信息。


### 3.6 动作分类实验结果
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

## 四、识别结果

<div align=center>
<img src="readme_files/图8 识别结果示意图.png" width=832>
<center>图8 识别结果示意图 </center> 
</div>

&emsp;&emsp;识别结果说明：动作分类模型SGCN的输入为如图2所示的14个人体关键点（除去头部的三个关键点）的坐标与置信度，输出为动作类别与置信度。若输入的人体关键点置信度低的关键点大于等于4个，则输出“未知姿态”。

## 五、代码接口说明
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