import torch
from model.st_gcn import Model


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

class Detect():
    def __init__(self, weights, class_thr,  classmap_path, device, num_joints=14, keypoints_thr=0.45, num_invalid=4):
        self.weights = weights
        self.class_thr = class_thr
        self.in_channels = 3
        self.num_class = 4
        self.num_joints = num_joints
        self.keypoints_thr = keypoints_thr
        self.num_invalid = num_invalid
        self.edge_importance_weighting = True
        self.graph_args = {'layout': 'noeye_ear_coco', 'strategy': 'spatial'}
        self.device = device
        self.load_model()

        with open(classmap_path) as f:
            label_name = f.readlines()
            self.label_name = [line.rstrip() for line in label_name]

    def inference_single_frame(self, detection):
        person = self.get_skeleton(detection)
        action_result = []
        for i, det in enumerate(person):
            one_person = dict()
            k = det.get('keypoints')
            k = k.view(self.num_joints, 3)
            invalid_list = list(k[:, 2] < self.keypoints_thr)
            invalid_keypoints = invalid_list.count(True)
            if invalid_keypoints < self.num_invalid:
                k[..., 2] = 1
                kpts = torch.zeros(1, self.num_joints, 3)
                kpts[:, :, :] = k
                with torch.no_grad():
                    keypoints = kpts.float().to(self.device)
                    output = self.model(keypoints)
                probability = torch.softmax(output, dim=1)

                pred_label = output.argmax()
                action_label = self.label_name[pred_label]

                one_person['bbox'] = det.get('bbox')
                one_person['action'] = action_label
                one_person['confidence'] = probability[0][pred_label]
                action_result.append(one_person)
            else:
                one_person['bbox'] = det.get('bbox')
                one_person['action'] = 'unsure_pose'
                one_person['confidence'] = 0
                action_result.append(one_person)
        return action_result

    def inference_single_object(self, keypoint):
        k = keypoint[-self.num_joints * 3:]
        k = k.view(self.num_joints, 3)
        invalid_list = list(k[:, 2] < self.keypoints_thr)
        invalid_keypoints = invalid_list.count(True)
        if invalid_keypoints < self.num_invalid:
            k[..., 2] = 1
            kpts = torch.zeros(1, self.num_joints, 3)
            kpts[:, :, :] = k
            with torch.no_grad():
                keypoints = kpts.float().to(self.device)
                output = self.model(keypoints)
            probability = torch.softmax(output, dim=1)

            pred_label = output.argmax()
            action_label = self.label_name[pred_label]
            confidence = probability[0][pred_label]
        else:
            action_label = 'unsure_pose'
            confidence = 0
        return action_label, confidence

    def get_skeleton(self, detection):
        person = []
        for det_index, (*xyxy, conf, cls) in enumerate(detection[:, :6]):
            single = dict()
            if cls == 0 and conf >= self.class_thr:
                nohead_kpts = detection[det_index, 21:]
                single['keypoints'] = nohead_kpts
                coord = xyxy
                single['bbox'] = coord
            person.append(single)
        return person

    def load_model(self):
        pretrained_dict = torch.load(self.weights)
        self.model = Model(self.in_channels, self.num_class, self.num_joints, self.graph_args, self.edge_importance_weighting)
        self.model.load_state_dict(pretrained_dict, strict=False)
        self.model.to(self.device)
        self.model.eval()



