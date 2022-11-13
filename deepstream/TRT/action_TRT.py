import os
import cv2
import torch
import matplotlib
import numpy as np
import tensorrt as trt
# import pycuda.autoinit
import pycuda.driver as cuda
cuda.init()     # cuda手动初始化
import tqdm

from time import time
from PIL import Image
from pathlib import Path


class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


class Action_TRT():
    def __init__(self, engine_path, class_path, class_thres=0.5, img_size=[720, 1280], shape=[1, 14, 3],
                 num_classes=4, kpts_thr=0.6, num_invalid=4) -> None:
        self.ctx = cuda.Device(0).make_context()
        self.engine_path = engine_path  # TRT模型文件路径
        self.class_path = class_path  # 类别文件路径
        self.class_thres = class_thres  # 分类阈值
        self.img_size = img_size  # 图片尺寸
        self.shape = shape  # 输入模型的骨骼点序列尺寸
        self.num_classes = num_classes  # 动作类别数
        self.kpts_thr = kpts_thr  # 关键点置信度，小于此值将被置为无效关键点
        self.num_invalid = num_invalid  # 无效关键点个数

        self.class_name = self.load_class(class_path)
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.load_engine(engine_path, self.runtime)
        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers(self.engine)
        self.ctx.pop()

    def warmup(self, times=5):
        print(f'warmup {times} times...')
        h, w = self.img_size
        b, l, c = self.shape
        dummy_kpts = np.zeros(self.shape, dtype=np.float32)
        dummy_kpts[..., 0] = np.random.randint(0, w, (b, l))
        dummy_kpts[..., 1] = np.random.randint(0, h, (b, l))
        dummy_kpts[..., 2] = np.random.randn(b, l)
        self.preprocess(dummy_kpts, self.inputs[0].host)
        for _ in range(times):
            self.do_inference()

    def load_engine(self, engine_path, runtime):
        # load engine model
        trt.init_libnvinfer_plugins(None, '')  # load all avalilable official pluging
        with open(engine_path, "rb") as f:
            serialized_engine = f.read()
        engine = runtime.deserialize_cuda_engine(serialized_engine)
        return engine

    def load_class(self, class_path):
        # load action categories name
        with open(class_path) as f:
            class_name = f.readlines()
            class_name = [line.rstrip() for line in class_name]
        return class_name

    def allocate_buffers(self, engine):
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()
        for binding in engine:
            size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            bindings.append(int(device_mem))
            # Append to the appropriate list.
            if engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))
        return inputs, outputs, bindings, stream

    def preprocess(self, kpts, pagelocked_buffer):
        # copy data to host
        kpts = np.asarray(kpts, dtype=np.float32)
        np.copyto(pagelocked_buffer, (kpts.astype(trt.nptype(trt.float32)).ravel()))
        return kpts

    def do_inference(self):
        self.ctx.push()
        # Transfer input data to the GPU.
        [cuda.memcpy_htod_async(inp.device, inp.host, self.stream) for inp in self.inputs]
        # Run inference.
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        # Transfer predictions back from the GPU.
        [cuda.memcpy_dtoh_async(out.host, out.device, self.stream) for out in self.outputs]
        # Synchronize the stream
        self.stream.synchronize()
        self.ctx.pop()
        # Return only the host outputs.
        return [out.host for out in self.outputs]

    def __call__(self, detection, mode='frame', *args, **kwds):
        '''执行动作分类推理
            Arguments:
                detection: 待识别关键点, 形状为[N,57]或者[51,],分别对应多个对象和单个对象的情况
                mode: 'frame': 多个对象推理模式, 输入形状需为[N,57], 'object': 单个对象模式, 输入形状需为[36,]
            Return:
                action_result: 识别结果,不同模式略有不同, 具体查看对应函数(inference_single_frame, inference_single_object)
        '''
        assert mode in ['frame', 'object']
        if mode == 'object' or detection.ndim == 1:
            action_result = self.inference_single_object(detection)
        else:
            action_result = self.inference_single_frame(detection)

        return action_result

    def inference_single_frame(self, detection):
        '''推理单张图片的姿态关键点
            Arguments:
                detection: numpy.ndarray, 单帧图片的姿态估计识别结果, 尺寸: [N,57], 57对应[xyxy,conf,cls,kpts]
            Return:
                action_result: list, 对应的动作识别结果
        '''
        person = self.get_skeleton(detection)
        action_result = []
        for i, det in enumerate(person):
            one_person = dict()
            kpts = det.get('keypoints')
            kpts = kpts.reshape(self.shape)
            invalid_count = torch.sum(kpts[..., 2] < self.kpts_thr)
            if invalid_count < self.num_invalid:
                kpts[..., 2] = 1
                kpts = self.preprocess(kpts, self.inputs[0].host)
                out = self.do_inference()[0]
                pred = self.postprocess(out)

                pred_label = pred[0]
                confidence = pred[1]
                action_label = self.class_name[pred_label]
                one_person['bbox'] = det.get('bbox')
                one_person['action'] = action_label
                one_person['confidence'] = confidence
            else:
                one_person['bbox'] = det.get('bbox')
                one_person['action'] = 'unsure_pose'
                one_person['confidence'] = 0
            action_result.append(one_person)
        return action_result

    def inference_single_object(self, keypoint):
        '''推理单个对象的姿态关键点
            Arguments:
                keypoint: numpy.ndarray, 单个人的姿态估计识别结果, 尺寸: [42,], 14个关键点 x 3
            Return:
                action_label: str, 动作类别
                confidence: float, 动作置信度(概率)
        '''
        kpts = keypoint.reshape(self.shape)

        invalid_count = torch.sum(kpts[..., 2] < self.kpts_thr)
        if invalid_count < self.num_invalid:
            # t1 = time()
            kpts[..., 2] = 1
            kpts = self.preprocess(kpts, self.inputs[0].host)
            # t2 = time()
            out = self.do_inference()[0]
            # t3 = time()
            pred = self.postprocess(out)
            # t4 = time()
            # print(f'pre: {(t2-t1)*1000:.2f}, infer: {(t3-t2)*1000:.2f},post: {(t4-t3)*1000:.2f}')

            pred_label = pred[0]
            confidence = pred[1]
            action_label = self.class_name[pred_label]
        else:
            confidence = 0
            action_label = 'unsure_pose'
        return action_label, confidence

    def inference_multi_object(self, keypoints):
        '''推理多个对象的姿态关键点\n
            Arguments:
                keypoints: numpy.ndarray, 多个人的姿态估计识别结果, 尺寸: [N,42], 14个关键点 x 3
            Return:
                action_labels: list, 动作类别
                confidences: list, 动作置信度(概率)
        '''
        action_labels, confidences = [], []
        for kpt in keypoints:
            action_result = self.inference_single_object(kpt)
            action_labels.append(action_result[0])
            confidences.append(action_result[1])
        return action_labels, confidences

    def postprocess(self, model_out):
        # get action label and prob
        def softmax(logits):
            e_x = np.exp(logits)
            probs = e_x / np.sum(e_x, axis=-1, keepdims=True)
            return probs

        out = softmax(model_out)
        pred_label = np.argmax(out, axis=-1)
        pred_prob = out[pred_label]
        return pred_label, pred_prob

    def get_skeleton(self, detection):
        person = []
        for det_index, (*xyxy, conf, cls) in enumerate(detection[:, :6]):
            single = dict()
            if cls == 0 and conf >= self.class_thres:
                kpts = detection[det_index, 6:]
                single['keypoints'] = kpts
                coord = xyxy
                single['bbox'] = coord
            person.append(single)
        return person

def plot_action_label(x, im, font_color=None, label=None, line_thickness=3):
    # Plots one bounding box on image 'im' using OpenCV
    assert im.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to plot_on_box() input image.'
    tl = line_thickness or round(0.002 * (im.shape[0] + im.shape[1]) / 2) + 1  # line/font thickness
    # color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    # cv2.rectangle(im, c1, c2, font_color, thickness=tl*1//3, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 6, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(im, c1, c2, font_color, -1, cv2.LINE_AA)  # filled
        cv2.putText(im, label, (c1[0], c1[1] - 2), 0, tl / 6, [0,0,0], thickness=tf//2, lineType=cv2.LINE_AA)


def sample1():
    import torch
    ## 测试范例
    # ctx = cuda.Device(0).make_context()
    engine_path = '../../weights/sgcn_3layer_nohead.trt'  # 动作识别TensorRT模型文件路径
    classmap_path = '../../label_map.txt'  # 动作识别类别文件路径
    class_thres = 0.5  # 动作识别阈值, 低于该值不进行动作识别, 仅在frame推理模式(inference_single_frame)下有效
    img_size = [720, 1280]  # 待识别对象所在图片的尺寸  [height,weight]
    shape = [1, 14, 3]  # 动作识别模型的输入尺寸    [batch_size,length,channel]
    num_classes = 4  # 动作识别类别数量
    keypoints_thr = 0.45
    num_invalid = 4

    classfier = Action_TRT(engine_path, classmap_path, class_thres, img_size, shape, num_classes, keypoints_thr,
                           num_invalid)
    h, w = img_size  # 待识别对象所在图片的尺寸  [height,weight]
    b, l, c = shape  # 动作识别模型的输入尺寸    [batch_size,length,channel]

    # # single frame mode
    # dets = np.array([0,0,0,0,0.8,0]).reshape(b,-1)  # [xyxy,conf,class]
    # kpts = np.zeros(shape,dtype=np.float32)
    # kpts[...,0] = np.random.randint(0,w,(b,l))  # fake input
    # kpts[...,1] = np.random.randint(0,h,(b,l))
    # kpts[...,2] = np.random.randn(b,l)
    # kpts = kpts.reshape(b,-1)
    # dets = np.concatenate([dets,kpts],axis=-1)  # [1,57]
    #
    # action_result = classfier(dets,mode='frame')
    # print(action_result)

    # single object mode
    kpts = np.zeros(shape, dtype=np.float32)
    kpts[..., 0] = np.random.randint(0, w, (b, l))  # fake input
    kpts[..., 1] = np.random.randint(0, h, (b, l))
    kpts[..., 2] = np.random.randn(b, l)
    kpts = torch.tensor(kpts)
    action_result = classfier(kpts, mode='object')

    print(action_result)
    # ctx.pop()


def sample2():
    # 结合姿态估计使用
    from pose_TRT import Pose_TRT

    # load pose model
    engine_path = '/root/autodl-tmp/wmh/yolopose/weights/yolov5l6_pose_832_scratch-FP16.trt'
    input_shape = [1, 3, 832, 832]
    confidence = 0.4
    iou_thresh = 0.5
    pose_estimater = Pose_TRT(engine_path, input_shape=input_shape, conf=confidence, iou=iou_thresh)

    # load action model
    engine_path = '/root/autodl-tmp/wmh/sgcn/weights/sgcn_3layer_nohead_fp16.trt'
    classmap_path = '/root/autodl-tmp/wmh/yolopose/label_map.txt'
    class_thres = 0.5
    img_size = [720, 1280]
    shape = [1, 14, 3]
    num_classes = 4
    keypoints_thr = 0.45
    num_invalid = 4

    classfier = Action_TRT(engine_path, classmap_path, class_thres, img_size, shape, num_classes, keypoints_thr,
                           num_invalid)
    h, w = img_size
    b, l, c = shape

    # inference
    img_path = '/root/autodl-tmp/wmh/sgcn/test.jpg'
    im = cv2.imread(img_path)
    img = pose_estimater.preprocess(im, pose_estimater.inputs[0].host)
    out = pose_estimater.do_inference_v2()
    pred = pose_estimater.postprocess(out, im, img, has_nms=False, draw=True)
    boxes, confs, kepts = pred
    for per_img in kepts:  # kpts的数据结构为:  [image1关键点,  image2关键点,  ...]
        for kpt in per_img:  # per_img数据结构为: [person1关键点, person2关键点, ...]
            kpt = kpt[-l * 3:]
            kpt = torch.from_numpy(kpt)
            action_result = classfier(kpt, mode='object')
            print(action_result)


def video_sample():
    # 结合姿态估计使用
    from pose_TRT import Pose_TRT
    line_thickness = 3

    # load pose model
    engine_path = '/root/autodl-tmp/wmh/yolopose/weights/yolov5l6_pose_832_scratch-FP16.trt'
    input_shape = [1, 3, 832, 832]
    confidence = 0.4
    iou_thresh = 0.5
    pose_estimater = Pose_TRT(engine_path, input_shape=input_shape, conf=confidence, iou=iou_thresh)

    # load action model
    engine_path = '/root/autodl-tmp/wmh/sgcn/tools/sgcn_scale.trt'
    classmap_path = '/root/autodl-tmp/wmh/yolopose/label_map.txt'
    class_thres = 0.5
    img_size = [720, 1280]
    shape = [1, 14, 3]
    num_classes = 4
    keypoints_thr = 0.6
    num_invalid = 4

    classfier = Action_TRT(engine_path, classmap_path, class_thres, img_size, shape, num_classes, keypoints_thr,
                           num_invalid)
    h, w = img_size
    b, l, c = shape

    # inference
    video_path = '/root/autodl-tmp/wmh/data/futi_test_video/4mm_36.mp4'

    vid_cap = cv2.VideoCapture(video_path)
    fps, h, w = vid_cap.get(cv2.CAP_PROP_FPS), vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT), vid_cap.get(
        cv2.CAP_PROP_FRAME_WIDTH)
    fps, h, w = int(fps), int(h), int(w)
    vid_writer = cv2.VideoWriter('/root/autodl-tmp/wmh/data/futi_test_video/res1/4mm_36_res_trt_fp32_flip.mp4',
                                 cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    stillgo, im = vid_cap.read()

    pbar = tqdm(desc='FPS:0', total=int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    b_time = time()
    # llll = []
    while stillgo:
        read_start = time()
        stillgo, im = vid_cap.read()
        if not stillgo:
            break
        pre_start = time()
        img = pose_estimater.preprocess(im, pose_estimater.inputs[0].host)
        pre_end = time()
        out = pose_estimater.do_inference_v2()
        pose_end = time()
        pred = pose_estimater.postprocess(out, im, img, has_nms=False, draw=True)
        pose_postpro = time()
        boxes, confs, kepts = pred
        # print('boxes[0]: ', boxes[0])
        # print('kepts[0]: ', kepts[0])

        if len(kepts) == 0:
            pass
        else:
            for p, kpt in enumerate(kepts[0]):  # per_img数据结构为: [person1关键点, person2关键点, ...]
                kpt = kpt[-l * 3:]
                # llll.append(kkk)
                kpt = torch.from_numpy(kpt)
                action_start_time = time()
                action_label, action_confidence = classfier(kpt, mode='object')
                action_end_time = time()

                if action_confidence <= 0.7 and action_label != 'unsure_pose':
                    total_label = 'unsure' + ' ' + action_label
                else:
                    total_label = action_label + ' ' + str(round(float(action_confidence), 2))


                if 'normal' in total_label:
                    plot_action_label(boxes[0][p], im, font_color=[255, 255, 255], label=total_label,
                                      line_thickness=line_thickness)
                elif 'unsure' in total_label:
                    plot_action_label(boxes[0][p], im, font_color=[0, 255, 255], label=total_label,
                                      line_thickness=line_thickness)
                else:
                    plot_action_label(boxes[0][p], im, font_color=[0, 0, 255], label=total_label,
                                      line_thickness=line_thickness)

        post_end = time()

        read_time = pre_start - read_start
        pre_time = pre_end - pre_start
        pose_infer_time = pose_end - pre_end
        post_time = pose_postpro - pose_end
        # action_use_time = action_end_time - action_start_time
        use_time = post_end - read_start
        msg = 'r:{:.1f},p:{:.1f},i:{:.1f},p:{:.1f},t:{:.1f}'.format(read_time * 1000, pre_time * 1000,
                                                                    pose_infer_time * 1000, post_time * 1000,
                                                                    use_time * 1000)
        print('time: ', msg)
        FPS = int(1 / use_time)
        # print(f'{(time()-t)*1000:.2f}ms, infer->{(t1-t)*1000:.2f}, post->{(time()-t1)*1000:.2f}')
        cv2.putText(im, f'FPS:{FPS}', (20, 20), cv2.LINE_AA, 1, (0, 255, 255))
        # cv2.putText(im, f'get image: {read_time * 1000:.2f}ms', (20, 50), cv2.LINE_AA, 1, (255, 255, 255))
        # cv2.putText(im, f'preprocess: {pre_time * 1000:.2f}ms', (20, 80), cv2.LINE_AA, 1, (255, 255, 0))
        # cv2.putText(im, f'inference: {infer_time * 1000:.2f}ms', (20, 110), cv2.LINE_AA, 1, (255, 255, 0))
        # cv2.putText(im, f'postprocess: {post_time * 1000:.2f}', (20, 140), cv2.LINE_AA, 1, (255, 255, 0))
        # cv2.putText(im, f'sum: {use_time * 1000:.2f}', (20, 170), cv2.LINE_AA, 1, (255, 255, 0))
        vid_writer.write(im)
        pbar.set_description_str(F'FPS:{FPS}')
        pbar.set_postfix_str(msg)
        pbar.update()
    e_time = time()
    print('total time----', e_time - b_time)
    pbar.close()
    vid_writer.release()


if __name__ == "__main__":
    # sample1()
    # sample2()
    video_sample()


