#!/usr/bin/env python3
# cython: language_level=3
# coding=utf-8
import os
import sys
import cv2
import time
import pyds
import ctypes
import numpy as np
from pathlib import Path
from datetime import datetime
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'    # 设置cuda同步，方便debug
ROOT = Path(__file__).parent.parent.absolute().resolve().__str__()
DIR = os.path.dirname(__file__)
sys.path.append(ROOT)

import gi
gi.require_version("Gst", "1.0")
gi.require_version("GstRtspServer", "1.0")
from gi.repository import Gst, GstRtspServer, GLib

from deepstream.TRT.action_TRT import Action_TRT
from deepstream.utils.utils import make_element, is_aarch64, create_source_bin, bus_call, set_tracker_config
from deepstream.utils.utils import get_total_outshape, postprocess
from deepstream.utils.display import dispaly_frame_pose,add_obj_meta


MUX_OUTPUT_WIDTH  = 640 
MUX_OUTPUT_HEIGHT = 640 
INFER_SHAPE = (1,3,MUX_OUTPUT_HEIGHT,MUX_OUTPUT_WIDTH)
OUT_SHAPE = get_total_outshape(INFER_SHAPE)
INPUT_STREAM = [
                # "file:///media/nvidia/SD/project/test/2in1_2.mp4",
                "file:///media/nvidia/SD/project/test/merge.mp4",
                ]
DEBUG = False
codec = "H264"
start_time = time.time()
start_time2 = 0
vid_writer = None



def pose_src_pad_buffer_probe(pad, info, action_classifier):
    t = time.time()
    global  data,vid_writer
    frame_number = 0
    num_rects = 0

    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer ")
        return

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))

    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        if DEBUG:
            # Getting Image data using nvbufsurface
            # the input should be address of buffer and batch_id
            n_frame = pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
            frame_image = np.array(n_frame, copy=True, order="C")
            frame_image = cv2.cvtColor(frame_image, cv2.COLOR_RGBA2BGR)

        frame_number = frame_meta.frame_num
        num_rects = frame_meta.num_obj_meta
        pad_index = frame_meta.pad_index
        l_usr = frame_meta.frame_user_meta_list

        while l_usr is not None:
            try:
                # Casting l_obj.data to pyds.NvDsUserMeta
                user_meta = pyds.NvDsUserMeta.cast(l_usr.data)
            except StopIteration:
                break

            # get tensor output
            if (user_meta.base_meta.meta_type !=
                    pyds.NvDsMetaType.NVDSINFER_TENSOR_OUTPUT_META):  # NVDSINFER_TENSOR_OUTPUT_META
                try:
                    l_usr = l_usr.next
                except StopIteration:
                    break
                continue

            try:
                tensor_meta = pyds.NvDsInferTensorMeta.cast(user_meta.user_meta_data)
                layer = pyds.get_nvds_LayerInfo(tensor_meta, 0)
                # load float* buffer to python
                ptr = ctypes.cast(pyds.get_ptr(layer.buffer), ctypes.POINTER(ctypes.c_float))
                out = np.ctypeslib.as_array(ptr, shape=OUT_SHAPE)
                
                if DEBUG:
                    pred = postprocess(out,(MUX_OUTPUT_HEIGHT,MUX_OUTPUT_WIDTH),INFER_SHAPE[2:],frame_image)
                    boxes, confs, kpts = pred
                else:
                    pred = postprocess(out,(MUX_OUTPUT_HEIGHT,MUX_OUTPUT_WIDTH),INFER_SHAPE[2:])
                    boxes, confs, kpts = pred
                
                if len(boxes)>0 and len(confs)>0 and len(kpts)>0:
                    ##############################################################################
                    ## 动作识别
                    # 多对象推理
                    # acts_label,acts_conf = action_classifier.inference_multi_object(kpts[0])
                    
                    # 单对象推理
                    acts_label,acts_conf = [],[]
                    for per_img in kpts:    # kpts的数据结构为:  [image1关键点,  image2关键点,  ...]
                        for kpt in per_img: # per_img数据结构为: [person1关键点, person2关键点, ...]
                            # act_result = action_classifier.inference_single_object(kpt)
                            act_result = action_classifier(kpt, mode='object')
                            acts_label.append(act_result[0])
                            acts_conf.append(act_result[1])
                    ##############################################################################
                    add_obj_meta(frame_meta,batch_meta,boxes[0],confs[0],acts_label,acts_conf)
                    dispaly_frame_pose(frame_meta,batch_meta,boxes[0],confs[0],kpts[0])
            except StopIteration:
                break

            try:
                l_usr = l_usr.next
            except StopIteration:
                break

        if DEBUG:
            if vid_writer == None:  # new video
                vid_writer = cv2.VideoWriter(
                    "record.avi",
                    cv2.VideoWriter_fourcc("X", "V", "I", "D"),
                    25,
                    (frame_image.shape[1], frame_image.shape[0]),
                )

            vid_writer.write(frame_image.copy())

        
        global start_time, start_time2
        cur_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        CurFPS = 1 / (time.time() - start_time)
        AvgFPS = frame_number / (time.time() - start_time2)
        # print(boxes )
        num_person = sum([len(box) for box in boxes])
        display_text = f'{cur_time} Person={num_person} Frames={frame_number} FPS={CurFPS:.0f} AvgFPS={AvgFPS:.1f}'
        print(display_text)
        
        start_time = time.time()
        if int(start_time2) == 0:
            start_time2 = time.time()
        try:
            l_frame = l_frame.next
        except StopIteration:
            break
        pyds.nvds_acquire_meta_lock(batch_meta)
        frame_meta.bInferDone=True
        pyds.nvds_release_meta_lock(batch_meta)

    print(f'probe function time cost:{(time.time()-t)*1000:.2f}ms')
    return Gst.PadProbeReturn.OK


def osd_sink_pad_buffer_probe(pad, info, u_data):
        buffer = info.get_buffer()
        batch = pyds.gst_buffer_get_nvds_batch_meta(hash(buffer))

        l_frame = batch.frame_meta_list
        while l_frame:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            l_obj = frame_meta.obj_meta_list
            while l_obj:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                # obj_meta.text_params.display_text = "person{}: {:.2f}".format(obj_meta.object_id ,obj_meta.tracker_confidence)
                text = pyds.get_string(obj_meta.text_params.display_text).rsplit(' ')[0]
                obj_meta.text_params.display_text = 'id:{} {}'.format(obj_meta.object_id ,text)
                track_box = obj_meta.tracker_bbox_info.org_bbox_coords
                # print(track_box.left,track_box.top,track_box.height,track_box.width)
                # rect_params = obj_meta.rect_params
                # rect_params.left = track_box.left
                # rect_params.top = track_box.top
                # rect_params.width = track_box.width
                # rect_params.height = track_box.height
                l_obj = l_obj.next
            l_frame = l_frame.next
        return Gst.PadProbeReturn.OK


def main(args):
    # Standard GStreamer initialization
    # Since version 3.11, calling threads_init is no longer needed. See: https://wiki.gnome.org/PyGObject/Threading
    # GObject.threads_init()
    Gst.init(None)

    # Create gstreamer elements
    # Create Pipeline element that will form a connection of other elements
    print("Creating Pipeline \n ")
    pipeline = Gst.Pipeline()
    if not pipeline:
        sys.stderr.write(" Unable to create Pipeline \n")

    nvvidconv_postosd = make_element("nvvideoconvert", "convertor_postosd")

    nvvidconv1 = make_element("nvvideoconvert", "convertor_pre")

    caps1 = Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA")
    filter1 = make_element("capsfilter", "filter1")
    filter1.set_property("caps", caps1)

    nvosd = make_element("nvdsosd", "onscreendisplay")
    nvosd.set_property('display-bbox',1)
    nvosd.set_property('display-text',1)

    # Create a caps filter
    caps = make_element("capsfilter", "filter")
    caps.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=I420"))

    transform = Gst.ElementFactory.make("nvegltransform", "nvegl-transform")

    sink = make_element("nveglglessink", "nvvideo-renderer")

    sink.set_property('sync', False)

    # Create nvstreammux instance to form batches from one or more sources.
    streammux = make_element("nvstreammux", "Stream-muxer-left")
    #streammux.set_property("enable-padding", True)
    streammux.set_property("width", MUX_OUTPUT_WIDTH)
    streammux.set_property("height", MUX_OUTPUT_HEIGHT)
    streammux.set_property("batch-size", 1)
    streammux.set_property("batched-push-timeout", 40000)
    streammux.set_property("live-source", 1)  # rtsp
    pipeline.add(streammux)

    pgie = make_element("nvinfer", "primary-inference-left")
    pgie.set_property("config-file-path", f"{DIR}/config_infer_primary.txt")

    tracker = make_element("nvtracker", "tracker")
    set_tracker_config(f"{DIR}/config_tracker.txt",tracker)
    

    number_src = len(INPUT_STREAM)
    for i in range(number_src):
        print("Creating source_bin ", i, " \n ")
        uri_name = INPUT_STREAM[i]
        print(uri_name)

        source_bin = create_source_bin(i, uri_name)
        if not source_bin:
            sys.stderr.write("Unable to create source bin \n")
        pipeline.add(source_bin)

        padname = "sink_%u" % i
        sinkpad = streammux.get_request_pad(padname)
        if not sinkpad:
            sys.stderr.write("Unable to create sink pad bin \n")
        srcpad = source_bin.get_static_pad("src")
        if not srcpad:
            sys.stderr.write("Unable to create src pad bin \n")
        srcpad.link(sinkpad)


    print("Adding elements to Pipeline \n")
    pipeline.add(pgie)
    pipeline.add(tracker)
    pipeline.add(nvvidconv_postosd)
    pipeline.add(nvvidconv1)
    pipeline.add(nvosd)
    pipeline.add(filter1)
    pipeline.add(caps)
    pipeline.add(transform)
    pipeline.add(sink)

    # Link the elements together:
    print("Linking elements in the Pipeline \n")
    streammux.link(nvvidconv1)
    nvvidconv1.link(filter1)
    filter1.link(pgie)
    pgie.link(tracker)
    tracker.link(nvosd)
    nvosd.link(nvvidconv_postosd)
    nvvidconv_postosd.link(caps)
    caps.link(transform)
    transform.link(sink)

    # create and event loop and feed gstreamer bus mesages to it
    loop = GLib.MainLoop()

    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)

    ##################################################
    # 初始化动作识别模型
    engine_path = F'{ROOT}/weights/sgcn.plan'   # TensorRT模型文件路径
    class_path  = F'{ROOT}/label_map.txt'       # 动作类别名称文件路径
    class_thres = 0.5                           # 分类阈值, 人的置信度大于该值才进行动作分类分类, 仅在图片推理(inference_single_frame)中起作用
    img_size    = INFER_SHAPE[2:]               # 推理的关键点坐标所在的图片的尺寸, (h,w)
    shape       = [1,14,3]                      # 动作识别的关键点尺寸，14个关键点的x,y坐标以及置信度构成
    num_classes = 4                             # 动作识别的类别数量
    keypoints_thr = 0.6                      # 关键点置信度，小于此值将被置为无效关键点
    num_invalid = 4                             # 无效关键点个数
    action_classifier = Action_TRT(engine_path,class_path,class_thres,img_size,shape,num_classes,keypoints_thr,num_invalid)

    # from detect import Detect
    # conf_thres = 0.1
    # action_weights = 'weights/sgcn.pt'
    # label_name_path = 'label_map.txt'
    # engine_path='/home/nvidia/project/yolo-pose/trt_models/yolov5s6_pose_640_ti_lite.trt'
    # action_classifier = Detect(action_weights, conf_thres, label_name_path)
    ###################################################

    pgiepad = pgie.get_static_pad("src")
    if not pgiepad:
        sys.stderr.write(" Unable to get src pad of tracker \n")
    ##########################
    pgiepad.add_probe(Gst.PadProbeType.BUFFER, pose_src_pad_buffer_probe, action_classifier)    # 传入动作识别模型
    ##########################

    osdpad = nvosd.get_static_pad("sink")
    if not osdpad:
        sys.stderr.write(" Unable to get sink pad of tracker \n")
    osdpad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe, 0)

    print("Starting pipeline \n")

    # start play back and listed to events
    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except:
        if vid_writer:
            vid_writer.release()
    # cleanup
    pipeline.set_state(Gst.State.NULL)
    print("End pipeline \n")


if __name__ == "__main__":
    sys.exit(main(sys.argv))