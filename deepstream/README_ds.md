# Action Recognize with DeepStream

## 1. 代码内容：
主要的内容为如下两个脚本:  
* TRT/action_TRT.py:    动作识别模型的TensorRT推理脚本
* deepstream_action.py: 动作识别的DeepStream使用demo

## 2. 使用
### （1）使用TensorRT python API
动作识别模型的TensorRT推理封装在action_TRT.py中的 **Action_TRT** 类中, 使用范例详见action_TRT.py中代码 `smaple1` 和 `sample2` 两个函数, 设置相关文件路径后，运行:
```bash
python3 deepstream/TRT/action_TRT.py
```
**Action_TRT** 类的TensorRT推理函数主要有 `__call__` , `inference_single_frame` , `inference_single_object` , `inference_multi_frame`

### （1）使用DeepStream
在DeepStream中的使用方式为: 在其 pipeline 的 probe 函数中调用TensorRT python API（ **Action_TRT** ） 进行推理, demo为deepstream_action.py, 设置相关文件路径后，运行:
```bash
python3 deepstream/deepstream_action.py
```