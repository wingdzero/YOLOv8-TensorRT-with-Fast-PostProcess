# YOLOv8 TensorRT with Fast PostProcess

## 介绍
本项目基于yolov8与TensorRT实现高性能推理。

Fast PostProcess是在模型转换时通过将模型的6个检测头拆分，先对原始特征进行置信度过滤，再进行nms操作。以测试图片 images/in/bus.jpg 为例，后处理总体时间由4.8ms（仅nms）降至1.5ms（DFL+nms），可以显著提高模型整体推理速度。

项目基于Fastapi访问接口，可以实现Batch推理。

## 快速上手

### 1. 环境配置
- TensorRT版本：

    本项目使用的TensorRT版本为10.1
- 安装项目依赖：

```angular2html
pip install -r requirements.txt
```

### 2. ONNX模型导出及推理（使用FastPostProcess）

为了在推理时首先基于置信度进行有效grid过滤，需要在导出模型时对检测头进行拆分，具体修改代码在 ultralytics/nn/modules/head.py 中的Detect类的forward方法中。本项目只修改了检测模型的导出，其他如关键点检测模型修改方法同理。

- 导出onnx模型（需要Batch推理的可以在导出onnx时设置）：
```angular2html
$ python export_onnx.py
```
- 推理代码中有详细的注释解释Fast PostProcess如何运行
```angular2html
$ python onnx_inference.py
```


### 3. Engine模型导出及推理（使用FastPostProcess）

- engine模型由ONNX模型转换得到，使用如下脚本进行转换：

```angular2html
$ python export_engine.py
```
- engine模型推理：
```angular2html
$ python engine_inference.py
```

### 4. api推理
- 配置推理权重及相关参数：

  模型路径、置信度阈值等参数存放在 configs/config.ini 中

- 首先启动engine推理 api服务：
```angular2html
$ python api_inference_server.py
```

- 之后发送图片并返回结果：
```angular2html
$ python send.py
```