#
# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
import sys
import time
import argparse
import tensorrt as trt

# 将当前脚本目录添加到系统路径中
sys.path.insert(1, os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

from utils.image_batcher import ImageBatcher
from utils.utils import *


class TensorRTInfer:

    def __init__(self, engine_path):
        # 加载 TensorRT engine
        self.logger = trt.Logger(trt.Logger.ERROR)  # 记录推理过程中的错误和警告日志
        trt.init_libnvinfer_plugins(self.logger, namespace="")  # 初始化 TensorRT 的所有插件库
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            assert runtime  # 确保 runtime 正常加载
            self.engine = runtime.deserialize_cuda_engine(f.read())  # 反序列化引擎
        assert self.engine
        self.context = self.engine.create_execution_context()  # 创建推理执行上下文，用于管理推理过程中实际的计算和资源调度
        assert self.context

        # 为输入和输出的张量分配 GPU 内存，并将其与实际的数据绑定
        self.inputs = []
        self.outputs = []
        self.allocations = []
        for i in range(self.engine.num_io_tensors):  # 逐个获取模型的 tensor，并判断是输入 tensor 还是输出 tensor
            name = self.engine.get_tensor_name(i)  # 获取 tensor 的名称
            is_input = False
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:  # 判断是否为输入张量
                is_input = True
            dtype = np.dtype(trt.nptype(self.engine.get_tensor_dtype(name)))  # 获取 tensor 的数据类型
            shape = self.context.get_tensor_shape(name)  # 获取 tensor 的形状
            if is_input and shape[0] < 0:  # 动态输入时调整为最大形状
                assert self.engine.num_optimization_profiles > 0
                profile_shape = self.engine.get_tensor_profile_shape(name, 0)
                assert len(profile_shape) == 3  # profile 包含 min、opt、max
                self.context.set_input_shape(name, profile_shape[2])  # 设置最大输入 shape
                shape = self.context.get_tensor_shape(name)
            if is_input:
                self.batch_size = shape[0]  # 设置 batch 大小
            size = dtype.itemsize
            for s in shape:
                size *= s  # 计算 tensor 所需的内存大小
            allocation = cuda_call(cudart.cudaMalloc(size))  # 使用 pycuda 为 tensor 分配 GPU 内存
            host_allocation = None if is_input else np.zeros(shape, dtype)  # 如果是输出张量，则在 CPU 上分配内存
            binding = {
                "index": i,
                "name": name,
                "dtype": dtype,
                "shape": list(shape),
                "allocation": allocation,
                "host_allocation": host_allocation,
            }
            self.allocations.append(allocation)
            if is_input:
                self.inputs.append(binding)  # 存储输入 tensor 信息
            else:
                self.outputs.append(binding)  # 存储输出 tensor 信息
            print(
                "{} '{}' with shape {} and dtype {}".format(
                    "Input" if is_input else "Output",
                    binding["name"],
                    binding["shape"],
                    binding["dtype"],
                )
            )

        # 确保有有效的 batch_size、输入和输出张量
        assert self.batch_size > 0
        assert len(self.inputs) > 0
        assert len(self.outputs) > 0
        assert len(self.allocations) > 0

    def input_spec(self):
        """
        获取模型输入 tensor 的规格（形状和数据类型）
        return: 输入 tensor 的 shape 和 numpy 数据类型
        """
        return self.inputs[0]["shape"], self.inputs[0]["dtype"]

    def make_anchors(self, h, w, grid_cell_offset=0.5):
        """生成 anchor points 坐标，用于检测框预测"""
        sx = np.arange(w).astype(np.float32) + grid_cell_offset  # 生成网格横坐标
        sy = np.arange(h).astype(np.float32) + grid_cell_offset  # 生成网格纵坐标
        sx, sy = np.meshgrid(sx, sy)  # 生成网格点
        anchor_point = np.stack((sx, sy), -1).reshape((1, h, w, -1))  # 生成 anchor 点
        return anchor_point

    def analysis_output(self, outputs, reg_max=16, conf_thresh=0.5, stride=None):
        """解析模型的输出，基于置信度筛选出有效目标"""
        if stride is None:
            stride = [8, 16, 32]
        all_output_grid_feature = []  # 用于存储所有特征层的 grid 信息
        for i in range(len(outputs)):
            c, h, w = outputs[i][1].shape
            target_grid_index = np.where(outputs[i][1] > logit(conf_thresh))  # 根据置信度阈值，筛选有效的 grid 索引
            slices = (slice(None), target_grid_index[1], target_grid_index[2])  # 获取目标位置的索引
            target_grid = outputs[i][0][slices].transpose()  # 获取box特征
            if target_grid.size > 0:
                anchor_points = self.make_anchors(h, w)  # 生成anchor点
                target_grid_anchor = anchor_points[(slice(None), target_grid_index[1], target_grid_index[2])]
                dbox_rest = dfl_rest(target_grid, reg_max)  # 解析box特征
                bbox_rest = dist2bbox(dbox_rest, target_grid_anchor, False) * stride[i]  # 还原检测框
                confs = sigmoid(outputs[i][1][slices].transpose())  # 解析置信度
                confs = confs.reshape(1, *confs.shape)
                output_grid_feature = np.concatenate((confs, bbox_rest), axis=2)  # 拼接输出
                all_output_grid_feature.append(output_grid_feature)
        return np.concatenate(all_output_grid_feature, axis=0) if len(all_output_grid_feature) > 0 else None

    def post_precessing(self, dets, r, left, top, iou_thresh=0.45, nc=1):
        """对检测结果进行 NMS 和 box 还原"""
        classes = np.argmax(dets[:, :nc], axis=1).reshape(dets.shape[0], -1)  # 获取检测类别
        scores = np.max(dets[:, :nc], axis=1).reshape(dets.shape[0], -1)  # 获取置信度分数
        boxes = dets[:, nc:nc + 4]  # 获取检测框的坐标
        out = np.concatenate((boxes, scores, classes), axis=1)  # 将检测框、分数和类别拼接成最终输出
        reserve_ = my_nms(out, iou_thresh)  # 通过 NMS 过滤重叠框
        output = out[reserve_]  # 获取经过 NMS 过滤的检测结果
        output = restore_box(output, r, left, top)  # 还原检测框到原图尺寸
        return output

    def infer(self, batch):
        """
        执行批量图像推理
        param batch: numpy 数组形式的图像批量数据
        return: 推理结果输出的 numpy 数组列表
        """
        memcpy_host_to_device(self.inputs[0]["allocation"], batch)  # 将输入数据复制到 GPU 内存

        # 执行推理
        t0 = time.time()
        self.context.execute_v2(self.allocations)
        print(f'Inference time: {(time.time() - t0):.3f}s')

        # 将输出数据从 GPU 内存复制到 CPU 内存
        for o in range(len(self.outputs)):
            memcpy_device_to_host(
                self.outputs[o]["host_allocation"], self.outputs[o]["allocation"]
            )

        return [o["host_allocation"] for o in self.outputs]

    def process(self, batch, scales=None, nc=80, nms_threshold=0.5, conf_thresh=0.5):
        """
            执行 batch 推理。图片应当已经进行 batch 打包与前处理。
            param batch: batch 打包后的图片组(ndarray).
            param scales: 图片经过letterbox后的压缩比 r 与原图在新图上的左上角坐标 left top
            param nc: 模型识别的类别总数
            param nms_threshold: NMS 时的 IoU 阈值
            return: list, 由每个 batch 的框的 boxes, scores, classes 组成
        """
        # 进行Batch推理
        ori_outputs = self.infer(batch)

        # 分别获取3层的总Batch输出
        layer1_output = [ori_outputs[0], ori_outputs[1]]
        layer2_output = [ori_outputs[2], ori_outputs[3]]
        layer3_output = [ori_outputs[4], ori_outputs[5]]
        outputs = [layer1_output, layer2_output, layer3_output]

        # Batch输出结果解析与后处理
        t0 = time.time()
        final_outputs = []
        # for i in range(self.batch_size):
        for i in range(len(scales)):
            # 获取当前Batch的模型输出
            batch_outputs = [[ori_outputs[0][i], ori_outputs[1][i]], [ori_outputs[2][i], ori_outputs[3][i]],
                             [ori_outputs[4][i], ori_outputs[5][i]]]
            # 首先根据置信度进行过滤，并将DFL输出解析为box坐标
            one_batch_output = self.analysis_output(outputs=batch_outputs, conf_thresh=conf_thresh)
            # NMS 并还原框至原始图像尺寸
            one_batch_boxs = self.post_precessing(np.squeeze(one_batch_output), r=scales[i][0], left=scales[i][1],
                                                  top=scales[i][2],
                                                  iou_thresh=nms_threshold, nc=nc)
            final_outputs.append(one_batch_boxs)
        print(f"Post processing time: {(time.time() - t0):.4f}s")
        return final_outputs

    def cleanup(self):
        # 释放 GPU 内存
        for i in range(len(self.inputs)):
            cuda_call(cudart.cudaFree(self.inputs[i]["allocation"]))
        for o in range(len(self.outputs)):
            cuda_call(cudart.cudaFree(self.outputs[o]["allocation"]))


def main(args):
    if args.output:
        output_dir = os.path.realpath(args.output)
        os.makedirs(output_dir, exist_ok=True)

    labels = []
    if args.labels:
        with open(args.labels) as f:
            for i, label in enumerate(f):
                labels.append(label.strip())
    print("Building Engine")
    trt_infer = TensorRTInfer(args.engine)
    print("Build Engine successfully")

    # 如果给定输入，则进行推理
    if args.input:
        print(f"\nInferring data in {args.input}\n")

        # Batch图像前处理
        batcher = ImageBatcher(args.input, *trt_infer.input_spec())
        for batch, images, scales in batcher.get_batch():
            print(f"Processing Image {batcher.image_index} / {batcher.num_images}")

            # 推理并进行后处理
            detections = trt_infer.process(batch, scales, args.number_of_classes, args.nms_threshold, args.conf_threshold)

            # 画图
            for i in range(len(images)):
                one_batch_box = detections[i]
                draw_img = draw_box_and_save(images[i], one_batch_box, labels, args.output)  # 画图并保存至args.output，返回带框图片
            print("\n")
        # 推理完成后清理GPU内存
        trt_infer.cleanup()
        print("Memory cleanup finished")

    # 未给定输入，执行推理测试
    else:
        print("No input provided, running in benchmark mode")
        spec = trt_infer.input_spec()
        batch = 255 * np.random.rand(*spec[0]).astype(spec[1])
        iterations = 200
        times = []
        for i in range(20):  # GPU warmup iterations
            trt_infer.infer(batch)
        for i in range(iterations):
            start = time.time()
            trt_infer.infer(batch)
            times.append(time.time() - start)
            print("Iteration {} / {}".format(i + 1, iterations), end="\r")
        print("Benchmark results include time for H2D and D2H memory copies")
        print("Average Latency: {:.3f} ms".format(1000 * np.average(times)))
        print(
            "Average Throughput: {:.1f} ips".format(
                trt_infer.batch_size / np.average(times)
            )
        )
    print("Finished Processing")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e",
        "--engine",
        default='weights/yolov8n.engine',
        help="The serialized TensorRT engine",
    )
    parser.add_argument(
        "-i", "--input", default='images/in', help="Path to the image or directory to process"
    )
    parser.add_argument(
        "-o",
        "--output",
        default='images/out',
        help="Directory where to save the visualization results",
    )
    parser.add_argument(
        "-l",
        "--labels",
        default="configs/labels_coco.txt",
        help="File to use for reading the class labels from, default: ./labels_coco.txt",
    )
    parser.add_argument(
        "-nc",
        "--number_of_classes",
        type=int,
        default=80,
        help="Override the number of classes to use, default: 80",
    )
    parser.add_argument(
        "-nms",
        "--nms_threshold",
        type=float,
        default=0.5,
        help="Override the score threshold for the NMS operation, if higher than the built-in threshold",
    )
    parser.add_argument(
        "-conf",
        "--conf_threshold",
        type=float,
        default=0.5,
        help="Override the confidence score threshold, if higher than the built-in threshold",
    )
    args = parser.parse_args()
    main(args)