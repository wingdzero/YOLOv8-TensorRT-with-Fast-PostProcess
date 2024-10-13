import os
import cv2
import numpy as np

from cuda import cuda, cudart


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def logit(x):
    if x <= 0 or x >= 1:
        raise ValueError("Input value must be in the range (0, 1)")
    return np.log(x / (1 - x))


def softmax(x, axis=None):
    # 减去最大值以提高数值稳定性
    x_max = np.max(x, axis=axis, keepdims=True)
    x_exp = np.exp(x - x_max)
    x_sum = np.sum(x_exp, axis=axis, keepdims=True)
    return x_exp / x_sum


def dfl(x, reg_max):
    b, h, w, _ = x.shape
    x = x.reshape((b, h, w, 4, reg_max)).transpose(0, 1, 2, 4, 3)
    x = softmax(x, 3)
    dfl_conv = np.tile(np.arange(reg_max).reshape(1, 1, 1, 16, 1), (1, h, w, 1, 4))
    dbox = np.sum(x * dfl_conv, axis=3).reshape(1, h, w, 4)
    return dbox


def dfl_rest(x, reg_max):
    n, _ = x.shape
    x = x.reshape((n, -1, reg_max)).transpose(0, 2, 1)
    x = softmax(x, 1)
    dfl_conv = np.tile(np.arange(reg_max).reshape(1, 16, 1), (n, 1, 4))
    dbox = np.sum(x*dfl_conv, axis=-2)
    return dbox


def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    """Transform distance(ltrb) to box(xywh or xyxy)."""
    lt, rb = np.split(distance, 2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return np.concatenate((c_xy, wh), dim)  # xywh bbox
    return np.concatenate((x1y1, x2y2), dim)  # xyxy bbox



def my_nms(boxes, iou_thresh):  # nms
    index = np.argsort(boxes[:, 4])[::-1]
    keep = []
    while index.size > 0:
        i = index[0]
        keep.append(i)
        x1 = np.maximum(boxes[i, 0], boxes[index[1:], 0])
        y1 = np.maximum(boxes[i, 1], boxes[index[1:], 1])
        x2 = np.minimum(boxes[i, 2], boxes[index[1:], 2])
        y2 = np.minimum(boxes[i, 3], boxes[index[1:], 3])

        w = np.maximum(0, x2 - x1)
        h = np.maximum(0, y2 - y1)

        inter_area = w * h
        union_area = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1]) + (
                boxes[index[1:], 2] - boxes[index[1:], 0]) * (boxes[index[1:], 3] - boxes[index[1:], 1])
        iou = inter_area / (union_area - inter_area)
        idx = np.where(iou <= iou_thresh)[0]
        index = index[idx + 1]
    return keep

def restore_box(boxes, r, left, top):  # 返回原图上面的坐标
    boxes[:, [0, 2]] -= left
    boxes[:, [1, 3]] -= top

    boxes[:, [0, 2]] /= r
    boxes[:, [1, 3]] /= r
    return boxes

def draw_box_and_save(img_path, one_batch_box, labels, output):  # 将框画到图上并保存
    draw_img = cv2.imread(img_path)
    for j in range(one_batch_box.shape[0]):
        one_box = one_batch_box[j]
        draw_img = cv2.rectangle(draw_img, (int(one_box[0]), int(one_box[1])), (int(one_box[2]), int(one_box[3])),
                                 (0, 0, 255), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX  # 字体类型
        fontScale = 1  # 字体大小
        color = (255, 255, 255)  # 白色
        thickness = 2  # 字体粗细
        lineType = cv2.LINE_AA  # 抗锯齿线条

        # 在图像上绘制文本
        cls_index = int(one_box[5])
        cls_name = labels[cls_index]
        cv2.putText(draw_img, f"{cls_name} {one_box[4]:.2f}", (int(one_box[0]), int(one_box[1]) - 10 * fontScale),
                    font, fontScale, color, thickness, lineType)
    file_name, file_ext = os.path.splitext(os.path.basename(img_path))
    img_save_path = os.path.join(output, f"{file_name}_result{file_ext}")
    cv2.imwrite(img_save_path, draw_img)
    return draw_img

def stream_draw_box_and_save(draw_img, one_batch_box, labels, output, img_name):  # 将框画到图上并保存
    for j in range(one_batch_box.shape[0]):
        one_box = one_batch_box[j]
        draw_img = cv2.rectangle(draw_img, (int(one_box[0]), int(one_box[1])), (int(one_box[2]), int(one_box[3])),
                                 (0, 0, 255), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX  # 字体类型
        fontScale = 1  # 字体大小
        color = (255, 255, 255)  # 白色
        thickness = 2  # 字体粗细
        lineType = cv2.LINE_AA  # 抗锯齿线条

        # 在图像上绘制文本
        cls_index = int(one_box[5])
        cls_name = labels[cls_index]
        cv2.putText(draw_img, f"{cls_name} {one_box[4]:.2f}", (int(one_box[0]), int(one_box[1]) - 10 * fontScale),
                    font, fontScale, color, thickness, lineType)
    # file_name, file_ext = os.path.splitext(os.path.basename(img_path))
    img_save_path = os.path.join(output, f"{img_name}_result.jpg")
    cv2.imwrite(img_save_path, draw_img)
    return draw_img

def check_cuda_err(err):
    # 检查是否为 CUDA Driver API 的错误类型 CUresult
    if isinstance(err, cuda.CUresult):
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError("Cuda Error: {}".format(err))
    # 检查是否为 CUDA Runtime API 的错误类型 cudaError_t
    if isinstance(err, cudart.cudaError_t):
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError("Cuda Runtime Error: {}".format(err))
    # 未知错误
    else:
        raise RuntimeError("Unknown error type: {}".format(err))

def cuda_call(call):
    # 调用 CUDA 函数，并检查其返回的错误码，确保调用成功
    err, res = call[0], call[1:]
    check_cuda_err(err)
    if len(res) == 1:
        res = res[0]
    return res

# 将数据从 CPU 内存复制到 GPU 内存
def memcpy_host_to_device(device_ptr: int, host_arr: np.ndarray):
    nbytes = host_arr.size * host_arr.itemsize
    cuda_call(cudart.cudaMemcpy(device_ptr, host_arr, nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice))

# 将数据从 GPU 内存复制到 CPU 内存
def memcpy_device_to_host(host_arr: np.ndarray, device_ptr: int):
    nbytes = host_arr.size * host_arr.itemsize
    cuda_call(cudart.cudaMemcpy(host_arr, device_ptr, nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost))

# 递归求list的维度
def get_shape(lst):
    if isinstance(lst, list) and lst:
        return (len(lst),) + get_shape(lst[0])
    return ()