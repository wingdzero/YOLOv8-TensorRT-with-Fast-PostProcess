import onnx
import onnxruntime
import numpy as np
import cv2
import time


def make_anchors(h, w, grid_cell_offset=0.5):
    sx = np.arange(w).astype(np.float32) + grid_cell_offset  # shift x
    sy = np.arange(h).astype(np.float32) + grid_cell_offset  # shift y
    sx, sy = np.meshgrid(sx, sy)
    anchor_point = np.stack((sx, sy), -1).reshape((1, h, w, -1))
    # stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return anchor_point


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def logit(y):
    if y <= 0 or y >= 1:
        raise ValueError("Input value must be in the range (0, 1)")
    return np.log(y / (1 - y))


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

def analysis_output(x, reg_max=16, conf_thresh=0.5):
    stride = [8, 16, 32]
    z = []
    for i in range(len(x)):
        b, c, h, w = x[i][1].shape
        yi = np.where(x[i][1] > logit(conf_thresh))  # 根据反向sigmoid的置信度阈值取出符合阈值的n个位置, [4, n]
        # 只使用第 0、2、3 维进行索引
        # 创建一个切片对象，忽略第 1 维度
        slices = (yi[0], slice(None), yi[2], yi[3])
        ybox = x[i][0][slices]  # 取出根据置信度过滤出的位置的box64维特征，[n, 64]
        if ybox.size > 0:
            anchor_point = make_anchors(h, w)
            anchor_point_t = anchor_point[(slice(None), yi[2], yi[3], slice(None))]
            dbox_rest = dfl_rest(ybox, reg_max)
            bbox_rest = dist2bbox(dbox_rest, anchor_point_t, False) * stride[i]
            confs = sigmoid(x[i][1][slices])
            confs = confs.reshape(1, *confs.shape)
            r = np.concatenate((confs, bbox_rest), axis=2)
            z.append(r)
    return np.concatenate(z, axis=0) if len(z) > 0 else None

def nms(boxes, iou_thresh):
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


def letter_box(img, size=(640, 640)):
    h, w, c = img.shape
    r = min(size[0] / h, size[1] / w)  # 取压缩比最大的那边
    new_h, new_w = int(h * r), int(w * r)
    top = int((size[0] - new_h) / 2)
    left = int((size[1] - new_w) / 2)

    bottom = size[0] - new_h - top
    right = size[1] - new_w - left
    img_resize = cv2.resize(img, (new_w, new_h))
    img = cv2.copyMakeBorder(img_resize, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT,
                             value=(114, 114, 114))
    return img, r, left, top


def restore_box(boxes, r, left, top):  # 返回原图上面的坐标
    boxes[:, [0, 2]] -= left
    boxes[:, [1, 3]] -= top

    boxes[:, [0, 2]] /= r
    boxes[:, [1, 3]] /= r
    return boxes


def post_precessing(dets, r, left, top, conf_thresh=0.4, iou_thresh=0.45, nc=1):  # 检测后处理

    dets_ = dets
    classes = np.argmax(dets_[:, :nc], axis=1).reshape(dets_.shape[0], -1)
    scores = np.max(dets_[:, :nc], axis=1).reshape(dets_.shape[0], -1)
    boxes = dets_[:, nc:nc + 4]
    out = np.concatenate((boxes, scores, classes), axis=1)
    reserve_ = nms(out, iou_thresh)
    output = out[reserve_]
    output = restore_box(output, r, left, top)
    return output

def detect_pre_precessing(ori_img, img_size):  # 检测前处理
    img = ori_img.copy()
    img, r, left, top = letter_box(img, img_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # cv2.imwrite("1.jpg",img)
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = (img / 255).astype(np.float32)
    img = img.reshape(1, *img.shape)  # 添加batch
    return img, r, left, top

if __name__ == '__main__':

    img_path = 'images/in/bus.jpg'
    onnx_file_name = 'weights/yolov8n_1b.onnx'
    output_path = 'result.jpg'

    ori_img = cv2.imread(img_path)

    # 加载 ONNX 模型，创建 InferenceSession
    print("Loading ONNX model")
    ort_session = onnxruntime.InferenceSession(onnx_file_name, providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'])
    # 查看当前正在使用的 ExecutionProvider (第一个 provider)
    current_provider = ort_session.get_providers()[0]
    print("Loaded ONNX model, Current Execution Provider:", current_provider)

    # 获取输入信息
    input_info = ort_session.get_inputs()
    input_size = ort_session.get_inputs()[0].shape[2:]

    # 输入预处理（letter box，归一化）
    img, r, left, top = detect_pre_precessing(ori_img, input_size)
    print("Input preprocessing")

    # onnxruntime推理
    ort_inputs = {'images': img}
    t0 = time.time()
    outputs = ort_session.run(None, ort_inputs)
    print(f"Inference time: {(time.time() - t0):.3f}s")

    # Fast PostProcess
    # 1. 解析输出
    output0 = outputs[0:2]
    output1 = outputs[2:4]
    output2 = outputs[4:6]
    outputs = [output0, output1, output2]
    result = analysis_output(x=outputs, conf_thresh=0.5)  # 基于置信度对三层输出进行过滤

    # 2. 输出后处理
    result = post_precessing(np.squeeze(result), r, left, top, nc=80)
    print(f"Detect {len(result)} target")

    # 画图
    for one_box in result:
        ori_img = cv2.rectangle(ori_img, (int(one_box[0]), int(one_box[1])), (int(one_box[2]), int(one_box[3])), (0, 0, 255), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        color = (255, 255, 255)
        thickness = 2
        lineType = cv2.LINE_AA
        cv2.putText(ori_img, f"{int(one_box[5])} {one_box[4]:.3f}", (int(one_box[0]), int(one_box[1]) - 10), font, fontScale, color, thickness, lineType)
    cv2.imwrite(output_path, ori_img)