from fastapi import FastAPI, UploadFile, File
from typing import List

import configparser

from engine_inference import TensorRTInfer
from utils.stream_image_batch import StreamImageBatcher
from utils.utils import *

app = FastAPI()

config = configparser.ConfigParser()

config.read('config.ini')

label_path = config.get('Paths', 'label_path')
output_path = config.get('Paths', 'output_path')

labels = []
with open(label_path) as f:
    for i, label in enumerate(f):
        labels.append(label.strip())

# 全局初始化TensorRT引擎，使其只加载一次
trt_infer = None

@app.on_event("startup")
async def startup_event():
    global trt_infer
    engine_path = config.get('Paths', 'model_path')
    # engine_path = './yolov8n.engine'
    trt_infer = TensorRTInfer(engine_path)
    print("TensorRT 引擎已加载！\n")


@app.post("/upload/")
async def run_inference(files: List[UploadFile] = File(...)):

    print('\n-----------------------------------------Start Inference-----------------------------------------\n')

    # 将上传的图像加载到批次中以进行推理
    images = []
    image_names = []
    for file in files:
        filename = file.filename
        print(f"Read {filename} successfully")
        image_names.append(filename)
        contents = await file.read()
        np_img = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        images.append(img)

    print(f'\nGet {len(images)} image, Processing')

    batcher = StreamImageBatcher(images, *trt_infer.input_spec())

    results = []

    batch_index = 0
    for batch, images, scales in batcher.get_batch():

        print(f"\nInferencing batch {batch_index}: {image_names[batch_index: batch_index + batcher.batch_size]}")

        # 推理并进行后处理
        detections = trt_infer.process(batch, scales, config.getint('Settings', 'model_cls_num'), config.getfloat('Settings', 'iou_threshold'), config.getfloat('Settings', 'conf_threshold'))

        for i in range(len(images)):
            img_name = image_names[batch_index + i]
            one_batch_box = detections[i]
            draw_img = stream_draw_box_and_save(images[i], one_batch_box, labels, output_path, img_name)
        results.append(detections[0].tolist())
        batch_index += 1

    print('\n-----------------------------------------Done-----------------------------------------\n')
    # 后处理，返回JSON格式的结果
    return {"results": results}


@app.on_event("shutdown")
async def shutdown_event():
    trt_infer.cleanup()
    print("TensorRT 引擎清理完成！")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8010)