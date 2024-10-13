from ultralytics import YOLO

# Load a model
model = YOLO("weights/yolov8n.pt")

# Export the model
model.export(format="onnx", imgsz=640, opset=17, batch=1)