from ultralytics import YOLO

# Load a model
model = YOLO("../weights/yolov8n.pt")  # load an official model

# Export the model
model.export(format="onnx", dynamic=True, opset=17)