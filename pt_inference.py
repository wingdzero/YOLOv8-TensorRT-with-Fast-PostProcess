from ultralytics import YOLO


model = YOLO("/home/yyj/Code/ultralystic/yolov8n.pt")

model.predict('images/in/bus.jpg', save=True, imgsz=640, conf=0.3, show_labels=False, show_conf=False, line_width=3, max_det=50)
