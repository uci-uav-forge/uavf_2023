from ultralytics import YOLO

model = YOLO('yolov8n-cls.yaml')
model.train(data='letter16', epochs=50, imgsz=128)