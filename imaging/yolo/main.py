from ultralytics import YOLO # pip install ultralytics (tested with python 3.8)
# https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data
# Load a model
# model = YOLO("yolov8n.yaml")  # build a new model from scratch
model = YOLO("yolov5n.pt")  # load a pretrained model (recommended for training)

# Train the model
model.train(data ="forge10k.yaml", epochs=100, imgsz=512)
model.export("yolov5n_forge10k.pt")