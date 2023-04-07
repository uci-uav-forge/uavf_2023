from ultralytics import YOLO

task='segmentation'
if task=='segmentation':
    model = YOLO('yolov8n-seg.yaml')
elif task=='detection':
    model = YOLO('yolov8n.yaml')
    
model.train(data='forge10k.yaml', epochs=100)