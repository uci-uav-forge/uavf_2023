from ultralytics import YOLO

task='segmentation'
if task=='segmentation':
    model = YOLO('yolov8n-seg.yaml')

    model.train(data='forge10k-seg.yaml', epochs=100)
elif task=='detection':
    model = YOLO('yolov8n.yaml')

    model.train(data='forge10k.yml', epochs=100)