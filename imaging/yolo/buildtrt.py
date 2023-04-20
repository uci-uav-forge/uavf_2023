from ultralytics import YOLO


#model_file = input("input .pt file name from trained models directory to build with tensorrt: ")

#model = YOLO('yolov8m-seg.yaml')
model = YOLO('./trained_models/seg-v8n.pt')
#model.to('cuda')
model.export(verbose=True, opset=17, simplify=True, workspace=6, int8=False, batch=1, half=True, optimize=False, nms=True, dynamic=False, device='0', format="engine")
