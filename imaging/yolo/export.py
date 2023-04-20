from ultralytics import YOLO

# Load a model
model = YOLO('trained_models/seg-v8n.pt')  # load a custom trained

# Export the model
model.export(format='engine', device=0)