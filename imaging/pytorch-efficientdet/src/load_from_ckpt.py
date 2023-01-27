import torch
from model import EfficientDetModel

backbone_name = "efficientnet_b0"
checkpoint = torch.load("tb_logs/UAV Forge Shape Detection/version_3/checkpoints/epoch=24-step=23274.ckpt")
model = EfficientDetModel(
    num_classes=13,
    img_size=512,
    model_architecture=backbone_name # this is the name of the backbone. For some reason it doesn't work with the corresponding efficientdet name.
    )
model.load_state_dict(checkpoint["state_dict"])
torch.save(model.state_dict(), f'{backbone_name}_pytorch_{25}epoch.pt')