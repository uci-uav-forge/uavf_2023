
from multiprocessing import freeze_support
from pathlib import Path

import torch
import torch.onnx
from .data_utils import ShapeDatasetAdaptor, EfficientDetDataModule
import pandas as pd
from shape_det_model import EfficientDetModel
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelPruning, QuantizationAwareTraining
if __name__ == '__main__':
    freeze_support()# crashes without this for multi gpu train
    dataset_path = Path('data-gen/output')

    train_annotations = pd.read_csv(dataset_path/'trainannotations.csv')
    train_data_path = dataset_path/'train'
    val_annotations = pd.read_csv(dataset_path/'validationannotations.csv')
    val_data_path = dataset_path/'validation'
    train_ds = ShapeDatasetAdaptor(train_data_path, train_annotations)
    val_ds = ShapeDatasetAdaptor(val_data_path, val_annotations)

    dm = EfficientDetDataModule(train_dataset_adaptor=train_ds, 
            validation_dataset_adaptor=val_ds,
            num_workers=4,
            batch_size=4)

    backbone_name = "efficientdet_d0"
    model = EfficientDetModel(
        num_classes=13,
        img_size=512,
        model_architecture=backbone_name # this is the name of the backbone. For some reason it doesn't work with the corresponding efficientdet name.
        )

    logger = TensorBoardLogger("tb_logs", name="UAV Forge Shape Detection Model")
    num_epochs = 25
    trainer = Trainer(
            logger=logger,
            strategy="ddp_find_unused_parameters_false",
            gpus=[0], max_epochs=num_epochs, num_sanity_val_steps=1,
            # callbacks = [ModelPruning("l1_unstructured", amount=0.5)]
        )
    # to upload logs: tensorboard dev upload --logdir tb_logs
    trainer.fit(model, dm)
    torch.save(model.state_dict(), f'{backbone_name}_{num_epochs}epochs.pt')
