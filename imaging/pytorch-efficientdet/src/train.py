
from multiprocessing import freeze_support
from pathlib import Path

import torch
import torch.onnx
from data_utils import CarsDatasetAdaptor, EfficientDetDataModule
import pandas as pd
from model import EfficientDetModel
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

if __name__ == '__main__':
    freeze_support()# crashes without this for multi gpu train
    dataset_path = Path('../data-gen/output')

    df = pd.read_csv(dataset_path/'trainannotations.csv')
    train_data_path = dataset_path/'train'
    cars_train_ds = CarsDatasetAdaptor(train_data_path, df)

    dm = EfficientDetDataModule(train_dataset_adaptor=cars_train_ds, 
            validation_dataset_adaptor=cars_train_ds,
            num_workers=4,
            batch_size=4)

    backbone_name = "efficientnet_b0"
    model = EfficientDetModel(
        num_classes=13,
        img_size=512,
        model_architecture=backbone_name # this is the name of the backbone. For some reason it doesn't work with the corresponding efficientdet name.
        )

    logger = TensorBoardLogger("tb_logs", name="UAV Forge Shape Detection")
    num_epochs = 5
    trainer = Trainer(
            logger=logger,
            strategy="ddp_find_unused_parameters_false",
            gpus=[0,1], max_epochs=num_epochs, num_sanity_val_steps=1
        )
    # to upload logs: tensorboard dev upload --logdir tb_logs
    trainer.fit(model, dm)
    torch.save(model.state_dict(), f'{backbone_name}_pytorch_{num_epochs}epoch.pt')
