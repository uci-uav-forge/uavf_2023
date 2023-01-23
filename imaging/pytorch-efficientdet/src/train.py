
from multiprocessing import freeze_support
from pathlib import Path

import numpy as np
import torch
import torch.onnx
from data_utils import CarsDatasetAdaptor, EfficientDetDataModule
import pandas as pd
from model import EfficientDetModel
from pytorch_lightning import Trainer


if __name__ == '__main__':
    freeze_support()# crashes without this for multi gpu train
    dataset_path = Path('../data-gen/output')

    df = pd.read_csv(dataset_path/'annotations.csv')
    train_data_path = dataset_path/'train'
    cars_train_ds = CarsDatasetAdaptor(train_data_path, df)

    dm = EfficientDetDataModule(train_dataset_adaptor=cars_train_ds, 
            validation_dataset_adaptor=cars_train_ds,
            num_workers=4,
            batch_size=2)


    model = EfficientDetModel(
        num_classes=13,
        img_size=512,
        model_architecture="efficientnet_b0" # this is the name of the backbone. For some reason it doesn't work with the corresponding efficientdet name.
        )
    trainer = Trainer(
            gpus=[0,1], max_epochs=20, num_sanity_val_steps=1, auto_scale_batch_size=True
        )

    trainer.fit(model, dm)
    # dummy_input = np.zeros(shape=model)
    # torch.onnx.export(model)
    torch.save(model.state_dict(), 'efficientdet_b0_pytorch_20epoch')
    print("done")