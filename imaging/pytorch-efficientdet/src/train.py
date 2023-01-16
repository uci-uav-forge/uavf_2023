
from multiprocessing import freeze_support
from pathlib import Path

import torch
from data_utils import CarsDatasetAdaptor, EfficientDetDataModule



if __name__ == '__main__':
    freeze_support()# crashes without this for multi gpu train
    dataset_path = Path('../data-gen/output')
    import pandas as pd

    df = pd.read_csv(dataset_path/'annotations.csv')
    train_data_path = dataset_path/'train'
    cars_train_ds = CarsDatasetAdaptor(train_data_path, df)

    dm = EfficientDetDataModule(train_dataset_adaptor=cars_train_ds, 
            validation_dataset_adaptor=cars_train_ds,
            num_workers=4,
            batch_size=2)

    from model import EfficientDetModel

    model = EfficientDetModel(
        num_classes=13,
        img_size=512,
        model_architecture="tf_efficientnetv2_b0"
        )
    from pytorch_lightning import Trainer
    trainer = Trainer(
            gpus=[0,1], max_epochs=5, num_sanity_val_steps=1,
        )

    trainer.fit(model, dm)
    torch.save(model.state_dict(), 'trained_effdet_custom')
    print("done")