
from multiprocessing import freeze_support
from pathlib import Path

import numpy as np
import torchmetrics.detection.mean_ap as mean_ap
import torch
import torch.onnx
from data_utils import CarsDatasetAdaptor, EfficientDetDataModule
import pandas as pd
from model import EfficientDetModel
from pytorch_lightning import Trainer
from pytorch_lightning.plugins import DDPPlugin
from pprint import pprint

if __name__ == '__main__':
    freeze_support()# crashes without this for multi gpu train
    dataset_path = Path('../data-gen/output')

    test_annotations = pd.read_csv(dataset_path/'testannotations.csv')
    test_data_path = dataset_path/'test'
    test_ds = CarsDatasetAdaptor(test_data_path, test_annotations)

    num_samples = 20#len(test_ds.images)

    img_labels_list = [test_ds.get_image_and_labels_by_idx(i) for i in range(num_samples)]

    model = EfficientDetModel(
        num_classes=13,
        img_size=512,
        model_architecture="efficientnet_b0" # this is the name of the backbone. For some reason it doesn't work with the corresponding efficientdet name.
        )
    model.load_state_dict(torch.load('efficientdet_b0_pytorch_50epoch'))
    model.eval()

    images = [x[0] for x in img_labels_list]
    actual_labels = [x[2].values for x in img_labels_list]
    actual_boxes = [x[1] for x in img_labels_list]
    predicted_bboxes, predicted_class_labels, predicted_class_confidences  = model.predict(images)
    metric  = mean_ap.MeanAveragePrecision()
    metric.update(
        preds=[
            {
            "labels":torch.Tensor(labels),
            "scores":torch.Tensor(scores),
            "boxes": torch.Tensor(boxes)
            } 
            for labels, scores, boxes in zip(predicted_class_labels, predicted_class_confidences, predicted_bboxes)
        ],
        target=[
            {
            "labels":torch.Tensor(labels),
            "boxes": torch.Tensor(boxes)
            } 
            for labels, boxes in zip(actual_labels, actual_boxes)
        ],
    )
    pprint(metric.compute())
