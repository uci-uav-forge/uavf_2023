from pathlib import Path
import math

import torchmetrics.detection.mean_ap as mean_ap
import torch
import torch.onnx
from data_utils import CarsDatasetAdaptor
import pandas as pd
from model import EfficientDetModel
from pprint import pprint

if __name__ == '__main__':
    metric  = mean_ap.MeanAveragePrecision(class_metrics=True)

    dataset_path = Path('../data-gen/output')

    test_annotations = pd.read_csv(dataset_path/'testannotations.csv')
    test_data_path = dataset_path/'test'
    test_ds = CarsDatasetAdaptor(test_data_path, test_annotations)

    num_samples = len(test_ds.images)
    batch_size = 30

    for batch_idx in range(math.ceil(num_samples/batch_size)):
        img_labels_list = [test_ds.get_image_and_labels_by_idx(i) for i in range(batch_idx*batch_size,min(num_samples,(batch_idx+1)*batch_size))]

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
