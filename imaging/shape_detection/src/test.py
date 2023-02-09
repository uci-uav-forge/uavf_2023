from pathlib import Path
import math

import torchmetrics.detection.mean_ap as mean_ap
import torch
# import torch_tensorrt
import torch.quantization
from .data_utils import ShapeDatasetAdaptor
import pandas as pd
from shape_det_model import EfficientDetModel
from pprint import pprint
import numpy as np
from time import perf_counter

def main():
    metric  = mean_ap.MeanAveragePrecision(class_metrics=True)

    dataset_path = Path('data-gen/output')

    test_annotations = pd.read_csv(dataset_path/'testannotations.csv')
    test_data_path = dataset_path/'test'
    test_ds = ShapeDatasetAdaptor(test_data_path, test_annotations)

    num_samples = len(test_ds.images)
    batch_size = 4
    num_batches = math.ceil(num_samples/batch_size)
    model = EfficientDetModel(
        num_classes=13,
        img_size=512,
        model_architecture="efficientnet_b0"
        )
    model_file=f'trained_models/efficientnet_b0_pytorch_25epoch.pt'

    model.load_state_dict(torch.load(model_file))
    # model = torch_tensorrt.compile(model, 
    # inputs = [torch_tensorrt.Input((4,3,512,512), dtype=torch.float32)],
    # enabled_precisions=torch.float32,
    # workspace_size=1<<22)
    model.eval()
    model.to(device="cuda")
    start_time = perf_counter()
    for batch_idx in range(num_batches):
        img_labels_list = [test_ds.get_image_and_labels_by_idx(i) for i in range(batch_idx*batch_size,min(num_samples,(batch_idx+1)*batch_size))]

        images = ([x[0] for x in img_labels_list])
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
        print("\r[{0}{1}] {2} ({3}%)".format("="*int(batch_idx/num_batches*20), " "*(20-int(batch_idx/num_batches*20)), f"Finished {batch_idx}/{num_batches}", int(batch_idx/num_batches*100)), end="")
    end_time=perf_counter()
    print()
    pprint(metric.compute())
    total_time = end_time-start_time
    print(f"Time taken: {total_time:.3f} seconds")
    print(f"Per image: {total_time/num_samples:.3f} seconds")

if __name__ == '__main__':
    main()