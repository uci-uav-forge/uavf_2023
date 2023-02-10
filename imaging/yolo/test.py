from ultralytics import YOLO
from ultralytics.yolo.engine.results import Results

from pathlib import Path
import math

import torchmetrics.detection.mean_ap as mean_ap
import torch
from shape_detection.src.data_utils import ShapeDatasetAdaptor
import pandas as pd
from pprint import pprint
from time import perf_counter

def main():
    metric  = mean_ap.MeanAveragePrecision(class_metrics=True)

    dataset_path = Path('shape_detection/data-gen/images')

    test_annotations = pd.read_csv(dataset_path/'testannotations.csv')
    test_data_path = dataset_path/'test'
    test_ds = ShapeDatasetAdaptor(test_data_path, test_annotations)

    num_samples = len(test_ds.images)
    batch_size = 4
    num_batches = math.ceil(num_samples/batch_size)
    model = YOLO("yolo/trained_models/v8n.pt")
    start_time = perf_counter()
    for batch_idx in range(num_batches):
        img_labels_list = [test_ds.get_image_and_labels_by_idx(i) for i in range(batch_idx*batch_size,min(num_samples,(batch_idx+1)*batch_size))]

        images = ([x[0] for x in img_labels_list])
        actual_labels = [x[2].values for x in img_labels_list]
        actual_boxes = [x[1] for x in img_labels_list]
        predictions: list[Results] = model.predict(images, verbose=False) # the results have 6-element tensors which are x1,y1,x2,y2,confidence, class label
        metric.update(
            preds=[
                {
                "labels":prediction.boxes.boxes[:,5]+1,
                "scores":prediction.boxes.boxes[:,4],
                "boxes": prediction.boxes.boxes[:,:4]
                } 
                for prediction in [x.to('cpu') for x in predictions]
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