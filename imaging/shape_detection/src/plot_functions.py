import matplotlib.pyplot as plt
from matplotlib import patches
import cv2 as cv

from imaging.colordetect.color_segment import ColorSegmentationResult


def get_rectangle_edges_from_pascal_bbox(bbox):
    xmin_top_left, ymin_top_left, xmax_bottom_right, ymax_bottom_right = bbox

    bottom_left = (xmin_top_left, ymax_bottom_right)
    width = xmax_bottom_right - xmin_top_left
    height = ymin_top_left - ymax_bottom_right

    return bottom_left, width, height


def draw_pascal_voc_bboxes(
        plot_ax: plt,
        bboxes,
        labels,
        confidences=None,
        get_rectangle_corners_fn=get_rectangle_edges_from_pascal_bbox,
        labels_dict=None
):
    for bbox, label, i in zip(bboxes, labels, range(len(labels))):
        bottom_left, width, height = get_rectangle_corners_fn(bbox)

        rect_1 = patches.Rectangle(
            bottom_left,
            width,
            height,
            linewidth=4,
            edgecolor="black",
            fill=False,
        )
        rect_2 = patches.Rectangle(
            bottom_left,
            width,
            height,
            linewidth=2,
            edgecolor="white",
            fill=False,
        )
        if labels_dict is not None:
            label = labels_dict[int(label)]
        if confidences is not None:
            label = f"{label} ({confidences[i]:.1%})"

        plot_ax.text(x=bottom_left[0] + width // 2,
                     y=bottom_left[1] + height - 10,
                     s=label,
                     color="white",
                     backgroundcolor="black",
                     horizontalalignment="center",
                     verticalalignment="center")
        # Add the patch to the Axes
        plot_ax.add_patch(rect_1)
        plot_ax.add_patch(rect_2)


def show_image(
        image, bboxes=None, labels=None, confidences=None, draw_bboxes_fn=draw_pascal_voc_bboxes, figsize=(10, 10),
        labels_dict=None, file_name=None
):
    fig, ax = plt.subplots(1, figsize=figsize)
    ax.imshow(image)

    if bboxes is not None:
        draw_bboxes_fn(ax, bboxes, labels, confidences=confidences, labels_dict=labels_dict)
    if file_name is not None:
        plt.savefig(file_name)
    else:
        plt.show()


def show_image_cv(
        image: cv.Mat, bboxes=None, labels=None, confidences=None, file_name=None, font_scale=2,
        thickness=1, box_color=(255, 255, 255), text_color=(255, 255, 255),
        color_results: "list[ColorSegmentationResult]" = None
):
    for bbox, label, color_res in zip(bboxes, labels,color_results):
        x0, y0, x1, y1 = map(int, bbox)
        image = cv.rectangle(image, (x0, y0), (x1, y1), color=box_color, thickness=thickness)
        cv.putText(image, label, (x0, y0), cv.FONT_HERSHEY_PLAIN, font_scale, text_color,thickness)
        
        cv.putText(image, "Shape", (max(0,x0-40), y1+20), cv.FONT_HERSHEY_PLAIN, font_scale, text_color,thickness, bottomLeftOrigin=False)
        cv.putText(image, "Letter", (x1, y1+20), cv.FONT_HERSHEY_PLAIN, font_scale, text_color,thickness, bottomLeftOrigin=False)
        
        cv.circle(image, (x0,y1), 5, color_res.shape_color.tolist(), -1)
        cv.circle(image, (x1,y1), 5, color_res.letter_color.tolist(), -1)
    if file_name is not None:
        cv.imwrite(file_name, image)
    else:
        plt.imshow(image)
        plt.show()


def compare_bboxes_for_image(
        image,
        predicted_bboxes,
        actual_bboxes,
        predicted_labels,
        actual_labels,
        confidences=None,
        draw_bboxes_fn=draw_pascal_voc_bboxes,
        figsize=(20, 20),
        labels_dict=None
):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    ax1.imshow(image)
    ax1.set_title("Prediction")
    ax2.imshow(image)
    ax2.set_title("Actual")

    draw_bboxes_fn(ax1, predicted_bboxes, predicted_labels, confidences, labels_dict=labels_dict)
    draw_bboxes_fn(ax2, actual_bboxes, actual_labels, labels_dict=labels_dict)

    plt.show()
