import matplotlib.pyplot as plt
from matplotlib import patches

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
            label=labels_dict[int(label)]
        if confidences is not None:
            label=f"{label} ({confidences[i]:.1%})"

        plot_ax.text(x=bottom_left[0]+width//2, 
                     y=bottom_left[1]+height-10,
                    s=label, 
                    color="white", 
                    backgroundcolor="black",
                    horizontalalignment="center",
                    verticalalignment="center")
        # Add the patch to the Axes
        plot_ax.add_patch(rect_1)
        plot_ax.add_patch(rect_2)

def show_image(
    image, bboxes=None, labels=None, confidences=None,draw_bboxes_fn=draw_pascal_voc_bboxes, figsize=(10, 10), labels_dict=None
):
    fig, ax = plt.subplots(1, figsize=figsize)
    ax.imshow(image)

    if bboxes is not None:
        draw_bboxes_fn(ax, bboxes, labels, confidences=confidences,labels_dict=labels_dict)

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