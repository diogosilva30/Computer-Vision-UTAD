"""
Module containing utility methods
"""
import numpy as np


def non_max_suppression(bounding_boxes, treshold):
    """
    Performs non max suppresion on a list of bounding boxes.
    Any bouding boxes overlapping by a certain treshold are merged.

    Parameters
    ----------
    bounding_boxes: list
        List of bounding boxes
    treshold: float
        The treshold to merge overlapping bounding boxes.
    """
    try:
        if len(bounding_boxes) == 0:
            return []

        if bounding_boxes.dtype.kind == "i":
            boxes = bounding_boxes.astype("float")

        pick = []

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)

        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            overlap = (w * h) / area[idxs[:last]]

            idxs = np.delete(
                idxs, np.concatenate(([last], np.where(overlap > treshold)[0]))
            )

        return boxes[pick].astype("int")
    except Exception as e:
        print("Exception occurred in non_max_suppression : {}".format(e))
