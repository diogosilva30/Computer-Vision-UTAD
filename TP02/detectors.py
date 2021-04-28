import numpy as np
import cv2
from abc import ABC, abstractmethod


class BaseDetector(ABC):
    @abstractmethod
    def detect(self, frame: np.ndarray) -> list:
        """
        Abstract method that concrete dectetors should implement
        to implement objects in a frame
        """


class HaarCascadeDectector(BaseDetector):
    pass


class DeepLearningDetector(BaseDetector):
    def __init__(
        self,
        proto_path: str,
        model_path: str,
        detection_treshold: float,
        detection_classes: list,
    ):

        # Instantiate a Neural Net from the weights and config paths

        self.net = cv2.dnn.readNetFromCaffe(prototxt=proto_path, caffeModel=model_path)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        # self.net.setInputSize(320, 320)
        # self.net.setInputScale(1.0 / 127.5)
        #  self.net.setInputMean((127.5, 127.5, 127.5))
        # self.net.setInputSwapRB(True)

        self.class_names = [
            "background",
            "aeroplane",
            "bicycle",
            "bird",
            "boat",
            "bottle",
            "bus",
            "car",
            "cat",
            "chair",
            "cow",
            "diningtable",
            "dog",
            "horse",
            "motorbike",
            "person",
            "pottedplant",
            "sheep",
            "sofa",
            "train",
            "tvmonitor",
        ]

        # Save the minimum detection treshold
        self.detection_treshold = detection_treshold
        # Save the classes that should be detected
        # Use a set for optimum look-up performance [ O(1) instead of O(n) ]
        # Normalize each element to upper string
        self.detection_classes = detection_classes

    def detect(self, frame: np.ndarray) -> list:

        (H, W) = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)

        self.net.setInput(blob)
        detections = self.net.forward()

        rects = []
        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self.detection_treshold:
                idx = int(detections[0, 0, i, 1])

                if not self.class_names[idx] in self.detection_classes:
                    continue

                box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                rects.append(box)

        boundingboxes = np.array(rects)
        boundingboxes = boundingboxes.astype(int)

        return boundingboxes
        # # Make detections on frame
        # class_ids, confidences, bounding_boxes = self.net.detect(
        #     frame, confThreshold=self.detection_treshold
        # )

        # # Check if any detection was made
        # if len(class_ids) > 0:
        #     # Start an empty list for the detections
        #     detections = []
        #     for class_id, confidence, bounding_box in zip(
        #         class_ids.flatten(), confidences.flatten(), bounding_boxes
        #     ):

        #         # Get object name
        #         object_name = self.class_names[class_id - 1].upper()
        #         # If object should be detected add it to detections list
        #         if object_name in self.detection_classes:
        #             detections.append((object_name, confidence, bounding_box))

        #     return detections
        # # Otherwise return None
        # return None
