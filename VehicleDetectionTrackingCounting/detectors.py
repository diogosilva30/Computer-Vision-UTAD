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
    def __init__(self, xml_path: str):
        self.model = cv2.CascadeClassifier(xml_path)

    def detect(self, frame: np.ndarray) -> list:
        # convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect cars with cascade classifier
        detections = self.model.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)

        rects = []

        for (x, y, w, h) in detections:
            rects.append([x, y, x + w, y + h])
            # cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 10)

        return np.array(rects)


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

    def detect(self, frame: np.ndarray) -> np.ndarray:

        """
        Perform detection using a Deep Learning Neural Network.

        Parameters
        ----------
        frame: np.ndarray
            A frame (image) to perform object detection.

        Returns
        -------
        list
        """
        (H, W) = frame.shape[:2]

        # Make a blob from image
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)

        # Use blob as input to the neural network
        self.net.setInput(blob)

        # Foward to get detections
        detections = self.net.forward()

        # Now we need to iterate over each detection and save the bounding boxes
        rects = []
        for i in np.arange(0, detections.shape[2]):
            # Extract the confidence of the detection
            confidence = detections[0, 0, i, 2]

            # Check if confidence if above minimum treshold
            if confidence > self.detection_treshold:
                idx = int(detections[0, 0, i, 1])

                # Check if detected object is in classes that should be detected
                # If not, ignore it
                if not self.class_names[idx] in self.detection_classes:
                    continue

                # Get the bouxing box of the object
                box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])

                # Append the bounding box to the list
                rects.append(box)

        bounding_boxes = np.array(rects)
        # Make sure bounding boxes are integer numbers
        bounding_boxes = bounding_boxes.astype(int)

        return bounding_boxes
