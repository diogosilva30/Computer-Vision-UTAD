import cv2
from trackers import TrackableObject, CentroidTracker
from arguments_parser import ap
import numpy as np
import datetime
import imutils

import time
from detectors import BaseDetector, DeepLearningDetector
from utils import non_max_suppression


class CarTracker:
    def __init__(
        self,
        detector: BaseDetector,
    ):
        # Save the detector
        self.detector = detector

        # Dictionary of tracked objects
        self.tracked_objects = {}

    def __update_or_create_tracked_object(self, object_id, centroid) -> TrackableObject:
        """
        Updates a tracked object. If does not exist, a new object is created.
        """
        # check if this object was been tracked before
        object = self.tracked_objects.get(object_id, None)

        # Check if object has not been tracked before
        if object is None:
            # If not create a new one and add to records
            object = TrackableObject(object_id, centroid=centroid)
        else:
            # Add the new centroid to the list
            object.add_centroid(new_centroid=centroid)

        # Update the collection of tracked objects
        self.tracked_objects[object_id] = object
        return object

    def process_video(self, video_file):
        # Instantiate a OpenCV video capture
        cap = cv2.VideoCapture(video_file)

        # instantiate our centroid tracker, then initialize a list to store
        # each of our dlib correlation trackers, followed by a dictionary to
        # map each unique object ID to a TrackableObject
        tracker = CentroidTracker(maxDisappeared=5, maxDistance=90)

        fps_start_time = datetime.datetime.now()

        total_cars = 0
        total_frames = 0

        # Start the frame reading loop
        while True:

            # Read frame
            success, frame = cap.read()
            # Check for success
            if not success:
                break

            # Resize frame for faster processing
            frame = imutils.resize(frame, width=1020)

            # Increment total frames variable
            total_frames = total_frames + 1

            # Perform detections on frame
            bounding_boxes = self.detector.detect(frame)

            # Perform non max supressing to eliminate "noisy" boxes
            rects = non_max_suppression(bounding_boxes, 0.5)

            # Draw horizontal line at middle of screen
            line_height = frame.shape[0] // 2
            line_width = frame.shape[1]

            cv2.line(
                frame,
                (0, line_height),
                (line_width, line_height),
                (0, 255, 255),
                2,
            )

            # Get objects from tracker
            objects = tracker.update(rects)
            # Iterate over tracked objects
            for (object_id, bbox) in objects.items():

                (x1, y1, x2, y2) = bbox
                x1 = int(x1)
                y1 = int(y1)
                x2 = int(x2)
                y2 = int(y2)

                # Calculate centroids
                cx = int((x1 + x2) / 2.0)
                cy = int((y1 + y2) / 2.0)

                # Update/Create the object
                object = self.__update_or_create_tracked_object(
                    object_id, centroid=(cx, cy)
                )

                # Draw a rectangle on object
                cv2.rectangle(frame, (x1, y1), (x2, y2), object.color, 2)

                # For each centroid of object draw a circle with the unique color
                # of the object (This is the object trail)
                for c in object.centroids:
                    cv2.circle(frame, c, 4, object.color, -1)

                # Write the object ID next to the box
                cv2.putText(
                    frame,
                    f"ID: {object_id}",
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    object.color,
                    2,
                )
                # Calculate FPS
                fps_end_time = datetime.datetime.now()
                time_diff = fps_end_time - fps_start_time
                if time_diff.seconds == 0:
                    fps = 0.0
                else:
                    fps = total_frames / time_diff.seconds

                info = [("Counter", total_cars), ("FPS", int(fps))]
                # loop over the info tuples and draw them on our frame
                for (i, (k, v)) in enumerate(info):
                    text = "{}: {}".format(k, v)
                    cv2.putText(
                        frame,
                        text,
                        (10, 100 - ((i * 20) + 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2,
                    )
                # Check if object has been counted already, and is bellow horizontal line
                if not object.counted and cy > line_height:
                    # Check if bellow horizontal line
                    total_cars += 1
                    object.counted = True

            cv2.imshow("Tracker", frame)
            key = cv2.waitKey(1)
            if key == ord("q"):
                break


# args = ap.parse_args()


detector = DeepLearningDetector(
    proto_path="MobileNetSSD_deploy.prototxt",
    model_path="MobileNetSSD_deploy.caffemodel",
    detection_treshold=0.55,
    detection_classes=["car"],
)


ct = CarTracker(detector=detector)

ct.process_video("video.mp4")
