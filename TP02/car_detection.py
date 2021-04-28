import cv2
from trackers import TrackableObject, CentroidTracker
from arguments_parser import ap
import numpy as np
import time
from detectors import BaseDetector, DeepLearningDetector


class CarTracker:
    def __init__(
        self,
        detector: BaseDetector,
        tracker,
    ):
        # Save the detector
        self.detector = detector

    def process_video(self, video_file):
        # Instantiate a OpenCV video capture
        cap = cv2.VideoCapture(video_file)

        # instantiate our centroid tracker, then initialize a list to store
        # each of our dlib correlation trackers, followed by a dictionary to
        # map each unique object ID to a TrackableObject
        centroid_tracker = CentroidTracker()
        trackers = cv2.MultiTracker_create()
        trackable_objects = {}

        total_frames = 0
        total_cars = 0
        # Start the frame reading loop
        while True:
            # Start a timer for FPS calculations
            timer = cv2.getTickCount()

            # Read a frame from the capture
            success, frame = cap.read()

            # Check for success
            if not success:
                break

            # initialize the current status along with our list of bounding
            # box rectangles returned by either (1) our object detector or
            # (2) the correlation trackers
            status = "Waiting"
            rects = []

            # check to see if we should run a more computationally expensive
            # object detection method to aid our tracker
            if total_frames % 5 == 0:

                # set the status and initialize our new set of object trackers
                status = "Detecting"
                # 1. Perform object detection
                detections = self.detector.detect(frame)
                if detections is not None:
                    trackers = cv2.MultiTracker_create()
                    for detection in detections:
                        # construct a dlib rectangle object from the bounding
                        # box coordinates and then start the dlib correlation
                        # tracker
                        bounding_box = tuple(detection[2])
                        # add the tracker to our list of trackers so we can
                        # utilize it durin skip frames
                        trackers.add(cv2.TrackerMOSSE_create(), frame, bounding_box)
                        rects.append(bounding_box)
            # otherwise, we should utilize our object *trackers* rather than
            # object *detectors* to obtain a higher frame processing throughput
            else:
                status = "Tracking"
                success, bounding_boxes = trackers.update(frame)

                for bounding_box in bounding_boxes:
                    rects.append(bounding_box)

            # Draw a horizontal line in the center of the frame
            # If a object crosses this line we will increase the counter
            line_height = frame.shape[0] // 2
            line_width = frame.shape[1]

            cv2.line(
                frame,
                (0, line_height),
                (line_width, line_height),
                (0, 255, 255),
                2,
            )

            # use the centroid tracker to associate the (1) old object
            # centroids with (2) the newly computed object centroids
            objects = centroid_tracker.update(rects)

            print(objects)
            # for box_id in objects:
            #     x, y, w, h, id = box_id

            #     x = int(x)
            #     y = int(y)
            #     w = int(w)
            #     h = int(h)
            #     cx = int(x + (w / 2))
            #     cy = int(y + (h / 2))

            #     cv2.putText(
            #         frame,
            #         str(id),
            #         (x, y - 15),
            #         cv2.FONT_HERSHEY_PLAIN,
            #         2,
            #         (255, 0, 0),
            #         2,
            #     )
            #     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            #     cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

            #     # Check for counting
            #     if cy > line_height:
            #         total_cars += 1
            # # loop over the tracked objects
            for (object_id, centroid) in objects.items():
                # check to see if a trackable object exists for the current
                # object ID
                trackable_object = trackable_objects.get(object_id, None)
                # if there is no existing trackable object, create one
                if trackable_object is None:
                    trackable_object = TrackableObject(object_id, centroid)

                # otherwise, there is a trackable object.
                else:
                    # check to see if the object has been counted or not
                    # and if the centroid is bellow the horizontal line
                    if not trackable_object.counted and centroid[1] > line_height:
                        total_cars += 1
                        trackable_object.counted = True
                # store the trackable object in our dictionary
                trackable_objects[object_id] = trackable_object

            #
            #     break
            # Calculate the FPS
            fps = int(cv2.getTickFrequency() / (cv2.getTickCount() - timer))
            # construct a tuple of information we will be displaying on the
            # frame
            info = [("Counter", total_cars), ("Status", status), ("FPS", fps)]
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

            # show the output frame
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(10) & 0xFF

            if key == ord("q"):
                break
            total_frames += 1

        cap.release()
        # close any open windows
        cv2.destroyAllWindows()


# args = ap.parse_args()

configPath = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "frozen_inference_graph.pb"
detector = DeepLearningDetector(
    weights_path="frozen_inference_graph.pb",
    config_path="ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt",
    detection_treshold=0.55,
    detection_classes=["car"],
    class_names_path="coco.names",
)
tracker = cv2.TrackerKCF_create()

ct = CarTracker(detector=detector, tracker=tracker)

ct.process_video("video.mp4")

# Initialize video capture

# # Intantiate a tracker
# tracker =

# trackers = []
# total_frames = 0
# while True:
#     # Instantiate a timer for FPS calculations
#     timer = cv2.getTickCount()
#     success, frame = cap.read()

#     fps = int(cv2.getTickFrequency() / (cv2.getTickCount() - timer))

#     cv2.putText(
#         frame, f"FPS: {fps}", (75, 50), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2
#     )
#     if not success:
#         break

#     # Perform object detection
#     class_id, confidence, bounding_box = self.detector.detect(frame)
#     object_name = class_names[class_id - 1].upper()

#     cv2.imshow("Tracking", frame)

#     key = cv2.waitKey(1)
#     if key == ord("q"):
#         break
#     if key == ord("p"):
#         cv2.waitKey(-1)
#     # # Intiialize the current status along with our list
#     # # of bounding box rectangles returned by either:
#     # # (1) - Object detector
#     # # (2) - Correlation Trackers

#     # status = "Waiting"
#     # rects = []

#     # # Check if we should enter "Detection" Mode
#     # if total_frames % args.skip_frames == 0:

#     #     class_id, confidence, bounding_box = self.detector.detect(frame)
#     #     object_name = class_names[class_id - 1].upper()

#     #     # Check if object belong to the "car" class and confidence is above treshold
#     #     if object_name != "car" or confidence < args.confidence:
#     #         continue

#     #     # Construct a dlib rectangle object from the bounding box
#     #     # coordinates and then start the dlib correlation tracker
#     #     tracker = dlib.correlation_tracker()
#     #     start_x, start_y, end_x, end_y = bounding_box.astype("int")
#     #     rect = dlib.rectangle(start_x, start_y, end_x, end_y)
#     #     tracker.start_track(frame, rect)

#     #     # Add the tracker to our list of tracker so we can
#     #     # utilize it during skip frames
#     #     trackers.append(tracker)
#     # # Otherwise enter "Tracking" Mode
#     # else:
#     #     # Loop over each tracker
#     #     # set the status of our system to "Tracking"
#     #     status = "Tracking"

#     #     # Update the tracker and grab the updated position
#     #     tracker.update(frame)

#     #     position = tracker.get_position()

#     #     # Unpack the position object
#     #     start_x = int(position.left())
#     #     start_y = int(position.top())
#     #     end_x = int(position.right())
#     #     end_y = int(position.bottom())

#     #     rects.append((start_x, start_y, end_x, end_y))

#     # # Draw a horizontal line in the center of the frame
#     # # If a object crosses this line we will increase the counter
#     # height = frame.shape[0]
#     # width = frame.shape[1]
#     # cv2.line(frame, (0, height // 2), (width, height // 2), (0, 255, 255), 2)

#     # # Use the centroide tracker to associate the (1) old object centroid
#     # # with (2) the newly computed object centroids
#     # objects = ct.update(rects)

# cap.release()
# cv2.destroyAllWindows()
