import argparse

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
# ap.add_argument(
#     "-p",
#     "--prototxt",
#     required=True,
#     default=''
#     help="path to Caffe 'deploy' prototxt file",
# )
# ap.add_argument(
#     "-m",
#     "--model",
#     required=True,
#     help="path to Caffe pre-trained model",
# )
ap.add_argument(
    "-i",
    "--input",
    type=str,
    required=True,
    help="path to optional input video file",
)
ap.add_argument(
    "-o",
    "--output",
    type=str,
    help="path to optional output video file",
)
ap.add_argument(
    "-c",
    "--confidence",
    type=float,
    default=0.4,
    help="minimum probability to filter weak detections",
)
ap.add_argument(
    "-s",
    "--skip-frames",
    type=int,
    default=30,
    help="# of skip frames between detections",
)
