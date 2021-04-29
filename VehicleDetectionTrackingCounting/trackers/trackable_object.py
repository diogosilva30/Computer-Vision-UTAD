import numpy as np


class TrackableObject:
    def __init__(self, id, centroid):
        # Save the id of the object
        self.id = id
        # Initialize a list of centroids with the
        # first element being the centroid passed
        # in the constructor
        self.centroids = [centroid]

        # Initialize a boolean flag to indicate if the
        # object already has been counted or not
        self.counted = False

        # Generate a random color for the object
        color = np.random.choice(range(256), size=3)
        # convert data types int64 to int
        color = (int(color[0]), int(color[1]), int(color[2]))
        self.color = tuple(color)

    def add_centroid(self, new_centroid):
        """
        Appends a new centroid to the list of centroids.
        """
        self.centroids.append(new_centroid)

    @property
    def centroid(self):
        """
        Property that returns the current centroid
        (last one in the list)
        """
        return self.centroids[-1]