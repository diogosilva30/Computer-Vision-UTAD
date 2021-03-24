"""
Pratical Assignment 1 - Computer Vision @ UTAD
Goal of this work:
- Implement 2/3 Segmentation Algorithms (Global Tresholding)
- Implement 2/3 Evaluation Metrics
- Perform a critical analysis of the results
"""
# Import dependencies
import os
from abc import ABC, abstractmethod
from timeit import default_timer as timer

import glob
import numpy as np

from skimage import filters, io, color, metrics
from sklearn.cluster import KMeans

from mdutils.mdutils import MdUtils


def img_to_markdown(path: str) -> str:
    """
    Creates a Markdown Image.

    Parameters
    ----------
    path: str
        The image path.
    """
    return MdUtils.new_inline_image(text=path, path=path)


def scale_img(img: np.ndarray) -> np.ndarray:
    """
    Scales image to 0-255, and type 'uint8'

    Parameters
    ----------
    img: np.ndarray
        The image to be scaled

    Returns
    -------
    np.ndarray
        The scaled image
    """
    return ((img - img.min()) * (1 / (img.max() - img.min()) * 255)).astype("uint8")


class Metric(ABC):
    """
    Abstract class for Segmentation Metrics
    """

    @abstractmethod
    def evaluate(self, ground_truth, segmentation):
        """
        Evalutes the segmentation, when compared to the ground truth.
        """

    def __str__(self):
        """
        Returns the name of the class without "Metric".
        """
        return self.__class__.__name__.split("Metric")[0]


class MeanStructuralSimilitaryIndexMetric(Metric):
    def evaluate(self, ground_truth, segmentation):
        """
        Evaluates a segmentation using the Mean Structural
        Similitary Index
        """
        return metrics.structural_similarity(ground_truth, segmentation)


class MeanSquaredErrorMetric(Metric):
    def evaluate(self, ground_truth, segmentation):
        """
        Evaluates a segmentation using the Mean Squared Error
        """
        return metrics.mean_squared_error(ground_truth, segmentation)


class PeakSignalNoiseRatioMetric(Metric):
    def evaluate(self, ground_truth, segmentation):
        """
        Evaluates a segmentation using the Peak Signal-to-Noise Ratio
        """
        return metrics.peak_signal_noise_ratio(ground_truth, segmentation)


class Segmentation(ABC):
    """
    Abstract class for Segmentation Algorithms
    """

    def __init__(self, metrics: list[Metric] = None) -> None:
        """
        Base constructor for a Segmentation algorithm

        Parameters
        ----------
        metrics: list of Metric (optional)
            A list of Metric objects, to evaluate the segmentations.
        """
        self.metrics = metrics
        # Make sure Results dir exists
        if not os.path.exists("Results"):
            os.mkdir("Results")

    def save(self, img: str, img_name: str) -> str:
        """
        Saves an image

        Parameters
        ----------
        img: np.ndarray
            The image to be saved.
        img_name: str
            The name of the image.

        Returns
        -------
        str
            The path where the image was saved.
        """
        method_name = str(self).replace(" ", "").replace("-", "_")
        # Save the image to original folder,
        # adding name of segmentation method to name
        dest = f"Results/{img_name.split('.jpg')[0]}_{method_name}.jpg"

        # Save the img
        io.imsave(dest, (img))
        # Return where the img was saved
        return dest

    @abstractmethod
    def segment(self, img: np.ndarray, img_location: str) -> str:
        """
        Performs segmentation on a image

        Parameters
        ----------
        img: np.ndarray
            The image to perform segmentation.

        img_name: str
            The name of the image.
        Returns
        -------
        str
            The path where the segmented image was saved.
        """

    def evaluate(self, original_img: np.ndarray, segmented_img: np.ndarray) -> dict:
        """
        Computes metric values for a segmentation.

        Parameters
        ----------
        original_img: np.ndarray
            The ground truth image.
        segmented_img: np.ndarray
            The segmented image.

        Returns
        -------
        dict
            Returns a dictionary where the keys are the metrics names,
            and the values the metrics values.
        """
        if not self.metrics:
            raise ValueError("No metrics were passed in the constructor.")
        return {
            str(metric): metric.evaluate(original_img, segmented_img)
            for metric in self.metrics
        }

    def __str__(self) -> str:
        """
        Returns the name of the class without "Segmentation".
        """
        return self.__class__.__name__.split("Segmentation")[0]


class OtsuSegmentation(Segmentation):
    """
    Otsu's MultiTresholding Algorithm
    for image segmentation.
    """

    def __init__(self, n_thresholds: int = 3, **kwargs) -> None:
        self.n_thresholds = n_thresholds
        super().__init__(**kwargs)

    def segment(self, img):
        """
        Applies Otsu's Multi-Tresholding to a Image

        Parameters
        ----------
        img: np.ndarray
            The image to perform segmentation.

        Returns
        -------
        str
            The path where the segmented image was saved.
        """

        # Apply Gaussian Blur to smooth the image (5x5 kernel)
        img = filters.gaussian(img)
        # Apply otsu's global thresholding method
        thresholds = filters.threshold_multiotsu(img, classes=self.n_thresholds)
        # Use the found thresholds to segment image
        img = np.digitize(img, bins=thresholds)

        # Scale the image
        img = scale_img(img)

        return img

    def __str__(self) -> str:
        return f"Otsu's - {self.n_thresholds} thresholds"


class KMeansSegmentation(Segmentation):
    """
    Implementation of K-Means++ Algorithm
    for image segmentation.
    """

    def __init__(self, n_clusters=3, **kwargs) -> None:
        self.n_clusters = n_clusters
        super().__init__(**kwargs)

    def segment(self, img):
        """
        Applies K-means Segmentation to a Image

        Parameters
        ----------
        img: np.ndarray
            The image to perform segmentation.

        Returns
        -------
        str
            The path where the segmented image was saved.
        """
        # Create a line array, the lazy way
        segmented_img = img.reshape((-1, 1))
        # Define the k-means clustering problem
        kmeans = KMeans(n_clusters=self.n_clusters)
        # Solve the k-means clustering problem
        kmeans.fit(segmented_img)
        # Get the coordinates of the clusters centres as a 1D array
        values = kmeans.cluster_centers_.squeeze()
        # Get the label of each point
        labels = kmeans.labels_

        # create an array from labels and values
        img_segmented = np.choose(labels, values)

        # Reshape image back to original
        img_segmented.shape = img.shape

        # Scale image
        img_segmented = scale_img(img_segmented)
        return img_segmented

    def __str__(self) -> str:
        """
        String representation of this class.
        """
        return f"K-means++ - {self.n_clusters} clusters"


def main():
    # Find all the images
    images = glob.glob("Images/**")
    # print(images)
    # raise ValueError
    # Create Markdown file
    mdFile = MdUtils(file_name="Results", title="Pratical Work - Segmentation")

    # Define the list of segmentation metrics to use
    segmentation_metrics = [
        MeanStructuralSimilitaryIndexMetric(),
        MeanSquaredErrorMetric(),
        PeakSignalNoiseRatioMetric(),
    ]

    # Define the list of segmentation methods to use
    segmentation_methods = [
        # KMeansSegmentation(n_clusters=2),
        KMeansSegmentation(
            n_clusters=3,
            metrics=segmentation_metrics,
        ),
        OtsuSegmentation(
            n_thresholds=3,
            metrics=segmentation_metrics,
        ),
        KMeansSegmentation(
            n_clusters=5,
            metrics=segmentation_metrics,
        ),
        OtsuSegmentation(
            n_thresholds=5,
            metrics=segmentation_metrics,
        ),
    ]

    # Let's create a markdown table for the segmentation results
    table_contents = ["Original Image"]
    # Add Method Headers
    table_contents.extend([str(method) for method in segmentation_methods])

    # Iterate over each image
    for i, img_location in enumerate(images):
        print(f"Segmenting image {i+1}/{len(images)}")
        # Extract img name
        img_name = img_location.split("\\")[-1]
        # Read img as grayscale
        gray = (color.rgb2gray(io.imread(img_location)) * 255).astype(np.uint8)

        # make sure Grayscale folder exists
        if not os.path.exists("Grayscale"):
            os.mkdir("Grayscale")

        # Save the gray version
        gray_location = f"Grayscale/{img_name}"
        io.imsave(gray_location, gray)

        # Add first column (original image)
        table_contents.extend([img_to_markdown(gray_location)])
        # Iterate over each segmentation method
        for method in segmentation_methods:
            # Segment the image
            # Record time it takes to segment image
            start = timer()
            segmented_img = method.segment(gray)
            end = timer()
            # Save the image
            segmented_img_location = method.save(segmented_img, img_name)
            # Evaluate the segmentation
            segmentation_results = method.evaluate(gray, segmented_img)
            # Convert the `segmentation_results` dictionary
            # to a more human redable version
            segmentation_results = (
                f"Time taken to segment: {(end-start):.3f} seconds <br/>"
                + "<br/>".join(
                    [
                        f"**{key}**: {value:.2f}"
                        for key, value in segmentation_results.items()
                    ]
                )
            )

            # Add the segmentation results to the Markdown Table
            table_contents.extend(
                [
                    f"{(img_to_markdown(segmented_img_location))}{str(segmentation_results)}"
                ]
            )

    # Create the table
    mdFile.new_table(
        columns=len(segmentation_methods) + 1,
        rows=len(images) + 1,
        text=table_contents,
        text_align="center",
    )

    # Save the Markdown file
    mdFile.create_md_file()


if __name__ == "__main__":
    main()