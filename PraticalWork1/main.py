"""
Pratical Assignment 1 - Computer Vision @ UTAD
Goal of this work:
- Implement 2/3 Segmentation Algorithms (Global Tresholding)
- Implement 2/3 Evaluation Metrics
- Perform a critical analysis of the results
"""

# Import dependencies
import glob
import numpy as np
import os
from skimage import filters, io, color
from abc import ABC, abstractmethod

from mdutils.mdutils import MdUtils

from sklearn.cluster import KMeans


def img_to_markdown(path):
    return MdUtils.new_inline_image(text=path, path=path)


def scale_img(img):
    """
    Scales image to 0-255, and type 'uint8'
    """
    return ((img - img.min()) * (1 / (img.max() - img.min()) * 255)).astype("uint8")


class Segmentation(ABC):
    def __init__(self) -> None:
        # Make sure Results dir exists
        if not os.path.exists("Results"):
            os.mkdir("Results")

    def save(self, img, img_name):
        """
        Saves an img
        """
        method_name = str(self).replace(" ", "").replace("-", "_")
        # Save the image to original folder,
        # adding name of segmentation method to name
        dest = f"Results/{img_name.split('.jpg')[0]}_{method_name}.jpg"

        # Save the img
        io.imsave(dest, (img))
        # Return where the img was saved
        return dest

    def _load_img(self, img_location):
        """
        Loads a image
        """
        return io.imread(img_location)

    @abstractmethod
    def segment(self, img_location):
        """
        Perform segmentation
        """

    def __str__(self) -> str:
        """
        Returns the name of the method without "Segmentation".
        """
        return self.__class__.__name__.split("Segmentation")[0]


class OtsuSegmentation(Segmentation):
    def __init__(self, n_thresholds: int = 3) -> None:
        self.n_thresholds = n_thresholds
        super().__init__()

    def segment(self, img, img_name):
        """
        Applies Otsu's Multi-Tresholding to a Image

        Parameters
        ----------
        img: np.ndarray
            The image to perform segmentation.

        Returns
        -------
        np.ndarray
            The segmented image
        """

        # Apply Gaussian Blur to smooth the image (5x5 kernel)
        img = filters.gaussian(img)
        # Apply otsu's global tresholding method
        thresholds = filters.threshold_multiotsu(img, classes=self.n_thresholds)
        # Use the found treshholds to segment image
        img = np.digitize(img, bins=thresholds)

        img = scale_img(img)

        return self.save(img, img_name)

    def __str__(self) -> str:
        return f"Otsu's - {self.n_thresholds} thresholds"


class KMeansSegmentation(Segmentation):
    def __init__(self, n_clusters=3) -> None:
        self.n_clusters = n_clusters
        super().__init__()

    def segment(self, img, img_name):
        """
        Applies K-means Segmentation to a Image

        Parameters
        ----------
        img : np.ndarray
            The image to perform segmentation.

        Returns
        -------
        np.ndarray
            The segmented image
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

        img_segmented.shape = img.shape
        img_segmented = scale_img(img_segmented)
        return self.save(img_segmented, img_name)

    def __str__(self) -> str:
        return f"K-means++ - {self.n_clusters} clusters"


def main():
    # Find all the images
    images = glob.glob("Images/**")

    # Create Markdown file
    mdFile = MdUtils(file_name="Results", title="Pratical Work - Segmentation")

    # Define the list of segmentation methods to use
    segmentation_methods = [
        # KMeansSegmentation(n_clusters=2),
        KMeansSegmentation(n_clusters=5),
        OtsuSegmentation(n_thresholds=5),
    ]
    # Let's create a markdown table for the segmentation results
    table_contents = ["Original Image"]
    # Add Method Headers
    table_contents.extend([str(method) for method in segmentation_methods])

    # Iterate over each image
    img_count = 1
    for img_location in images:
        # Display current image being processed
        print(f"Image {img_count}/{len(images)}")
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

        # Add the segmentation results to the Markdown Table
        table_contents.extend(
            [
                img_to_markdown(gray_location),
            ]
            + [
                img_to_markdown(method.segment(gray, img_name))
                for method in segmentation_methods
            ]
        )

        img_count += 1

    # Create the table
    mdFile.new_table(
        columns=len(segmentation_methods) + 1,
        rows=img_count,
        text=table_contents,
        text_align="center",
    )

    # Save the Markdown file
    mdFile.create_md_file()


if __name__ == "__main__":
    main()