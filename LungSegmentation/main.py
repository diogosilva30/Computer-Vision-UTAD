"""
Pratical Assignment 3 - Computer Vision @ UTAD
Goal of this work:
- Perform Lung Segmentation & Calculate volume
"""
# Import dependencies
import os
import time

import numpy as np
import cv2
import glob

from tqdm import tqdm
from mdutils.mdutils import MdUtils
from skimage import filters, morphology
from skimage import img_as_ubyte


def img_to_markdown(path: str) -> str:
    """
    Creates a Markdown Image.
    Parameters
    ----------
    path: str
        The image path.
    """
    return MdUtils.new_inline_image(text=path, path=path)


class LungSegmenter:
    """
    Performs Lung-Segmentation in TAC Images
    """

    def __init__(self, initial_crop_cordinates: tuple) -> None:
        """
        Base constructor for the Lung Segmenter algorithm

        Parameters
        ----------
        initial_crop_cordinates: tuple of int
            A 4 element tuple containing the coordinates of the image ROI.
        """
        self.initial_crop_cordinates = initial_crop_cordinates
        # Make sure Results dir exists
        if not os.path.exists("Results"):
            os.mkdir("Results")

    def __save_img(self, img: np.ndarray) -> str:
        # Generate timestamp
        timestamp = time.time()

        if not os.path.exists("MDImages"):
            os.mkdir("MDImages")
        image_path = f"MDImages/{timestamp}.jpg"
        cv2.imwrite(image_path, img)
        return image_path

    def __segment(self, img_location: str) -> np.ndarray:
        """
        Performs segmentation on a image
        Parameters
        ----------
        img_location: str
            The location of the image to segment.
        Returns
        -------
        np.ndarray
            The segmented lungs image.
        """
        # First read image (as grayscale)
        img = cv2.imread(img_location, 0)

        # Then crop the image to the ROI
        img_crop = img[
            int(self.initial_crop_cordinates[1]) : int(
                self.initial_crop_cordinates[1] + self.initial_crop_cordinates[3]
            ),
            int(self.initial_crop_cordinates[0]) : int(
                self.initial_crop_cordinates[0] + self.initial_crop_cordinates[2]
            ),
        ]

        # Add the cropped image to the Markdown report
        self.table_contents.extend([img_to_markdown(self.__save_img(img_crop))])

        # Now we apply a Gaussian Blur to smooth the image
        blur = filters.gaussian(img_crop)

        # Now we use Otsu's tresholding to find a treshold to binarize the image
        threshold = filters.threshold_otsu(blur)

        # Binarize the image with the found treshold
        binary = blur > threshold

        # Apply median blur to remove "salt and pepper" noise
        binary = filters.median(binary)

        # Add the binary image to the Markdown report
        self.table_contents.extend([img_to_markdown(self.__save_img(binary))])

        # Apply morphological closing on the image
        binary = morphology.binary_closing(binary, morphology.disk(1))

        # Add the closed binary image to the Markdown report
        self.table_contents.extend([img_to_markdown(self.__save_img(binary))])

        # Now we convert the skimage to a compatible opencv image
        binary = img_as_ubyte(binary)

        # Now we can use OpenCV routines

        # Lets find the image countours
        contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Now we sort the countours by countour area (from greatest to smallest)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        # Lets prepare our mask, full black for now
        img_crop_mask = np.zeros_like(img_crop)

        # We now need to check if at least 3 contours were found, the biggest is the "body",
        # the next 2 are the lungs
        if len(contours) >= 3:
            # The largest countour is the person "body" around the lungs, so we skip it
            # The second and third largest contour are the lungs
            lungs = [contours[1], contours[2]]

            for lung in lungs:
                # Check most frequent value of the detected lung
                most_frequent = np.bincount(lung.flatten()).argmax()

                # Check if most frequent value is black
                if most_frequent == 0:
                    # Use both lungs to build the mask
                    cv2.drawContours(img_crop_mask, [lung], -1, 255, cv2.FILLED, 1)

        # If no contours were found, the lungs are probably not visible, so we return a black mask

        # Add the mask image to the Markdown report
        self.table_contents.extend([img_to_markdown(self.__save_img(img_crop_mask))])

        # Return the mask
        return img_crop_mask

    def segment_images(self, images: list[str]) -> list[np.ndarray]:

        # Lets prepare the markdown report
        markdown_file = MdUtils(
            file_name="Results", title="Pratical Work - Lung Segmentation"
        )

        # Create the table contents
        self.table_contents = [
            "Original Image (cropped to ROI)",
            "Generated Mask",
        ]

        # Iterate over each image
        pbar = tqdm(images)

        segmented = []
        for img_location in pbar:
            # Write status to progress bar
            pbar.set_description(f"Segmenting image '{img_location}'")

            # Segment the image
            segmented.append(self.__segment(img_location))

        # Create the table
        markdown_file.new_table(
            columns=2,
            rows=len(images) + 1,
            text=self.table_contents,
            text_align="center",
        )

        # Save the Markdown file
        markdown_file.create_md_file()

        return np.array(segmented)

    def __str__(self) -> str:
        """
        Returns the name of the class.
        """
        return self.__class__.__name__


def main():
    # Find all the images
    images = glob.glob("Images/**")

    segmented_images = LungSegmenter(
        initial_crop_cordinates=(178, 174, 299, 206)
    ).segment_images(images)


if __name__ == "__main__":
    main()
