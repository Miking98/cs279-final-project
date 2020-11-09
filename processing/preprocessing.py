import numpy as np
from skimage import io, draw, color, exposure
from sklearn.cluster import KMeans
from scipy.ndimage import morphology
from matplotlib import pyplot as plt
import json


def apply_pre_processing(image):
    """
    Takes in an image of a thin film blood smear and applies the pre-processing pipeline to it.

    The pipeline consists of:
        - background detection based on saturation levels.
        - color correction based on the average background color.
        - hue normalization of the foreground to an average value of 0.65 (blue-purple).
        - brightness correction of the foreground to maximize contrast.
        - erasure of background pixels.

    :param image: RGB image (numpy array) of a thin film blood smear.
    :return: Fully processed image where the background has been erased.
    """
    # convert to HSV
    # https://en.wikipedia.org/wiki/HSL_and_HSV#/media/File:HSV_color_solid_cylinder_saturation_gray.png
    hsv_img = color.rgb2hsv(image)
    image = np.copy(image)

    # find background through a threshold saturation level
    # apply k-means to cluster the saturation values into two groups (foreground and background)
    saturation_values = hsv_img[:, :, 1].flatten().reshape(-1, 1)
    k_means = KMeans(n_clusters=2).fit(saturation_values)
    threshold_saturation = (k_means.cluster_centers_[0][0] + k_means.cluster_centers_[1][0]) / 2
    background_indices = hsv_img[:, :, 1] < threshold_saturation
    background_indices = ~morphology.binary_fill_holes(~background_indices)

    # apply RGB vector to correct average background color to pure white
    avg_background_color = np.average(image[background_indices], axis=0)
    to_white = np.ones(3) * 255 - avg_background_color
    image = (image + to_white).clip(0, 255).astype(np.uint8)
    hsv_img = color.rgb2hsv(image)

    # move average foreground hue value to purple = 0.65
    avg_foreground_hue = np.average(hsv_img[~background_indices, 0])
    print(avg_foreground_hue)
    hsv_img[:, :, 0] += -avg_foreground_hue + 0.65
    hsv_img[hsv_img[:, :, 0] < 0, 0] += 1
    hsv_img[hsv_img[:, :, 0] > 1, 0] -= 1
    image = (color.hsv2rgb(hsv_img) * 255).astype(np.uint8)

    # correct brightness
    # https://scikit-image.org/docs/dev/api/skimage.exposure.html#skimage.exposure.equalize_hist
    image = (exposure.equalize_hist(image, mask=~background_indices) * 255).astype(np.uint8)

    # remove background pixels
    image[background_indices] = [255, 0, 0]
    return image


def demo(path_to_data):
    """
    Demonstration of the pre-processing pipeline.

    Iterates over the test data and displays the original next to the pre-processed version of each
    image.

    :param path_to_data: Path to the folder containing "test.json", "training.json", and "images/"
    """
    color_mapping = {'difficult': [185, 51, 173],  # purple
                     'gametocyte': [255, 99, 25],  # orange
                     'leukocyte': [0, 0, 255],  # blue
                     'red blood cell': [255, 0, 0],  # red
                     'schizont': [252, 204, 10],  # yellow
                     'ring': [153, 102, 51],  # brown
                     'trophozoite': [0, 147, 60]  # green
                     }

    with open(path_to_data + "/training.json") as json_file:
        files = json.load(json_file)
        fig, axs = plt.subplots(ncols=2, nrows=1)
        for file in files:
            # read the image
            file_path = file['image']['pathname']
            fig.suptitle(file_path)
            image = io.imread(path_to_data + file_path)

            processed = apply_pre_processing(image)

            # draw bounding boxes
            for obj in file['objects']:
                bounding_box = obj['bounding_box']
                minimum = bounding_box['minimum']
                start = (minimum['r'], minimum['c'])
                maximum = bounding_box['maximum']
                stop = (maximum['r'] - 2, maximum['c'] - 2)
                rr, cc = draw.rectangle_perimeter(start=start, end=stop)
                image[rr, cc] = color_mapping[obj['category']]

            # show image before pre-processing on the left
            axs[0].imshow(image)
            # show the processed image on the right
            axs[1].imshow(processed)
            fig.show()
            plt.pause(0.25)


if __name__ == "__main__":
    demo("data")
