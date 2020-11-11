import timeit
import numpy as np
from skimage import io, draw, color, exposure, measure, filters
from skimage import morphology as morph
from sklearn.cluster import KMeans
from scipy.ndimage import morphology
from matplotlib import pyplot as plt
import json


def find_background(image, morph_filters=True):
    """
    Returns an index map of the pixels in input image believed to be background.

    This function first converts to HSV and then uses k-means to partition the saturation-values.
    The average of the two cluster-middle-points is used as the threshold to distinguish between
    foreground and background.
    Optionally, morphological filters may be applied to fill in gaps in the detected foreground
    (e.g., holes within detected cells) and remove background specks (debris that has the same color
    as the cells we want to detect).

    :param image: RGB image (either uint8 0-255, or float 0.0-1.0), with light background.
    :param morph_filters: boolean flag. Whether or not to apply morphological filters to
        improve background detection.
    :return: index map of pixels believed to be the background.
    """
    if image.dtype == np.uint8:
        image = image.astype(np.float64) / 255.
    # convert to HSV
    # https://en.wikipedia.org/wiki/HSL_and_HSV#/media/File:HSV_color_solid_cylinder_saturation_gray.png
    hsv_img = color.rgb2hsv(image)

    # find background through a threshold saturation level
    # apply k-means to cluster the saturation values into two groups (foreground and background)
    saturation_values = hsv_img[:, :, 1].flatten().reshape(-1, 1)
    k_means = KMeans(n_clusters=2).fit(saturation_values)
    threshold_saturation = (k_means.cluster_centers_[0][0] + k_means.cluster_centers_[1][0]) / 2
    background_mask = hsv_img[:, :, 1] < threshold_saturation
    if morph_filters:
        mask = np.ones_like(background_mask)
        mask[background_mask] = 0
        mask = morph.binary_closing(mask, selem=morph.disk(30))
        mask = morph.binary_dilation(mask, selem=morph.disk(5))
        mask = morph.binary_opening(mask, selem=morph.disk(30))
        background_mask = mask < 0.5
    return background_mask


def apply_color_correction(image, background_mask):
    """
    Returns a color-corrected version of the input image.

    - color correction based on the average background color.
    - hue normalization of the foreground to an average value of 0.65 (blue-purple).
    - brightness correction of the foreground to maximize contrast.

    :param image: RGB image (either uint8 0-255, or float 0.0-1.0), with light background.
    :param background_mask:
    :return: Color-corrected version of the input image.
    """
    if image.dtype == np.uint8:
        image = image.astype(np.float64) / 255.
    # apply RGB vector to correct average background color to pure white
    avg_background_color = np.average(image[background_mask], axis=0)
    to_white = np.ones(3) - avg_background_color
    image = (image + to_white).clip(0.0, 1.0)
    hsv_img = color.rgb2hsv(image)

    # move average foreground hue value to purple = 0.65
    avg_foreground_hue = np.average(hsv_img[~background_mask, 0])
    hsv_img[:, :, 0] += -avg_foreground_hue + 0.65
    hsv_img[hsv_img[:, :, 0] < 0, 0] += 1
    hsv_img[hsv_img[:, :, 0] > 1, 0] -= 1
    image = color.hsv2rgb(hsv_img)

    # correct brightness
    # https://scikit-image.org/docs/dev/api/skimage.exposure.html#skimage.exposure.equalize_hist
    image = exposure.equalize_hist(image, mask=~background_mask)

    return (image * 255).astype(np.uint8)


def apply_pre_processing(image, background_rem, morph_filter):
    """
    Takes in an image of a thin film blood smear and applies the pre-processing pipeline to it.

    The pipeline consists of:
        - background detection based on saturation levels.
        - color correction based on the average background color.
        - hue normalization of the foreground to an average value of 0.65 (blue-purple).
        - brightness correction of the foreground to maximize contrast.
        - erasure of background pixels.

    :param image: RGB image (numpy array) of a thin film blood smear.
    :param morph_filter: Whether or not to apply morphological filters during background detection.
    :param background_rem: Whether or not to erase all pixels detected to be part of the background.
    :return: Fully processed image.
    """
    background_mask = find_background(image, morph_filter)
    image = apply_color_correction(image, background_mask)

    if background_rem:
        image[background_mask] = [255, 255, 255]
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
        fig, axs = plt.subplots(ncols=2, nrows=2)
        for file in files:
            # read the image
            file_path = file['image']['pathname']
            fig.suptitle(file_path+"\nclick on plot to see next image")
            print(f"processing file: {file_path}")
            start_time = timeit.default_timer()
            image = io.imread(path_to_data + file_path)

            # no background removal (top right)
            processed1 = apply_pre_processing(image, background_rem=False, morph_filter=False)
            axs[0][1].imshow(processed1)
            axs[0][1].set_title("no background rem.")
            # background removal, w/o morphological filter on background mask (bottom left)
            processed2 = apply_pre_processing(image, background_rem=True, morph_filter=False)
            axs[1][0].imshow(processed2)
            axs[1][0].set_title("background rem. (no morph)")
            # background removal with morph. filters on background mask (bottom right)
            processed3 = apply_pre_processing(image, background_rem=True, morph_filter=True)
            axs[1][1].imshow(processed3)
            axs[1][1].set_title("background rem. (with morph)")

            # draw bounding boxes
            for obj in file['objects']:
                bounding_box = obj['bounding_box']
                minimum = bounding_box['minimum']
                start = (minimum['r'], minimum['c'])
                maximum = bounding_box['maximum']
                stop = (maximum['r'] - 2, maximum['c'] - 2)
                rr, cc = draw.rectangle_perimeter(start=start, end=stop)
                image[rr, cc] = color_mapping[obj['category']]

            # image before pre-processing with bounding boxes (top left)
            axs[0][0].imshow(image)
            axs[0][0].set_title("input with ground truth")
            fig.show()
            print(f"elapsed time: {timeit.default_timer() - start_time} seconds")
            plt.ginput()  # wait for click before doing more processing


if __name__ == "__main__":
    demo("D:\Workspace\CS279\data")
