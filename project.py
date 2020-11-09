import numpy as np
from skimage import data, filters, io, draw, color
from scipy import misc, ndimage
from matplotlib import pyplot as plt
import json
import matplotlib.patches as patches

# Mask R-CNN
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize

# Malaria Dataset
from Malaria import MalariaDataset, MalariaConfig

# For labeling boundary boxes in plots
CATEGORY_COLORS = { 
    'red blood cell' : 'red', 
    'trophozoite' : 'blue',
    'ring' : 'green',
    'schizont' : 'purple',
    'leukocyte' : 'orange',
    'gametocyte' : 'yellow',
    'difficult' : 'brown',
}

def remove_background(image):
    image = image.copy()
    # Convert to HSV
    #   https://en.wikipedia.org/wiki/HSL_and_HSV#/media/File:HSV_color_solid_cylinder_saturation_gray.png
    hsv_img = color.rgb2hsv(image)
    # Set V (=value) to 1.0, i.e. remove all darkness
    hsv_img[:, :, 2] = 1.0
    # Attempt to remove background pixels by removing pixels with S (=saturation) below a certain level
    median_saturation = np.median(hsv_img[:, :, 1])
    image[hsv_img[:, :, 1] < median_saturation * 3] = [255, 0, 0]
    return image

with open("data/training.json") as json_file:
    files = json.load(json_file)
    for file in files:
        # Read the image
        file_path = file['image']['pathname']
        # file_path = '/images/ffd59802-46c6-4b58-80fe-e534e39781a7.png'
        image = io.imread("./data" + file_path)

        # Set up plots
        f, ax = plt.subplots(2,2)
        f.suptitle("Image: " + file_path + " | Size: " + str(image.shape))

        # PLOT (0,0) => Show original image
        ax[0,0].set_title('Original')
        ax[0,0].imshow(image)

        # PLOT (0,1) => Show background highlighted
        ax[0,1].set_title('Background removed')
        ax[0,1].imshow(remove_background(image))

        # code for drawing bounding boxes
        types_in_image = set()
        boxes = []
        for obj in file['objects']:
            category = obj['category']
            bounding_box = obj['bounding_box']
            minimum = bounding_box['minimum']
            maximum = bounding_box['maximum']
            # start = (minimum['r'], minimum['c'])
            # stop = (maximum['r'], maximum['c'])
            # rr, cc = draw.rectangle_perimeter(start=start, end=stop)
            # image[rr, cc] = [0,0,0] if obj['category'] == 'red blood cell' else [0, 255, 0]
            start = (minimum['c'], minimum['r'])
            height = maximum['r'] - minimum['r']
            width = maximum['c'] - minimum['c']
            boxes += [ patches.Rectangle(start, width, height, linewidth = 1, edgecolor = CATEGORY_COLORS[category], facecolor = CATEGORY_COLORS[category], alpha = 0.2) ]
            types_in_image.add(obj['category'])
        
        # PLOT (1,0) => Show boundary boxes
        ax[1,0].set_title('Boundary boxes')
        ax[1,0].imshow(image)
        for p in boxes:
            ax[1,0].add_patch(p)
        # show the image
        plt.show()

# https://medium.com/analytics-vidhya/deep-learning-computer-vision-object-detection-infection-classification-on-malaria-images-3769b4f51de9
train = MalariaDataset()
train.load_dataset(is_train = True)
train.prepare()
test = MalariaDataset()
test.load_dataset(is_train = False)
test.prepare()