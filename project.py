import numpy as np
from skimage import data, filters, io, draw, color
from scipy import misc, ndimage
from matplotlib import pyplot as plt
import json
import matplotlib.patches as patches
from imgaug import augmenters as iaa
import os

# Mask R-CNN
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize

# Malaria Dataset
from Malaria import MalariaDataset, MalariaConfig, MalariaInferenceConfig

PROJECT_DIR = os.path.abspath('.')

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(PROJECT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(PROJECT_DIR, "logs")

# Results directory
# Save submission files here
RESULTS_DIR = os.path.join(PROJECT_DIR, "results/")

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

def display_training_set():
    with open("/Users/mwornow/desktop/data/training.json") as json_file:
        files = json.load(json_file)
        for file in files:
            # Read the image
            file_path = file['image']['pathname']
            # file_path = '/images/ffd59802-46c6-4b58-80fe-e534e39781a7.png'
            image = io.imread("/Users/mwornow/desktop/data" + file_path)

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

def train(model, dataset_dir):
    # References:
    #    https://medium.com/analytics-vidhya/deep-learning-computer-vision-object-detection-infection-classification-on-malaria-images-3769b4f51de9
    #    https://github.com/matterport/Mask_RCNN/blob/master/samples/nucleus/nucleus.py

    """Train the model."""
    # Training dataset.
    dataset_train = MalariaDataset()
    dataset_train.load_dataset(dataset_dir, is_train = True)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = MalariaDataset()
    dataset_val.load_dataset(dataset_dir, is_val = True)
    dataset_val.prepare()

    # Image augmentation
    # http://imgaug.readthedocs.io/en/latest/source/augmenters.html
    augmentation = iaa.SomeOf((0, 2), [
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.OneOf([iaa.Affine(rotate=90),
                   iaa.Affine(rotate=180),
                   iaa.Affine(rotate=270)]),
        iaa.Multiply((0.8, 1.5)),
        iaa.GaussianBlur(sigma=(0.0, 5.0))
    ])


    # If starting from imagenet, train heads only for a bit
    # since they have random weights
    print("Train network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=20,
                augmentation=augmentation,
                layers='heads',
                use_multiprocessing = False)

    print("Train all layers")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=40,
                augmentation=augmentation,
                layers='all',
                use_multiprocessing = False)




if __name__ == '__main__':
    import argparse

    '''
    HOW TO RUN: 
        python3 project.py train --dataset /Users/mwornow/Desktop/data --weights coco
    '''

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Malaria Dataset Classification')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'inference'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/dataset/",
                        help='Root directory of the dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco' or 'imagenet' or 'last'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "inference":
        assert True

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    config = MalariaConfig() if args.command == "train" else MalariaInferenceConfig()
    config.display()

    # Create model
    model = modellib.MaskRCNN(  mode="training" if args.command == "train" else "inference",
                                config=config,
                                model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model, args.dataset)
    elif args.command == "inference":
        inference(model, args.dataset)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'inference'".format(args.command))