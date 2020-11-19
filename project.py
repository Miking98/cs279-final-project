import numpy as np
from skimage import data, filters, io, draw, color
from scipy import misc, ndimage
from matplotlib import pyplot as plt
import json
import matplotlib.patches as patches
from imgaug import augmenters as iaa
import os
import datetime

# Mask R-CNN
from Mask_RCNN.mrcnn import utils
import Mask_RCNN.mrcnn.model as modellib
from Mask_RCNN.mrcnn import visualize

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

def inference(model, dataset_dir):
    """Run inference on images in the given directory."""
    print("Running on {}".format(dataset_dir))

    # Create directory
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    submit_dir = "submit_{:%Y%m%dT%H%M%S}".format(datetime.datetime.now())
    submit_dir = os.path.join(RESULTS_DIR, submit_dir)
    os.makedirs(submit_dir)

    # Read dataset
    dataset = MalariaDataset()
    dataset.load_dataset(dataset_dir, is_test = True)
    dataset.prepare()
    # Load over images
    submission = []
    for image_id in dataset.image_ids:
        # Load image and run inference
        image = dataset.load_image(image_id)
        # Detect objects
        r = model.detect([image], verbose=0)[0]
        # Save image with masks
        visualize.display_instances(
            image, r['rois'], r['masks'], r['class_ids'],
            dataset.class_names, r['scores'],
            show_bbox=True, show_mask=False,
            title="Predictions")
        plt.savefig("{}/{}.png".format(submit_dir, dataset.image_info[image_id]["id"]))
    print("---- DONE DETECTING ----")

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

    dataset_val.load_mask(0, visualization = True)
    exit()
    # If starting from imagenet, train heads only for a bit
    # since they have random weights
    print("Train network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=5, # Usually 20
                layers='heads',
                use_multiprocessing = False)

    print("Train all layers")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=10,
                layers='all',
                use_multiprocessing = False)




if __name__ == '__main__':
    import argparse

    '''
    TRAIN:
        python3 project.py train --dataset /Users/mwornow/Desktop/data --weights coco
    TEST:
        
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