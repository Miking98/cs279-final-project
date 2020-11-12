import os, json
from mrcnn import utils
from mrcnn.config import Config
import numpy as np
import skimage.io

class MalariaConfig(Config):
    """
    Derives from the base Config class and overrides values specific
    to the Malaria dataset.
    """
    # Give the configuration a recognizable name
    NAME = "malaria"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 7

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 128
    IMAGE_MAX_DIM = 128

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor size in pixels
    
    # ROIs kept after non-maximum supression (training and inference)
    POST_NMS_ROIS_TRAINING = 1000
    POST_NMS_ROIS_INFERENCE = 2000

    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.9

    # How many anchors per image to use for RPN training
    RPN_TRAIN_ANCHORS_PER_IMAGE = 64
    
    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 10

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5

    # Backbone network architecture
    # Supported values are: resnet50, resnet101
    BACKBONE = "resnet50"

    # Input image resizing
    # Random crops of size 512x512
    IMAGE_RESIZE_MODE = "crop"
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    IMAGE_MIN_SCALE = 2.0

    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask

class MalariaInferenceConfig(MalariaConfig):
    # Set batch size to 1 to run one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # Don't resize imager for inferencing
    IMAGE_RESIZE_MODE = "pad64"
    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.7

class MalariaDataset(utils.Dataset):
    CLASSES = {
        'red blood cell' : 1, 
        'trophozoite' : 2,
        'ring' : 3,
        'schizont' : 4,
        'leukocyte' : 5,
        'gametocyte' : 6,
        'difficult' : 7,
    }

    def get_image_id_from_pathname_entry_in_json(self, pathname):
        return pathname[len("/images/") : -4]

    # X DONE
    def load_dataset(self, dataset_dir, is_train = False, is_val = False, is_test = False):
        """Load the Malaria dataset.
        dataset_dir: Root directory of the dataset (no trailing slash)
        """
        self.dataset_dir = dataset_dir
        self.is_train = is_train
        # Add classes
        for class_name in self.CLASSES.keys():
            self.add_class('malaria', self.CLASSES[class_name], class_name)
        # Add data
        file_name = "training.json" if is_train or is_val else 'test.json'
        with open(os.path.join(dataset_dir, file_name)) as json_file:
            self.images = json.load(json_file)
            for img_idx, img in enumerate(self.images):
                # If train, skip every 10th image
                if is_train and img_idx % 10 == 0: continue
                # If validation, only keep every 10th image
                if is_val and img_idx % 10 != 0: continue
                # Read the image
                img_path = img['image']['pathname']
                img_id = self.get_image_id_from_pathname_entry_in_json(img_path)
                # NOTE: Need to adjust image path so that it doesn't have a leading "/" (otherwise os.path.join will be confused)
                if False:
                    print("ADD IMAGE", img_id, img_path)
                self.add_image('malaria', image_id = img_id, path = os.path.join(dataset_dir, img_path[1:]))

    # X DONE
    def load_image(self, image_idx):
        """Load image.
        """
        image_id = self.image_info[image_idx]['id']
        if False:
            print("LOAD IMAGE: ", image_id)
        return super().load_image(image_idx)

    # DONE
    def load_mask(self, image_idx):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        image_id = self.image_info[image_idx]['id']
        masks = None
        class_ids = []
        # Construct masks from bounding boxes
        for img in self.images:
            # If found the right Image in the JSON file...
            if image_id == self.get_image_id_from_pathname_entry_in_json(img['image']['pathname']):
                if False:
                    print("FOUND MASK FOR: ", image_id)
                img_contents = skimage.io.imread(os.path.join(self.dataset_dir, img['image']['pathname'][1:])) # NOTE: Need to get rid of leading slash
                # For each bounding box in 'objects'....
                masks = np.zeros((img_contents.shape[0], img_contents.shape[1], len(img['objects'])))
                for o_idx, o in enumerate(img['objects']):
                    label = o['category']
                    min_y, min_x = o['bounding_box']['minimum']['r'], o['bounding_box']['minimum']['c']
                    max_y, max_x = o['bounding_box']['maximum']['r'], o['bounding_box']['maximum']['c']
                    masks[min_y:max_y, min_x:max_x, o_idx] = 1
                    class_ids.append(self.CLASSES[label])
                break
        return masks, np.array(class_ids)

    # DONE
    def image_reference(self, image_idx):
        """Return the path of the image."""
        info = self.image_info[image_idx]
        if info["source"] == "malaria":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)