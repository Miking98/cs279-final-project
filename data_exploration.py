import numpy as np
from skimage import data, filters, io, draw, color
from scipy import misc, ndimage
from matplotlib import pyplot as plt
import json
import matplotlib.patches as patches
import os

plt.style.use('seaborn')


'''

- Average number of cells that each class type overlaps with

'''

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

def get_image_id_from_pathname_entry_in_json(pathname):
    return pathname[len("/images/") : -4]

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

def display_image(image_ids = None):
    # image_ids : If None, then show all images. Otherwise, only show images with ID in this array
    #
    with open("/Users/mwornow/desktop/data/training.json") as json_file:
        files = json.load(json_file)
        for file in files:
            # Read the image
            file_path = file['image']['pathname']
            image_id = get_image_id_from_pathname_entry_in_json(file_path)

            if image_ids is not None:
                # Only show if image_id in image_ids
                if image_id not in image_ids:
                    continue

            # file_path = '/images/ffd59802-46c6-4b58-80fe-e534e39781a7.png'
            image = io.imread("/Users/mwornow/desktop/data" + file_path)

            # Set up plots
            f, ax = plt.subplots(2,2)
            f.suptitle("Image: " + file_path + " | Size: " + str(image.shape) + "\n# Cells: " + str(len(file['objects'])))

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

def is_point_in_box(point, box):
    y, x = point
    miny, minx, maxy, maxx = box
    if y >= miny and y <= maxy and x >= minx and x <= maxx:
        return True
    return False    
def is_overlap(box1, box2):
    miny, minx, maxy, maxx = box1
    if is_point_in_box((miny, minx), box2) or is_point_in_box((maxy, maxx), box2):
        return True
    return False

def explore_data():
    with open("/Users/mwornow/desktop/data/training.json") as json_file:
        files = json.load(json_file)
        class_counts = { key : 0 for key in CATEGORY_COLORS.keys() }
        n_cells_in_image = []
        cell_location_counter = np.zeros((1000,1000))
        n_clusters = [] # Number of clusters in each image
        n_clusters_2 = [] # Number of clusters in each image (with size >= 2)
        n_cluster_sizes = [] # Number of cells in each cluster
        n_cluster_sizes_2 = [] # Number of cells in each cluster (with size >= 2)
        for file in files:
            # Read the image
            img_objects = file['objects']
            total_y, total_x = file['image']['shape']['r'], file['image']['shape']['c']
            clusters = [] # Array of arrays
            for o in img_objects:
                label = o['category']
                min_y, min_x = o['bounding_box']['minimum']['r'], o['bounding_box']['minimum']['c']
                max_y, max_x = o['bounding_box']['maximum']['r'], o['bounding_box']['maximum']['c']
                class_counts[label] += 1
                min_scale_y = int(min_y/total_y * 1000)
                max_scale_y = int(max_y/total_y * 1000)
                min_scale_x  = int(min_x/total_x * 1000) 
                max_scale_x = int(max_x/total_x * 1000)
                cell_location_counter[ min_scale_y : max_scale_y, min_scale_x : max_scale_x] = cell_location_counter[min_scale_y : max_scale_y, min_scale_x : max_scale_x] + 1
                # Add it to a cluster if relevant
                is_existing_cluster = False
                box = (min_y, min_x, max_y, max_x)
                for c in clusters:
                    if is_existing_cluster: break
                    for b in c:
                        if is_overlap(b, box):
                            c.append(box)
                            is_existing_cluster = True
                            break
                if not is_existing_cluster:
                    clusters.append([box])
            n_clusters.append(len(clusters))
            n_clusters_2.append(len([ c for c in clusters if len(c) > 1]))
            n_cluster_sizes += [ len(c) for c in clusters  ]
            n_cluster_sizes_2 += [ len(c) for c in clusters if len(c) > 1  ]
            n_cells_in_image.append(len(img_objects))
        if False:
            # Plot bar chart of class densities
            total_class_count = sum([ x for x in class_counts.values() ])
            plt.bar(class_counts.keys(), [ c / total_class_count for c in class_counts.values() ], tick_label = list(class_counts.keys()))
            plt.title("Distribution of Cell Classes")
            plt.ylabel("Frequency")
            plt.show()
        if False:
            # Plot histogram of # of cells in each image
            plt.hist(n_cells_in_image, density = True, bins = 30)
            plt.title("Distribution of Number of Cells in Images")
            plt.ylabel("Frequency")
            plt.xlabel("Number of Cells in Image")
            plt.show()
        if False:
            # Plot heatmap of cell locations in images
            heatmap = plt.imshow(cell_location_counter, cmap='hot', interpolation = None)
            plt.title("Heatmap of Cell Locations in Images")
            plt.xlabel("Width")
            plt.ylabel("Height")
            plt.colorbar(heatmap)
            plt.show()
        if False:
            # Plot histogram of number of clusters in each image
            plt.hist(n_clusters, density = True, bins = 30)
            plt.title("Distribution of Number of Cell Clusters (of Size >= 1 Cells) in Each Image")
            plt.ylabel("Frequency")
            plt.xlabel("Number of Cell Clusters in Image")
            plt.show()
            plt.hist(n_clusters_2, density = True, bins = 30)
            plt.title("Distribution of Number of Cell Clusters (of Size >= 2 Cells) in Each Image")
            plt.ylabel("Frequency")
            plt.xlabel("Number of Cell Clusters in Image")
            plt.show()
        if False:
            # Plot histogram of number of cells in each cluster
            plt.hist(n_cluster_sizes, density = True, bins = 20)
            plt.title("Distribution of Number of Cells in Each Cell Cluster (of Size >= 1 Cells)")
            plt.ylabel("Frequency")
            plt.xlabel("Number of Cells in Cell Cluster")
            plt.show()
            plt.hist(n_cluster_sizes_2, density = True, bins = 20)
            plt.title("Distribution of Number of Cells in Each Cell Cluster (of Size >= 2 Cells)")
            plt.ylabel("Frequency")
            plt.xlabel("Number of Cells in Cell Cluster")
            plt.show()
            
display_image(['10be6380-cbbb-4886-8b9e-ff56b1710576'])
# explore_data()