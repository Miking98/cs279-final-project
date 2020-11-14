import timeit
import processing.preprocessing as pre
import json
from skimage import io

path_to_data = "D:\Workspace\CS279\data"

cell_types = {}

with open(path_to_data + "/test.json") as json_file:
    files = json.load(json_file)
    i = 1
    start_time = timeit.default_timer()
    for file in files:
        # read the image
        file_path = file['image']['pathname']
        if i > 1:
            elapsed_time = (timeit.default_timer() - start_time)
            print(f"\tapprox. remaining time: {elapsed_time * 120 / (i - 1) - elapsed_time}")
        print(f"{i}:\tprocessing file: {file_path}")
        i += 1

        image = io.imread(path_to_data + file_path)
        processed = pre.apply_pre_processing(image, background_rem=True, morph_filter=True)

        path = f"{path_to_data}\\new_clustering{file_path}"
        io.imsave(path, processed)
