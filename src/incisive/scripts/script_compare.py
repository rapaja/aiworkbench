import os
from pathlib import Path

import cv2
from tqdm import tqdm

import utils
import visualization as vis


root_dir = Path(os.getcwd()) / "src" / "utils" / "results"
img_root_dir = root_dir / "comparison"
image_dirs = [
    img_root_dir / img_dir for img_dir in ["exp-incisive-1-n", "exp-incisive-6-y"]
]
results_dir = root_dir

# root_dir = Path(os.getcwd()).parent

# exp_name = "bbox/exp-test-l-alt"
# exp_dir = root_dir / "data" / "exp" / exp_name
# annotations_dir = root_dir / "data" / "annotations" / "exp-test-s" / "test"
# results_dir = root_dir / "data" / "results" / exp_name


def process_image(file_name):
    images = [utils.load_image(img_dir / f"{file_name}.png") for img_dir in image_dirs]
    h = images[0].shape[0]
    reshaped_images = [images[0]]
    for img in images[1:]:
        resized = utils.resize_image(img, height=h)
        reshaped_images.append(resized)

    concat_img = vis.concat_h(reshaped_images)
    utils.store_image(concat_img, results_dir / f"{file_name}.png")


def list_file_names(dir):
    files = os.listdir(str(dir))
    files = [f for f in files if os.path.isfile(dir / f)]
    return [Path(f).stem for f in files]


if __name__ == "__main__":
    file_names = list_file_names(image_dirs[0])

    for f in tqdm(file_names):
        process_image(f)
