import os
from pathlib import Path

import cv2
from tqdm import tqdm

import utils
import visualization as vis

detection_name = "exp-incisive-1-n"

root_dir = Path(os.getcwd()).parent / "results"
images_dir = root_dir / "images"
target_annotations_dir = root_dir / "labels"
predicted_annotations_dir = root_dir / "detect" / detection_name / "labels"
results_dir = root_dir / "comparison"

root_dir = Path(os.getcwd()).parent

exp_name = "bbox/exp-test-l-alt"
exp_dir = root_dir / "data" / "exp" / exp_name
annotations_dir = root_dir / "data" / "annotations" / "exp-test-s" / "test"
results_dir = root_dir / "data" / "results" / exp_name


def process_image(file_name):
    image = utils.load_image(exp_dir / f"{file_name}.png")
    [h, w] = image.shape[:2]
    contours = utils.load_relative_contours(annotations_dir / f"{file_name}.txt")
    contours = [utils.r2a_contour(c, w, h) for c in contours]
    annotated = image
    for c in contours:
        annotated = vis.draw_contour(annotated, c)
    utils.store_image(annotated, results_dir / f"{file_name}.png")


def list_file_names(dir):
    files = os.listdir(str(dir))
    files = [f for f in files if os.path.isfile(dir / f)]
    return [Path(f).stem for f in files]


if __name__ == "__main__":
    file_names = list_file_names(exp_dir)

    os.mkdir(results_dir)
    for f in tqdm(file_names):
        process_image(f)
