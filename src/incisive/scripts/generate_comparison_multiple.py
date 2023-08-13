import os
from pathlib import Path

from joblib import Parallel, delayed

import cv2
from tqdm import tqdm

import yamlut.utils as utils
import yamlut.visualization as vis


root_dir = Path(os.getcwd()).parent

# exp_names = ["seg/exp-seg-test-s", "seg/exp-seg-test-l"]
exp_names = ["seg/exp-seg-test-s", "seg/exp-seg-test-l"]
exp_dirs = [root_dir / "data" / "exp" / exp for exp in exp_names]
annotations_dir = root_dir / "data" / "annotations" / "labels" / "test"
results_dir = root_dir / "data" / "results"


def process_image(file_name):
    images = []
    for dir in exp_dirs:
        images.append(utils.load_image(dir / f"{file_name}.png"))
    [h, w] = images[0].shape[:2]
    contours = utils.load_relative_contours(annotations_dir / f"{file_name}.txt")
    contours = [utils.r2a_contour(c, w, h) for c in contours]
    processed = []
    for img in images:
        annotated = img
        for c in contours:
            annotated = vis.draw_contour(annotated, c)
        processed.append(annotated)
    compound = vis.concat_h(processed)
    utils.store_image(compound, results_dir / f"{file_name}.png")


if __name__ == "__main__":

    file_names = utils.list_file_names(annotations_dir)

    try:
        os.mkdir(results_dir)
    except:
        pass

    # for f in tqdm(file_names):
    #     process_image(f)
    Parallel(n_jobs=4)(delayed(process_image)(f) for f in tqdm(file_names))
