import os
from pathlib import Path

import cv2
from tqdm import tqdm

import utils
import visualization as vis


detection_name = "test_detections"

root_dir = Path(os.getcwd()) / "src" / "utils" / "results"
images_dir = root_dir / "mri" / "test_images"
target_bbox_dir = root_dir / "mri" / "test_labels"
predicted_bbox_dir = root_dir / "mri" / "test_detections"
results_dir = root_dir / "comparison" / "mri_test"

TARGET_COLOR = (255, 0, 0)
PRED_COLOR = (0, 0, 255)
WIDTH = 3

# root_dir = Path(os.getcwd()).parent

# exp_name = "bbox/exp-test-l-alt"
# exp_dir = root_dir / "data" / "exp" / exp_name
# annotations_dir = root_dir / "data" / "annotations" / "exp-test-s" / "test"
# results_dir = root_dir / "data" / "results" / exp_name


def process_image(file_name):
    image = utils.load_image(images_dir / f"{file_name}.png")
    [h, w] = image.shape[:2]
    annotated = image
    # TARGET BBOX
    rectangles = utils.load_relative_rectangles(target_bbox_dir / f"{file_name}.txt")
    rectangles = [utils.r2a_rectangle(r, w, h) for r in rectangles]
    for r in rectangles:
        annotated = vis.draw_rectangle(annotated, r, color=TARGET_COLOR, width=1)
    annotated = vis.blend_rect(annotated, rectangles, fill=TARGET_COLOR)
    # TARGET CONT
    # try:
    #     contours = utils.load_relative_contours(
    #         target_cont_dir / "test" / f"{file_name}.txt"
    #     )
    #     contours = [utils.r2a_contour(c, w, h) for c in contours]
    # except:
    #     contours = []
    # for c in contours:
    #     annotated = vis.draw_contour(annotated, c, color=TARGET_COLOR, width=WIDTH)
    # PREDICTIONS
    try:
        rectangles = utils.load_relative_rectangles(
            predicted_bbox_dir / f"{file_name}.txt"
        )
        rectangles = [utils.r2a_rectangle(r, w, h) for r in rectangles]
    except:
        rectangles = []
    for r in rectangles:
        annotated = vis.draw_rectangle(annotated, r, color=PRED_COLOR, width=WIDTH)
    # annotated = vis.blend_rect(annotated, rectangles, PRED_COLOR)
    # --
    annotated = utils.resize_image(annotated, height=640)
    utils.store_image(annotated, results_dir / f"{file_name}.png")


def list_file_names(dir):
    files = os.listdir(str(dir))
    files = [f for f in files if os.path.isfile(dir / f)]
    return [Path(f).stem for f in files]


if __name__ == "__main__":
    file_names = list_file_names(images_dir)

    # try:
    #     os.mkdir(results_dir)
    # except:
    #     pass

    for f in tqdm(file_names):
        process_image(f)
        # break
