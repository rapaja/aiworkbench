import os
from pathlib import Path
import numpy as np

import cv2
from tqdm import tqdm

import utils
import visualization as vis


from joblib import Parallel, delayed

mdl_ndx = 1
preprocessing = False

detection_name = f"exp-mri-{mdl_ndx}-test-full"
pp = "_pp" if preprocessing else ""

root_dir = Path(os.getcwd()) / "src" / "incisive" / "results"
images_dir = root_dir / "mri" / "test_images_015"
target_bbox_dir = root_dir / "mri" / "test_labels_full"
target_cont_dir = root_dir / "mg" / ("contours" + pp)
predicted_ann_dir = root_dir / "mri" / detection_name / "labels"
results_dir = root_dir / "comparison"

FILTER_PATTERN = None

TARGET_COLOR = (255, 0, 0)
PRED_COLOR = (0, 0, 255)
WIDTH = 3
OFFSET = 40
C_WIDTH = 3


def process_image(file_name):
    image = utils.load_image(images_dir / f"{file_name}.png")
    [h, w] = image.shape[:2]
    annotated = image
    annotated_c = image.copy()
    # TARGET BBOX
    try:
        rectangles = utils.load_relative_rectangles(
            target_bbox_dir / f"{file_name}.txt"
        )
        rectangles = [utils.r2a_rectangle(r, w, h) for r in rectangles]
    except:
        rectangles = []
    for r in rectangles:
        annotated = vis.draw_rectangle(annotated, r, color=TARGET_COLOR, width=WIDTH)
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
    #     annotated_c = vis.draw_contour(
    #         annotated_c, c, color=TARGET_COLOR, width=C_WIDTH
    #     )
    # annotated = vis.blend_contours(annotated, contours, fill=TARGET_COLOR)
    # PREDICTIONS
    try:
        rectangles = utils.load_relative_rectangles(
            predicted_ann_dir / f"{file_name}.txt"
        )
        rectangles = [utils.r2a_rectangle(r, w, h) for r in rectangles]
    except:
        rectangles = []
    for r in rectangles:
        annotated = vis.draw_rectangle(annotated, r, color=PRED_COLOR, width=WIDTH)
        annotated_c = vis.draw_rectangle(
            annotated_c, r, color=PRED_COLOR, width=C_WIDTH
        )
    # --
    # annotated = utils.resize_image(annotated, height=640)
    # res_dir = results_dir / file_name
    # os.mkdir(res_dir)
    # utils.store_image(annotated, res_dir / f"{file_name}.png")
    utils.store_image(annotated, results_dir / f"{file_name}.png")
    # ZOOM RECTANGLE
    # for i, (_, (c1, r1), (c2, r2)) in enumerate(rectangles):
    #     c1 = max(0, c1 - OFFSET)
    #     r1 = max(0, r1 - OFFSET)
    #     c2 = min(w, c2 + OFFSET)
    #     r2 = min(h, r2 + OFFSET)
    #     crop = annotated_c[r1:r2, c1:c2, :]
    #     utils.store_image(crop, res_dir / f"crop-{i}.png")
    # ZOOM CONTOUR
    # for i, (_, pts) in enumerate(rectangles):
    #     c1 = np.min([x for (x, _) in pts])
    #     c2 = np.max([x for (x, _) in pts])
    #     r1 = np.min([y for (_, y) in pts])
    #     r2 = np.max([y for (_, y) in pts])
    #     c1 = max(0, c1 - OFFSET)
    #     r1 = max(0, r1 - OFFSET)
    #     c2 = min(w, c2 + OFFSET)
    #     r2 = min(h, r2 + OFFSET)
    #     crop = annotated_c[r1:r2, c1:c2, :]
    #     utils.store_image(crop, res_dir / f"crop-{i}.png")


def list_file_names(dir):
    files = os.listdir(str(dir))
    files = [f for f in files if os.path.isfile(dir / f)]
    return [Path(f).stem for f in files]


if __name__ == "__main__":
    file_names = list_file_names(images_dir)
    if FILTER_PATTERN:
        file_names = [f for f in file_names if FILTER_PATTERN in f]

    # try:
    #     os.mkdir(results_dir)
    # except:
    #     pass

    Parallel(n_jobs=4)(delayed(process_image)(f) for f in tqdm(file_names))

    # for f in tqdm(file_names):
    #     process_image(f)
