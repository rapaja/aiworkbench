import os
from pathlib import Path

from rich import print

from joblib import Parallel, delayed

import cv2
from tqdm import tqdm

from . import utils
from . import visualization as vis
from . import metrics as ym

data_dir = Path(os.getcwd()).parent / "data"
images_dir = data_dir / "exp" / "bbox" / "exp-test-s"
target_annotations_dir = data_dir / "annotations" / "labels" / "test"
predicted_annotations_dir = data_dir / "exp" / "seg" / "labels" / "labels-l"
# results_dir = data_dir / "results"


def process_image(file_name):
    image = utils.load_image(images_dir / f"{file_name}.png")
    [h, w] = image.shape[:2]
    target_boxes = utils.load_relative_rectangles(
        target_annotations_dir / f"{file_name}.txt"
    )
    target_boxes = [utils.r2a_rectangle(r, w, h) for r in target_boxes]

    try:
        predicted_boxes = utils.load_relative_rectangles(
            predicted_annotations_dir / f"{file_name}.txt"
        )
        predicted_boxes = [utils.r2a_rectangle(rect, w, h) for rect in predicted_boxes]
    except:
        predicted_boxes = []

    target_mask = utils.create_mask_from_rectangles(target_boxes, w, h)
    predicted_mask = utils.create_mask_from_rectangles(predicted_boxes, w, h)

    return ym.evaluate(target_mask, predicted_mask)


def process_image_cnt(file_name):
    image = utils.load_image(images_dir / f"{file_name}.png")
    [h, w] = image.shape[:2]
    target_cnt = utils.load_relative_contours(
        target_annotations_dir / f"{file_name}.txt"
    )
    target_cnt = [utils.r2a_contour(r, w, h) for r in target_cnt]

    try:
        predicted_cnt = utils.load_relative_contours(
            predicted_annotations_dir / f"{file_name}.txt"
        )
        predicted_cnt = [utils.r2a_contour(rect, w, h) for rect in predicted_cnt]
    except:
        predicted_cnt = []

    target_mask = utils.create_mask_from_contours(target_cnt, w, h)
    predicted_mask = utils.create_mask_from_contours(predicted_cnt, w, h)

    return ym.evaluate(target_mask, predicted_mask)


if __name__ == "__main__":
    file_names = utils.list_file_names(images_dir)

    # try:
    #     os.mkdir(results_dir)
    # except:
    #     pass

    # for f in tqdm(file_names):
    #     process_image(f)
    # results = Parallel(n_jobs=4)(delayed(process_image_cnt)(f) for f in tqdm(file_names))
    results = (process_image_cnt(f) for f in tqdm(file_names))

    total_tp, total_tn, total_fp, total_fn = 0, 0, 0, 0
    for tp, tn, fp, fn in results:
        total_tp += tp
        total_fp += fp
        total_tn += tn
        total_fn += fn

    p, r, f1 = ym.p_r_f1(total_tp, total_tn, total_fp, total_fn)
    print(f"[bold blue]P[/]\t{p}")
    print(f"[bold blue]R[/]\t{r}")
    print(f"[bold blue]F1[/]\t{f1}")
