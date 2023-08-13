import os
from pathlib import Path

from rich import print

from joblib import Parallel, delayed

import cv2
from tqdm import tqdm

from . import utils
from . import visualizations as vis
from . import metrics as ym

import numpy as np

data_dir = Path(os.getcwd()).parent / "results"

exp_dir = data_dir / "exp" / "bbox" / "exp-test-s"
target_contours_dir = data_dir / "annotations" / "labels" / "test"
predicted_annotations_dir = data_dir / "exp" / "seg" / "labels" / "labels-l"


def tp_overlap_perc(baseline_mask, other_mask):
    h, w = baseline_mask.shape[:2]
    total_target_area = np.sum(baseline_mask) / 255
    relative_target_area = total_target_area / (h * w)
    common_area = np.sum((baseline_mask == other_mask) & (baseline_mask == 255))
    if total_target_area > 0:
        return common_area / total_target_area, relative_target_area


def process_image(file_name):
    image = utils.load_image(exp_dir / f"{file_name}.png")
    [h, w] = image.shape[:2]

    target_contours = utils.load_relative_contours(
        target_contours_dir / f"{file_name}.txt"
    )
    target_contours = [utils.r2a_contour(r, w, h) for r in target_contours]

    try:
        predicted_boxes = utils.load_relative_rectangles(
            predicted_annotations_dir / f"{file_name}.txt"
        )
        predicted_boxes = [utils.r2a_rectangle(r, w, h) for r in predicted_boxes]
    except:
        predicted_boxes = []

    # per lesion
    predicted_mask = utils.create_mask_from_rectangles(predicted_boxes, w, h)
    target_masks = [
        utils.create_mask_from_contours([tc], w, h) for tc in target_contours
    ]
    lesions_coverage = [tp_overlap_perc(tm, predicted_mask) for tm in target_masks]
    L_percentages = [(o, a) for o, a in lesions_coverage if a >= 1e-4]
    L = [o for o, _ in L_percentages]

    # per rectangle
    predicted_masks = [
        utils.create_mask_from_rectangles([pb], w, h) for pb in predicted_boxes
    ]
    target_mask = utils.create_mask_from_contours(target_contours, w, h)
    rectangles_coverage = [tp_overlap_perc(pm, target_mask) for pm in predicted_masks]
    R_percentages = [(o, a) for o, a in rectangles_coverage if a >= 1e-4]
    R = [o for o, _ in R_percentages]

    return L, R


def compute_lr(target_masks, predicted_masks, w, h):
    # computing L
    total_predicted_mask = np.zeros(shape=(h, w), dtype=np.uint8)
    for pm in predicted_masks:
        total_predicted_mask = 255 * ((pm == 255) | (total_predicted_mask == 255))
    lesions_coverage = [
        tp_overlap_perc(tm, total_predicted_mask) for tm in target_masks
    ]
    l_ratios = [overlap for overlap, area in lesions_coverage if area >= 1e-4]

    # computing R
    total_target_mask = np.zeros(shape=(h, w), dtype=np.uint8)
    for tm in target_masks:
        total_target_mask = 255 * ((tm == 255) | (total_target_mask == 255))
    rectangles_coverage = [
        tp_overlap_perc(pm, total_target_mask) for pm in predicted_masks
    ]
    r_ratios = [overlap for overlap, area in rectangles_coverage if area >= 1e-4]

    return l_ratios, r_ratios


def threshold_metrics(arr, th=0.1):
    return sum([1 for a in arr if a >= th]) / len(arr)


def fancy_metrics(arr, th=1 / 3):
    return np.sum(np.tanh(np.array(arr) / th)) / len(arr)


def evaluate(l_ratios, r_ratios, metrics):
    l = metrics(l_ratios)
    r = metrics(r_ratios)
    f = 2 * l * r / (l + r)
    return l, r, f


def process_image2(file_name):
    image = utils.load_image(exp_dir / f"{file_name}.png")
    [h, w] = image.shape[:2]

    target_contours = utils.load_relative_contours(
        target_contours_dir / f"{file_name}.txt"
    )
    target_contours = [utils.r2a_contour(r, w, h) for r in target_contours]

    try:
        predicted_boxes = utils.load_relative_rectangles(
            predicted_annotations_dir / f"{file_name}.txt"
        )
        predicted_boxes = [utils.r2a_rectangle(r, w, h) for r in predicted_boxes]
    except:
        predicted_boxes = []

    target_masks = [
        utils.create_mask_from_contours([tc], w, h) for tc in target_contours
    ]
    predicted_masks = [
        utils.create_mask_from_rectangles([pb], w, h) for pb in predicted_boxes
    ]

    l_ratios, r_ratios = compute_lr(target_masks, predicted_masks, w, h)

    return l_ratios, r_ratios


def process_image3(file_name):
    image = utils.load_image(exp_dir / f"{file_name}.png")
    [h, w] = image.shape[:2]

    target_contours = utils.load_relative_contours(
        target_contours_dir / f"{file_name}.txt"
    )
    target_contours = [utils.r2a_contour(r, w, h) for r in target_contours]

    try:
        predicted_contours = utils.load_relative_contours(
            predicted_annotations_dir / f"{file_name}.txt"
        )
        predicted_contours = [utils.r2a_contour(r, w, h) for r in predicted_contours]
    except:
        predicted_contours = []

    target_masks = [
        utils.create_mask_from_contours([tc], w, h) for tc in target_contours
    ]
    predicted_masks = [
        utils.create_mask_from_contours([pb], w, h) for pb in predicted_contours
    ]

    l_ratios, r_ratios = compute_lr(target_masks, predicted_masks, w, h)

    return l_ratios, r_ratios


def main1():
    file_names = utils.list_file_names(exp_dir)
    # results = Parallel(n_jobs=4)(delayed(process_image)(f) for f in tqdm(file_names))
    results = [process_image(f) for f in tqdm(file_names)]
    L_total = []
    R_total = []
    for l, r in results:
        L_total.extend(l)
        R_total.extend(r)

    # print(L_total)
    # print(R_total)

    # print(f"L1: {len(L_total)}")
    # print(f"L1: {len(R_total)}")
    #
    # print(L_total)
    # print(R_total)

    L1 = threshold_metrics(L_total, th=0.1)
    R1 = threshold_metrics(R_total, th=0.1)

    # L1 = fancy_metrics(L_total, th=0.2)
    # R1 = fancy_metrics(R_total, th=0.2)

    FFF1 = 2 * L1 * R1 / (L1 + R1)

    # L1 = threshold_metrics([l for l in L_total if l >= 1e-12], th=0.1)
    # R1 = threshold_metrics(R_total, th=0.1)

    # total_tp, total_tn, total_fp, total_fn = 0, 0, 0, 0
    # for tp, tn, fp, fn in results:
    #     total_tp += tp
    #     total_fp += fp
    #     total_tn += tn
    #     total_fn += fn
    #
    # p, r, f1 = ym.p_r_f1(total_tp, total_tn, total_fp, total_fn)
    print(f"[bold blue]L1[/]\t{L1}")
    print(f"[bold blue]R1[/]\t{R1}")
    print(f"[bold blue]FF1[/]\t{FFF1}")


def main2():
    file_names = utils.list_file_names(exp_dir)
    # results = Parallel(n_jobs=4)(delayed(process_image3)(f) for f in tqdm(file_names))
    results = [process_image3(f) for f in tqdm(file_names)]
    l_total = []
    r_total = []
    for l, r in results:
        l_total.extend(l)
        r_total.extend(r)

    l, r, f = evaluate(l_total, r_total, threshold_metrics)

    print(f"[bold blue]Lm[/]\t{l}")
    print(f"[bold blue]Rm[/]\t{r}")
    print(f"[bold blue]Fm[/]\t{f}")


if __name__ == "__main__":
    main2()
