import os
from pathlib import Path

from rich import print

from joblib import Parallel, delayed

import cv2
from tqdm import tqdm

import utils
import visualization as vis
import metrics as ym

import numpy as np

mdl_ndx = 7
TH = 0.5

for preprocessing in (False, True):
    detection_name = f"exp-incisive-{mdl_ndx}-{'y' if preprocessing else 'n'}"

    pp = "" if not preprocessing else "_pp"

    # data_dir = Path(os.getcwd()).parent / "results"
    data_dir = Path(os.getcwd()) / "src" / "utils" / "results" / "mg"

    images_dir = data_dir / ("images" + pp)
    target_annotations_dir = data_dir / ("contours" + pp) / "test"

    predicted_annotations_dir = data_dir / "detect" / detection_name / "labels"

    def tp_overlap_perc(baseline_mask, other_mask):
        h, w = baseline_mask.shape[:2]
        total_target_area = np.sum(baseline_mask) / 255
        relative_target_area = total_target_area / (h * w)
        common_area = np.sum((baseline_mask == other_mask) & (baseline_mask == 255))
        if total_target_area > 0:
            return common_area / total_target_area, relative_target_area
        else:
            return 0, 0

    def process_image(file_name):
        image = utils.load_image(images_dir / f"{file_name}.png")
        [h, w] = image.shape[:2]

        target_boxes = utils.load_relative_contours(
            target_annotations_dir / f"{file_name}.txt"
        )
        target_boxes = [utils.r2a_contour(r, w, h) for r in target_boxes]

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
            utils.create_mask_from_contours([tc], w, h) for tc in target_boxes
        ]
        lesions_coverage = [tp_overlap_perc(tm, predicted_mask) for tm in target_masks]
        L_percentages = [(o, a) for o, a in lesions_coverage if a >= 1e-4]
        L = [o for o, _ in L_percentages]

        # per rectangle
        predicted_masks = [
            utils.create_mask_from_rectangles([pb], w, h) for pb in predicted_boxes
        ]
        target_mask = utils.create_mask_from_contours(target_boxes, w, h)
        rectangles_coverage = [
            tp_overlap_perc(pm, target_mask) for pm in predicted_masks
        ]
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

    def main():
        print("")
        print("model: ", detection_name)
        print("images dir: ", images_dir)
        print("tgt ann dir: ", target_annotations_dir)
        print("prd ann dir: ", predicted_annotations_dir)
        print("th: ", TH)

        file_names = utils.list_file_names(images_dir)
        # file_names = [f for f in file_names if "_uns2_" in f]
        # results = Parallel(n_jobs=4)(delayed(process_image)(f) for f in tqdm(file_names))
        results = [process_image(f) for f in tqdm(file_names)]
        l_total = []
        r_total = []
        for l, r in results:
            l_total.extend(l)
            r_total.extend(r)

        mtr = lambda x: threshold_metrics(x, th=TH)

        l, r, f = evaluate(l_total, r_total, mtr)
        # ---
        print("")
        print(preprocessing, f"P={r:.3f}", f"R={l:.3f}", f"F1={f:.3f}")

    main()
