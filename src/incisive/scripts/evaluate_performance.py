import os
from pathlib import Path
from rich import print
from joblib import Parallel, delayed
from tqdm import tqdm
from dataclasses import dataclass
from typing import Callable

import utils
import visualization as vis
import metrics as ym

import numpy as np
from tqdm import trange

import cv2

USE_CONTOURS_PREDICTIONS = False
USE_CONTOURS_ANNOTATIONS = False


@dataclass
class FunctionSet:
    load_relative: Callable = None
    r2a: Callable = None
    create_mask: Callable = None


def choose_function_set(use_contours: bool) -> FunctionSet:
    if use_contours:
        return FunctionSet(
            utils.load_relative_contours,
            utils.r2a_contour,
            utils.create_mask_from_contours,
        )
    else:
        return FunctionSet(
            utils.load_relative_rectangles,
            utils.r2a_rectangle,
            utils.create_mask_from_rectangles,
        )


p_fcn = choose_function_set(use_contours=False)
t_fcn = choose_function_set(use_contours=False)


FILENAME_FILTER_PATTERN = None
DATA_DIR = Path(os.getcwd()) / "src" / "utils" / "results" / "mri"
USE_PARALLEL_PROCESSING = True

detection_name_fcn = lambda mdl_ndx, preprocessing: f"exp-mri-{mdl_ndx}-test"


def dirs_fcn(use_pp: bool, detection_name) -> tuple[Path, Path, Path]:
    pp = "" if not use_pp else "_pp"
    return (
        # images dir
        DATA_DIR / ("test_images" + pp),
        # targets dir
        DATA_DIR / ("test_labels" + pp),
        # predictions dir
        DATA_DIR / detection_name / "labels",
    )


def do_process(mdl_ndx, use_pp, verbose=True):
    detection_name = detection_name_fcn(mdl_ndx, use_pp)
    img_dir, t_ann_dir, p_ann_dir = dirs_fcn(use_pp, detection_name)

    def process_image(file_name):
        image = utils.load_image(img_dir / f"{file_name}.png")
        [h, w] = image.shape[:2]
        t_ann = t_fcn.load_relative(t_ann_dir / f"{file_name}.txt")
        t_ann = [t_fcn.r2a(r, w, h) for r in t_ann]

        try:
            p_ann = p_fcn.load_relative(p_ann_dir / f"{file_name}.txt")
            p_ann = [p_fcn.r2a(rect, w, h) for rect in p_ann]
        except:
            p_ann = []

        t_mask = t_fcn.create_mask(t_ann, w, h)
        p_mask = p_fcn.create_mask(p_ann, w, h)
        return ym.evaluate(t_mask, p_mask)

    def process_results(results, verbose):
        total_tp, total_tn, total_fp, total_fn = 0, 0, 0, 0
        for tp, tn, fp, fn in results:
            total_tp += tp
            total_fp += fp
            total_tn += tn
            total_fn += fn

        p, r, f1 = ym.p_r_f1(total_tp, total_tn, total_fp, total_fn)

        if verbose:
            print(f"[bold blue]P[/]\t{p:.3f}")
            print(f"[bold blue]R[/]\t{r:.3f}")
            print(f"[bold blue]F1[/]\t{f1:.3f}")

        return p, r, f1

    if verbose:
        print("")
        print("model: ", detection_name)
        print("using preprocessing: ", use_pp)
        print("images dir: ", img_dir)
        print("tgt ann dir: ", t_ann_dir)
        print("prd ann dir: ", p_ann_dir)

    f_names = utils.list_file_names(img_dir)
    if FILENAME_FILTER_PATTERN:
        f_names = [f for f in f_names if FILENAME_FILTER_PATTERN in f]

    r_data = tqdm(f_names, leave=False)
    if USE_PARALLEL_PROCESSING:
        results = Parallel(n_jobs=4)(delayed(process_image)(f) for f in r_data)
    else:
        results = (process_image(f) for f in r_data)

    p, r, f1 = process_results(results, verbose)

    return p, r, f1


def do_process_niksa(mdl_ndx, use_pp, th=0.5, verbose=True):
    detection_name = detection_name_fcn(mdl_ndx, use_pp)
    img_dir, t_ann_dir, p_ann_dir = dirs_fcn(use_pp, detection_name)

    def tp_overlap_perc(baseline_mask, other_mask):
        h, w = baseline_mask.shape[:2]
        total_target_area = np.sum(baseline_mask) / 255
        relative_target_area = total_target_area / (h * w)
        common_area = np.sum((baseline_mask == other_mask) & (baseline_mask == 255))
        if total_target_area > 0:
            return common_area / total_target_area, relative_target_area
        else:
            return 0, 0

    def evaluate(l_ratios, r_ratios, metrics):
        l = metrics(l_ratios)
        r = metrics(r_ratios)
        f = 2 * l * r / (l + r)
        return l, r, f

    def threshold_metrics(arr, th=0.1):
        return sum([1 for a in arr if a >= th]) / len(arr)

    def process_image(file_name):
        image = utils.load_image(img_dir / f"{file_name}.png")
        [h, w] = image.shape[:2]
        target_ann = t_fcn.load_relative(t_ann_dir / f"{file_name}.txt")
        target_ann = [t_fcn.r2a(r, w, h) for r in target_ann]

        try:
            predicted_ann = p_fcn.load_relative(p_ann_dir / f"{file_name}.txt")
            predicted_ann = [p_fcn.r2a(rect, w, h) for rect in predicted_ann]
        except:
            predicted_ann = []

        # per lesion
        predicted_mask = p_fcn.create_mask(predicted_ann, w, h)
        target_masks = [t_fcn.create_mask([tc], w, h) for tc in target_ann]
        lesions_coverage = [tp_overlap_perc(tm, predicted_mask) for tm in target_masks]
        L_percentages = [(o, a) for o, a in lesions_coverage if a >= 1e-4]
        L = [o for o, _ in L_percentages]

        # per rectangle
        predicted_masks = [p_fcn.create_mask([pb], w, h) for pb in predicted_ann]
        target_mask = t_fcn.create_mask(target_ann, w, h)
        rectangles_coverage = [
            tp_overlap_perc(pm, target_mask) for pm in predicted_masks
        ]
        R_percentages = [(o, a) for o, a in rectangles_coverage if a >= 1e-4]
        R = [o for o, _ in R_percentages]

        return L, R

    file_names = utils.list_file_names(img_dir)
    if FILENAME_FILTER_PATTERN:
        file_names = [f for f in file_names if FILENAME_FILTER_PATTERN in f]

    results = Parallel(n_jobs=4)(
        delayed(process_image)(f) for f in tqdm(file_names, leave=False)
    )
    # results = (process_image(f) for f in tqdm(file_names))

    l_total = []
    r_total = []
    for l, r in results:
        l_total.extend(l)
        r_total.extend(r)

    mtr = lambda x: threshold_metrics(x, th=th)

    l, r, f1 = evaluate(l_total, r_total, mtr)

    if verbose:
        print("")
        print("model: ", detection_name)
        print("using preprocessing: ", use_pp)
        print("images dir: ", img_dir)
        print("tgt ann dir: ", t_ann_dir)
        print("prd ann dir: ", p_ann_dir)
        # ---
        print(f"[bold blue]P[/]\t{r:.3f}")
        print(f"[bold blue]R[/]\t{l:.3f}")
        print(f"[bold blue]F1[/]\t{f1:.3f}")

    return r, l, f1


if __name__ == "__main__":
    with open("results.txt", "a") as f:
        f.write("========\n")

    for ndx in trange(3):
        res = do_process_niksa(ndx + 1, None, verbose=False)
        with open("results.txt", "a") as f:
            log = [ndx + 1]
            log.extend(res)
            line = ",".join([str(x) for x in log])
            f.write(line + "\n")
