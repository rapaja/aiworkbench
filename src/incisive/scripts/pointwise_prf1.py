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

import cv2

USE_CONTOURS_PREDICTIONS = True
USE_CONTOURS_ANNOTATIONS = True


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

detection_name_fcn = (
    lambda mdl_ndx, preprocessing: f"exp-no-incisive-{mdl_ndx}-inc-{'y' if preprocessing else 'n'}"
)


def do_process(mdl_ndx, preprocessing, verbose=True):
    detection_name = detection_name_fcn(mdl_ndx, preprocessing)

    DATA_DIR = Path(os.getcwd()) / "src" / "utils" / "results" / "mg"
    FILENAME_FILTER_PATTERN = None
    PP = "" if not preprocessing else "_pp"
    IMAGES_DIR = DATA_DIR / ("images" + PP)
    TARGET_ANN_DIR = DATA_DIR / ("labels" + PP)
    PREDICTED_ANN_DIR = DATA_DIR / "detect" / detection_name / "labels"

    def process_image(file_name):
        image = utils.load_image(IMAGES_DIR / f"{file_name}.png")
        [h, w] = image.shape[:2]
        target_ann = t_fcn.load_relative(TARGET_ANN_DIR / f"{file_name}.txt")
        target_ann = [t_fcn.r2a(r, w, h) for r in target_ann]

        try:
            predicted_ann = p_fcn.load_relative(PREDICTED_ANN_DIR / f"{file_name}.txt")
            predicted_ann = [p_fcn.r2a(rect, w, h) for rect in predicted_ann]
        except:
            predicted_ann = []

        target_mask = t_fcn.create_mask(target_ann, w, h)
        predicted_mask = p_fcn.create_mask(predicted_ann, w, h)
        return ym.evaluate(target_mask, predicted_mask)

    file_names = utils.list_file_names(IMAGES_DIR)
    if FILENAME_FILTER_PATTERN:
        file_names = [f for f in file_names if FILENAME_FILTER_PATTERN in f]

    results = Parallel(n_jobs=4)(delayed(process_image)(f) for f in tqdm(file_names))
    # results = (process_image(f) for f in tqdm(file_names))

    total_tp, total_tn, total_fp, total_fn = 0, 0, 0, 0
    for tp, tn, fp, fn in results:
        total_tp += tp
        total_fp += fp
        total_tn += tn
        total_fn += fn

    p, r, f1 = ym.p_r_f1(total_tp, total_tn, total_fp, total_fn)

    if verbose:
        print("")
        print("model: ", detection_name)
        print("using preprocessing: ", preprocessing)
        print("images dir: ", IMAGES_DIR)
        print("tgt ann dir: ", TARGET_ANN_DIR)
        print("prd ann dir: ", PREDICTED_ANN_DIR)
        # ---
        print(f"[bold blue]P[/]\t{p:.3f}")
        print(f"[bold blue]R[/]\t{r:.3f}")
        print(f"[bold blue]F1[/]\t{f1:.3f}")

    return p, r, f1


if __name__ == "__main__":
    with open("results.txt", "a") as f:
        f.write("========\n")

    for ndx in range(6):
        for pp in [False, True]:
            p, r, f1 = do_process(ndx + 1, pp, verbose=False)
            with open("results.txt", "a") as f:
                line = ",".join([str(x) for x in [ndx + 1, pp, p, r, f1]])
                f.write(line + "\n")
