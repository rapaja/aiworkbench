import os
from pathlib import Path
from rich import print
from joblib import Parallel, delayed
from tqdm import tqdm

import utils
import visualization as vis
import metrics as ym

import cv2


def do_process(mdl_ndx, preprocessing):
    detection_name = f"exp-incisive-{mdl_ndx}-{'y' if preprocessing else 'n'}"

    pp = "" if not preprocessing else "_pp"

    # data_dir = Path(os.getcwd()).parent / "results"
    data_dir = Path(os.getcwd()) / "src" / "utils" / "results" / "mg"

    images_dir = data_dir / ("images" + pp)
    target_cont_dir = data_dir / ("contours" + pp) / "test"

    predicted_annotations_dir = data_dir / "detect" / detection_name / "labels"

    def process_image(file_name):
        image = utils.load_image(images_dir / f"{file_name}.png")
        [h, w] = image.shape[:2]
        target_cont = utils.load_relative_contours(target_cont_dir / f"{file_name}.txt")
        target_cont = [utils.r2a_contour(r, w, h) for r in target_cont]

        try:
            predicted_boxes = utils.load_relative_rectangles(
                predicted_annotations_dir / f"{file_name}.txt"
            )
            predicted_boxes = [
                utils.r2a_rectangle(rect, w, h) for rect in predicted_boxes
            ]
        except:
            predicted_boxes = []

        target_mask = utils.create_mask_from_contours(target_cont, w, h)
        predicted_mask = utils.create_mask_from_rectangles(predicted_boxes, w, h)
        return ym.evaluate(target_mask, predicted_mask)

    file_names = utils.list_file_names(images_dir)
    file_names = [f for f in file_names if "_uns2_" in f]

    results = Parallel(n_jobs=4)(delayed(process_image)(f) for f in tqdm(file_names))
    # results = (process_image(f) for f in tqdm(file_names))

    total_tp, total_tn, total_fp, total_fn = 0, 0, 0, 0
    for tp, tn, fp, fn in results:
        total_tp += tp
        total_fp += fp
        total_tn += tn
        total_fn += fn

    print("")
    print("model: ", detection_name)
    print("using preprocessing: ", preprocessing)
    print("images dir: ", images_dir)
    print("tgt ann dir: ", target_cont_dir)
    print("prd ann dir: ", predicted_annotations_dir)
    # ---
    p, r, f1 = ym.p_r_f1(total_tp, total_tn, total_fp, total_fn)
    print(f"[bold blue]P[/]\t{p:.3f}")
    print(f"[bold blue]R[/]\t{r:.3f}")
    print(f"[bold blue]F1[/]\t{f1:.3f}")


mdl_index = 8
if __name__ == "__main__":
    do_process(mdl_index, False)
    do_process(mdl_index, True)
