# %%
import os
from pathlib import Path
from rich import print
from joblib import Parallel, delayed
from tqdm import tqdm

import utils
import visualization as vis
import metrics as ym

import matplotlib.pyplot as plt
import numpy as np

import cv2

# %%

detection_name = "exp-incisive-1-n"

# data_dir = Path(os.getcwd()).parent / "results"
data_dir = Path(os.getcwd()).parent / "results"

images_dir = data_dir / "images"
target_annotations_dir = data_dir / "labels"

predicted_annotations_dir = target_annotations_dir
# predicted_annotations_dir = data_dir / "detect" / detection_name / "labels"

# %%

file_names = utils.list_file_names(images_dir)

file_name = file_names[0]

# %%

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

predicted_boxes = [(0, (0.5, 0.5), (0.75, 0.75))]
predicted_boxes = [utils.r2a_rectangle(rect, w, h) for rect in predicted_boxes]

target_mask = utils.create_mask_from_rectangles(target_boxes, w, h)
predicted_mask = utils.create_mask_from_rectangles(predicted_boxes, w, h)

# %%

plt.subplot(1, 2, 1)
plt.imshow(predicted_mask)
plt.subplot(1, 2, 2)
plt.imshow(target_mask)

# %%

# return ym.evaluate(target_mask, predicted_mask)

agreed = target_mask == predicted_mask
tp = predicted_mask & agreed
tn = ~predicted_mask & agreed
fp = predicted_mask & ~agreed
fn = ~predicted_mask & ~agreed

# return int(np.sum(tp)), int(np.sum(tn)), int(np.sum(fp)), int(np.sum(fn))

# %%

plt.imshow(fn)

# %%

int(np.sum(tp)), int(np.sum(tn)), int(np.sum(fp)), int(np.sum(fn))

# %%

if __name__ == "__main__":
    file_names = utils.list_file_names(images_dir)

    file_names = [file_names[0]]
    # results = Parallel(n_jobs=4)(delayed(process_image)(f) for f in tqdm(file_names))
    results = (process_image(f) for f in tqdm(file_names))

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
