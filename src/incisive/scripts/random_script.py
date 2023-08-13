from pathlib import Path
import os
import shutil
from tqdm import tqdm

import utils

root_dir = Path(os.getcwd()) / "src" / "utils" / "results" / "mg"

img_dir = root_dir / "detection-selection" / "images"
ann_t_dir = root_dir / "detection-selection" / "labels"
ann_s_dir = root_dir / "labels"

file_names = utils.list_file_names(img_dir)
for f in tqdm(file_names):
    ann = ann_s_dir / (str(Path(f).stem) + ".txt")
    shutil.copy2(ann, ann_t_dir)
