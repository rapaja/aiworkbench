#%% Init
import pathlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(".").resolve()
PATH_IMG = 'path_img'
PATH_ANN = 'path_box_txt'


#%% Load data

# cbis_ddsm
# csaws
# inbreast
# incisive -- samo za tesiranje
# mias

# VINDR
# ---------
# data_file_name = "vindr"
# get_img_key = lambda r: (r['study'], r['series'])
# get_img_name = lambda s: Path(s).stem

# CBIS_DDSM
# ---------
# Unclear what is image name. Have duplicate images.
# ---------
data_file_name = "cbis_ddsm"
get_img_key = lambda r: (r['subject'])
get_img_name = lambda s: f"_{Path(s).parent}"

# INBREAST
# ---------
# No annotation data
# ---------
# data_file_name = "inbreast"
# get_img_key = lambda r: r['subject']
# get_img_name = lambda s: Path(s).stem

# CSAW
# ----
# data_file_name = "csaws"
# get_img_key = lambda r: Path(r[PATH_IMG]).stem.split("_")[0]
# get_img_name = lambda s: Path(s).stem

# INCISIVE
# ---------
# data_file_name = "incisive"
# get_img_key = lambda r: r['subject']
# get_img_name = lambda s: f"{Path(s).parent.parent.parent}_{Path(s).parent}_{Path(s).stem}"

# MIAS
# ---------
# Possible duplicates
# ---------
# data_file_name = "mias"
# get_img_key = lambda r: r['subject']
# get_img_name = lambda s: Path(s).stem


data = pd.read_csv(ROOT / f"datasets/{data_file_name}.csv")
data.info()

#%% Filter data

# Drop items with empty annotations
data = data.dropna(subset=[PATH_ANN])
# data = data[data[PATH_ANN].isna()]

data.info()

#%% Check image name duplicates

image_names = data[PATH_IMG].apply(get_img_name)

# ONLY FOR CBIS_DDSM
image_names = data["subject"] + image_names

duplicated_imgs = image_names[image_names.duplicated()]
if len(duplicated_imgs) != 0:
    print("Found duplicate image names.")
    print(duplicated_imgs)
else:
    print("No duplicate names found.")

#%% ...

data.drop_duplicates(keep=False)
data.info()

#%% ...

def append_to_subset(i, k):
    train_no, test_no, val_no = len(train_indices), len(test_indices), len(val_indices)
    total_no = train_no + test_no + val_no
    if total_no == 0:
        train_keys.add(k)
        train_indices.append(i)
    else:
        train_r, test_r, val_r = train_no/total_no, test_no/total_no, val_no/total_no
        train_err, test_err, val_err = \
            (train_ratio - train_r)/train_ratio, (test_ratio - test_r)/test_ratio, (val_ratio - val_r)/val_ratio
        if train_err >= max(test_err, val_err):
            train_keys.add(k)
            train_indices.append(i)
        elif test_err >= max(train_err, val_err):
            test_keys.add(k)
            test_indices.append(i)
        else:
            val_keys.add(k)
            val_indices.append(i)


#%% Split train-test-val

ndx = data.index.to_numpy()
np.random.shuffle(ndx)

train_keys = set()
test_keys = set()
val_keys = set()

train_indices = []
test_indices = []
val_indices = []

train_ratio = 0.7
test_ratio = 0.15
val_ratio = 0.15


for i in ndx:
    row = data.loc[i]
    key = get_img_key(row)
    if key in train_keys:
        train_indices.append(i)
    elif key in test_keys:
        test_indices.append(i)
    elif key in val_keys:
        val_indices.append(i)
    else:
        append_to_subset(i, key)


n = len(data)
len(train_indices)/n, len(test_indices)/n, len(val_indices)/n

#%% Save train_test_split to cvs

train_data = data.loc[train_indices]
train_data.to_csv(ROOT / f"{data_file_name}_train.csv")

test_data = data.loc[test_indices]
test_data.to_csv(ROOT / f"{data_file_name}_test.csv")

val_data = data.loc[val_indices]
val_data.to_csv(ROOT / f"{data_file_name}_val.csv")

#%%
