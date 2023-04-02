#%% Init
import pathlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(".").resolve()
PATH_IMG = 'path_img'
PATH_ANN = 'path_box_txt'

get_img_key = lambda r: (r['study'], r['series'])

#%% Load data

data_file_name = "vindr"
data = pd.read_csv(ROOT / f"{data_file_name}.csv")
data.info()

#%% Filter data

# Drop items with empty annotations
# data = data.dropna(subset=[PATH_ANN])
data = data[data[PATH_ANN].isna()]

data.info()

#%% Check image name duplicates

f = lambda s: Path(s).stem
image_names = data[PATH_IMG].apply(f)

assert len(image_names[image_names.duplicated()]) == 0, "Found duplicate image names."


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
train_data.to_csv(ROOT / f"{data_file_name}_train_neg.csv")

test_data = data.loc[test_indices]
test_data.to_csv(ROOT / f"{data_file_name}_test_neg.csv")

val_data = data.loc[val_indices]
val_data.to_csv(ROOT / f"{data_file_name}_val_neg.csv")

#%%
