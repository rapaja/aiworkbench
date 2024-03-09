# %%

import pandas as pd

# %%

data = pd.read_csv("./podaci.csv", header=None)
data.info()
data.head()
# %%

# %%

data = data.drop(0, axis=1)
data.info()
data.head()
# %%

data.plot(y=[1, 2, 3])
# %%

WINDOW_LEN = 10
slices = []
for i in range(WINDOW_LEN, len(data) - 1):
    feature = data.iloc[i - WINDOW_LEN : i, :]
    target = data.iloc[i + 1, :]
    slices.append((feature, target))

slices
# %%
