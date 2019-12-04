import numpy as np
import pandas as pd
import pickle

from persim import PersImage

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold
import matplotlib.pyplot as plt

pim_df = pd.read_csv("./pim_vectors_20.csv")
pim_df.gest = pim_df.gest.astype("category")
pim_df = pim_df.loc[pim_df.gest != 1, :] # excl. well seperated gest 1

### persistence image object ### might not need this
#pimsd = 1e-5
#px = 20
#pim = PersImage(pixels=[px,px], spread=pimsd)


######### first compute accuracy score with properly labeled gestures ##########

pims = pim_df.values[:, :-2] # predictor vectors: persistence images (864xpx**2)
gests = pim_df.values[:, -2].astype("int64") # data labels: gesture numbers
unq_gests = np.unique(gests).size

log_reg = LogisticRegression(
    penalty="l2",
    C=1e6,
    solver="newton-cg",
    fit_intercept=True,
    max_iter=5000,
    multi_class="multinomial",
    random_state=1)

skf = StratifiedKFold(n_splits = 5, random_state = 1, shuffle = True)

fold_acc = []
for train_idx, test_idx in skf.split(pims, gests):
    log_reg.fit(pims[train_idx, ], gests[train_idx])
    fold_acc.append(log_reg.score(pims[test_idx, ], gests[test_idx]))

print(fold_acc)
