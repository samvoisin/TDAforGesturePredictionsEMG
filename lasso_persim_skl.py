import numpy as np
import pandas as pd
import pickle
from time import time

from persim import PersImage

from sklearn.linear_model import LogisticRegression
#from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt


pim_df = pd.read_csv("./pim_vectors_mp20_sbst.csv")
pim_df.gest = pim_df.gest.astype("category")


pims = pim_df.values[:, :-2] # predictor vectors: persistence images (864xpx**2)
gests = pim_df.values[:, -2].astype("int64") # data labels: gesture numbers
unq_gests = np.unique(gests).size


# stratified kfold
flds = 3
skf = StratifiedKFold(n_splits=flds, shuffle=True)

fit_mat = np.zeros(100*2).reshape(-1, 2) # store C values and avg acc score

np.random.seed(1)
for n, lbda in enumerate(range(1, 1000, 10)):
    ### progress bar ###
    pb = "~"*int(n/50*100)+" "*int((1-n/50)*100)+"|"
    print(pb, end="\r")
    ####################
    ts = time()
    fv_acc = [] # store acc for each fold
    for trn_idx, tst_idx in skf.split(pims, gests):
        pims_trn, pims_tst = pims[trn_idx, :], pims[tst_idx, :]
        gests_trn, gests_tst = gests[trn_idx], gests[tst_idx]

        lasso_reg = LogisticRegression(
            penalty="l1",
            C=lbda,
            solver="saga",
            fit_intercept=True,
            max_iter=7000,
            multi_class="multinomial",
            random_state=1)

        lasso_reg.fit(pims_trn, gests_trn) # fit the model

        # append score for fold
        fv_acc.append(lasso_reg.score(pims_tst, gests_tst))
        te = time()

    acc = sum(fv_acc) / flds # avg acc
    print(f"iteration {n}; accuracy {acc}; took {(te-ts)/60} minutes")

    fit_mat[n, :] = np.array([lbda, acc])

plt.scatter(fit_mat[:, 0], fit_mat[:, 1])
plt.xlabel("lambda")
plt.ylabel("accuracy")
plt.savefig("./fitlasso.png")

## save model
#with open("./saved_models/lasso_skl.sav", "wb") as fh:
#    pickle.dump(lasso_reg, fh)
