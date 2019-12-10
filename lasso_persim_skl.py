import numpy as np
import pandas as pd
import pickle
from time import time

from persim import PersImage

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


pim_df = pd.read_csv("./pim_vectors_mp20_sbst.csv")
pim_df.gest = pim_df.gest.astype("category")


pims = pim_df.values[:, :-2] # predictor vectors: persistence images (864xpx**2)
gests = pim_df.values[:, -2].astype("int64") # data labels: gesture numbers
unq_gests = np.unique(gests).size


######### train/ test split ########
np.random.seed(1)
pims_train, pims_test, gests_train, gests_test = train_test_split(
    pims,
    gests,
    test_size=0.2)


lasso_reg = LogisticRegression(
    penalty="l1",
    C=30,
    solver="saga",
    fit_intercept=True,
    max_iter=7000,
    multi_class="multinomial",
    random_state=1)

oos_acc = lasso_reg.score(pims_test, gests_test)
print(f"Accuracy: {oos_acc * 100}%")

## save model
with open("./saved_models/lasso_skl.sav", "wb") as fh:
    pickle.dump(lasso_reg, fh)
