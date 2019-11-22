################################################################################
############### Spectral Clustering for Viewing Gesture Clusters ###############
############ Data set is persistence images - see pim_fullvec_set.py ###########
################################################################################

import numpy as np
import pandas as pd
import pickle

from persim import PersImage

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


pim_df = pd.read_csv("./pim_vectors_20.csv")

pimsd = 1e-5
px = 20
pim = PersImage(pixels=[px,px], spread=pimsd)


pims = pim_df.values[:, :-2] # predictor vectors: persistence images (864xpx**2)
gests = pim_df.values[:, -2].astype("int64") # data labels: gesture numbers
unq_gests = np.unique(gests).size

######## train/ test split ########
np.random.seed(1)
pims_train, pims_test, gests_train, gests_test = train_test_split(
    pims,
    gests,
    test_size=0.2,
    random_state=1)

######## Logistic Regression ########

log_reg = LogisticRegression(
    penalty="l2",
    C=1e6,
    solver="newton-cg",
    fit_intercept=True,
    max_iter=5000,
    multi_class="multinomial",
    random_state=1)

log_reg.fit(pims_train, gests_train)

log_reg.score(pims_test, gests_test)

# save model
with open("log_reg_skl.sav", "wb") as fh:
    pickle.dump(log_reg, fh)

# code to load model
#with open("log_reg_skl.sav", "rb") as fh:
#   log_reg = pickle.load(fh)


inverse_image = np.copy(log_reg.coef_).reshape(-1, px)
for i in range(6):
    pim.show(inverse_image[i*px:(i+1)*px, :])
    plt.title("Inverse Persistence Image for Gesture: " + str(i+1))
    plt.show()
