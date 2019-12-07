import numpy as np
import pandas as pd
import pickle

from persim import PersImage

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


pimsd = 1e-5
px = 20
pim = PersImage(pixels=[px,px], spread=pimsd)

pim_df = pd.read_csv("./pim_vectors_20.csv")
pim_df.gest = pim_df.gest.astype("category")
pim_df = pim_df.loc[pim_df.gest.isin([1,2,3,4]), :] # only main gestures

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

# code to load model
with open("log_reg_skl_sbst.sav", "rb") as fh:
   log_reg = pickle.load(fh)

# score
print(log_reg.score(pims_test, gests_test))

# inverse persims
inverse_image = np.copy(log_reg.coef_).reshape(-1, px)
for i in range(4):
    pim.show(inverse_image[i*px:(i+1)*px, :])
    plt.title("Inverse Persistence Image for Gesture: " + str(i+1))
    plt.show()
