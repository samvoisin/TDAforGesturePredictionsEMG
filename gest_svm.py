################################################################################
####### Support Vector Machine for Testing Gesture/ Subject Separability #######
############ Data set is persistence images - see pim_fullvec_set.py ###########
################################################################################

import pandas as pd
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn import svm
from sklearn.model_selection import StratifiedKFold
from sklearn.cluster import SpectralClustering

pim_df = pd.read_csv("./pim_vectors_mp40.csv")

####################### Visualizing Priciple Components ########################

pims = pim_df.values[:, :-2] # persistence image vectors
pimcov = pims.T @ pims

spect = la.eig(pimcov)
print(spect[0][:10])

eval = 2

# percent of var exp
print(f"Percent of Variation: {sum(spect[0][eval:eval+2]) / sum(spect[0])}")

eigbase = spect[1][:, eval:eval+2].real

PCApims = pims @ eigbase

pca_df = pd.DataFrame(np.c_[PCApims, pim_df.values[:, -2]])
pca_df.columns = ["V1", "V2", "gest"]
pca_df.gest = pca_df.gest.astype("int32")
pca_df.gest = pca_df.gest.astype("category")

sns.scatterplot("V1", "V2", data=pca_df)
plt.show()

print(f"Percent of Variation: {sum(spect[0][eval:eval+2]) / sum(spect[0])}")

eigbase = spect[1][:, eval:eval+3].real
PCApims = pims @ eigbase

pca_df3 = pd.DataFrame(np.c_[PCApims, pim_df.values[:, -2]])
pca_df3.columns = ["V1", "V2", "V3", "gest"]
pca_df3.gest = pca_df3.gest.astype("int32")

fig = px.scatter_3d(pca_df3, x='V1', y='V2', z='V3', color='gest')
fig.show()

############################# Spectral Clustering ##############################

#X = pim_df.values[:, :-2]

#spc = SpectralClustering(
#    n_clusters=6,
#    affinity="rbf",
#    assign_labels="discretize"
#    )

#clstr = spc.fit_predict(X)

#print(clstr)



################################# Fitting SVM ##################################

X = pim_df.values[:, :-2]
y = pim_df.values[:, -2]

folds=5

skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=1)
skf.get_n_splits(X, y)

clf = svm.SVC(gamma="auto")

acc_scores = []
for train_index, test_index in skf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    clf.fit(X_train, y_train)
    acc_scores.append(clf.score(X_test, y_test))

print(sum(acc_scores) / folds)
