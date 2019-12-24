################################################################################
################# PCA for Testing Gesture/ Subject Separability ################
############ Data set is persistence images - see pim_fullvec_set.py ###########
################################################################################

import pandas as pd
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.manifold import MDS

########################## Multi-Dimensional Scaling ###########################

pim_df = pd.read_csv("./pim_vectors_mp20_sbst.csv")

mds = MDS(3, metric=True)
embed = mds.fit_transform(pim_df.values[:, :-2])
print(embed.shape)

mds_df = pd.DataFrame(embed)
mds_df["gest"] = pim_df.gest
mds_df.columns = ["V1", "V2", "V3", "gest"]
mds_df.gest = mds_df.gest.astype("category")
fig = px.scatter_3d(mds_df, x='V1', y='V2', z='V3', color='gest')
fig.show()

####################### Principal Components Analaysis #########################

pims = pim_df.values[:, :-2] # persistence image vectors
pimcov = pims.T @ pims

evals, evecs = la.eig(pimcov)

# sort eigenvalues and eigenvectors in descending order
eidx = np.argsort(-evals.real)
evecs = evecs.real[:, eidx]
evals = evals.real[eidx]

# skree plot
skree = [sum(evals[:n+1]) for n, i in enumerate(evals)]
sns.scatterplot(range(1, 11), skree[:10])
plt.show()

print(f"Percent of var: {evals[:3].sum()/evals.sum()*100}")

########

eigbase = evecs[:, :3]
PCApims = pims @ eigbase

pca_df3 = pd.DataFrame(np.c_[PCApims, pim_df.values[:, -2]])
pca_df3.columns = ["V1", "V2", "V3", "gest"]
pca_df3.gest = pca_df3.gest.astype("category")

fig = px.scatter_3d(pca_df3, x='V1', y='V2', z='V3', color='gest')
fig.show()
