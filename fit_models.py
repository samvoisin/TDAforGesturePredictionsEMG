################################################################################
############### Generate Data Set of Persistence Diagram Vectors ###############
############# Calculating Rips complex on all channels w/ out time #############
################################################################################

import numpy as np
import pandas as pd
import numpy.linalg as la
import pickle

from persim import PersImage

from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt

# read in persistence image data sets
raw_pim_df = pd.read_csv("./Data/raw_pim_vectors.csv")
iso_pim_df = pd.read_csv("./Data/iso_pim_vectors.csv")
snf_pim_df = pd.read_csv("./Data/snf_pim_vectors.csv")

raw_pims = raw_pim_df.values[:, :-2]
iso_pims = iso_pim_df.values[:, :-2]
snf_pims = snf_pim_df.values[:, :-2]

raw_gest_lab = raw_pim_df.values[:, -1].astype("int32") # gesture nums
raw_subj_lab = raw_pim_df.values[:, -2].astype("int32") # subject nums

iso_gest_lab = iso_pim_df.values[:, -1].astype("int32") # gesture nums
iso_subj_lab = iso_pim_df.values[:, -2].astype("int32") # subject nums

snf_gest_lab = snf_pim_df.values[:, -1].astype("int32") # gesture nums
snf_subj_lab = snf_pim_df.values[:, -2].astype("int32") # subject nums

# initialize stratified 5-fold cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

################################################################################
######################### Model 1: K Nearest Neightbor #########################
################################################################################

print("1NN:")

from sklearn.neighbors import KNeighborsClassifier

raw_neigh = KNeighborsClassifier(n_neighbors=1, weights="uniform" ,p=1)
raw_acc = []
for trn_idx, tst_idx in skf.split(raw_pims, raw_gest_lab):
    X_train, X_test = raw_pims[trn_idx], raw_pims[tst_idx]
    y_train, y_test = raw_gest_lab[trn_idx], raw_gest_lab[tst_idx]
    raw_neigh.fit(X_train, y_train)
    raw_acc.append(raw_neigh.score(X_test, y_test))

print(np.array(raw_acc).mean() * 100)


iso_neigh = KNeighborsClassifier(n_neighbors=1, weights="uniform" ,p=1)
iso_acc = []
for trn_idx, tst_idx in skf.split(iso_pims, iso_gest_lab):
    X_train, X_test = iso_pims[trn_idx], iso_pims[tst_idx]
    y_train, y_test = iso_gest_lab[trn_idx], iso_gest_lab[tst_idx]
    iso_neigh.fit(X_train, y_train)
    iso_acc.append(iso_neigh.score(X_test, y_test))

print(np.array(iso_acc).mean() * 100)


snf_neigh = KNeighborsClassifier(n_neighbors=1, weights="uniform" ,p=1)
snf_acc = []
for trn_idx, tst_idx in skf.split(snf_pims, snf_gest_lab):
    X_train, X_test = snf_pims[trn_idx], snf_pims[tst_idx]
    y_train, y_test = snf_gest_lab[trn_idx], snf_gest_lab[tst_idx]
    snf_neigh.fit(X_train, y_train)
    snf_acc.append(snf_neigh.score(X_test, y_test))

print(np.array(snf_acc).mean() * 100)

################################################################################

print("3NN:")

from sklearn.neighbors import KNeighborsClassifier

raw_neigh = KNeighborsClassifier(n_neighbors=3, weights="uniform" ,p=1)
raw_acc = []
for trn_idx, tst_idx in skf.split(raw_pims, raw_gest_lab):
    X_train, X_test = raw_pims[trn_idx], raw_pims[tst_idx]
    y_train, y_test = raw_gest_lab[trn_idx], raw_gest_lab[tst_idx]
    raw_neigh.fit(X_train, y_train)
    raw_acc.append(raw_neigh.score(X_test, y_test))

print(np.array(raw_acc).mean() * 100)


iso_neigh = KNeighborsClassifier(n_neighbors=3, weights="uniform" ,p=1)
iso_acc = []
for trn_idx, tst_idx in skf.split(iso_pims, iso_gest_lab):
    X_train, X_test = iso_pims[trn_idx], iso_pims[tst_idx]
    y_train, y_test = iso_gest_lab[trn_idx], iso_gest_lab[tst_idx]
    iso_neigh.fit(X_train, y_train)
    iso_acc.append(iso_neigh.score(X_test, y_test))

print(np.array(iso_acc).mean() * 100)


snf_neigh = KNeighborsClassifier(n_neighbors=3, weights="uniform" ,p=1)
snf_acc = []
for trn_idx, tst_idx in skf.split(snf_pims, snf_gest_lab):
    X_train, X_test = snf_pims[trn_idx], snf_pims[tst_idx]
    y_train, y_test = snf_gest_lab[trn_idx], snf_gest_lab[tst_idx]
    snf_neigh.fit(X_train, y_train)
    snf_acc.append(snf_neigh.score(X_test, y_test))

print(np.array(snf_acc).mean() * 100)

################################################################################

print("5NN:")

from sklearn.neighbors import KNeighborsClassifier

raw_neigh = KNeighborsClassifier(n_neighbors=5, weights="uniform" ,p=1)
raw_acc = []
for trn_idx, tst_idx in skf.split(raw_pims, raw_gest_lab):
    X_train, X_test = raw_pims[trn_idx], raw_pims[tst_idx]
    y_train, y_test = raw_gest_lab[trn_idx], raw_gest_lab[tst_idx]
    raw_neigh.fit(X_train, y_train)
    raw_acc.append(raw_neigh.score(X_test, y_test))

print(np.array(raw_acc).mean() * 100)


iso_neigh = KNeighborsClassifier(n_neighbors=5, weights="uniform" ,p=1)
iso_acc = []
for trn_idx, tst_idx in skf.split(iso_pims, iso_gest_lab):
    X_train, X_test = iso_pims[trn_idx], iso_pims[tst_idx]
    y_train, y_test = iso_gest_lab[trn_idx], iso_gest_lab[tst_idx]
    iso_neigh.fit(X_train, y_train)
    iso_acc.append(iso_neigh.score(X_test, y_test))

print(np.array(iso_acc).mean() * 100)


snf_neigh = KNeighborsClassifier(n_neighbors=5, weights="uniform" ,p=1)
snf_acc = []
for trn_idx, tst_idx in skf.split(snf_pims, snf_gest_lab):
    X_train, X_test = snf_pims[trn_idx], snf_pims[tst_idx]
    y_train, y_test = snf_gest_lab[trn_idx], snf_gest_lab[tst_idx]
    snf_neigh.fit(X_train, y_train)
    snf_acc.append(snf_neigh.score(X_test, y_test))

print(np.array(snf_acc).mean() * 100)


################################################################################
######################### Model 2: Logistic Regression #########################
################################################################################

print("Logistic Regression:")

from sklearn.linear_model import LogisticRegression

log_reg_raw = LogisticRegression(
    penalty="l2",
    C=1e6,
    solver="newton-cg",
    fit_intercept=True,
    max_iter=5000,
    multi_class="multinomial",
    random_state=2,
    warm_start=True)

raw_acc = []
for trn_idx, tst_idx in skf.split(raw_pims, raw_gest_lab):
    X_train, X_test = raw_pims[trn_idx], raw_pims[tst_idx]
    y_train, y_test = raw_gest_lab[trn_idx], raw_gest_lab[tst_idx]
    log_reg_raw.fit(X_train, y_train)
    raw_acc.append(log_reg_raw.score(X_test, y_test))

print(np.array(raw_acc).mean() * 100)


log_reg_iso = LogisticRegression(
    penalty="l2",
    C=1e6,
    solver="newton-cg",
    fit_intercept=True,
    max_iter=5000,
    multi_class="multinomial",
    random_state=2,
    warm_start=True)

iso_acc = []
for trn_idx, tst_idx in skf.split(iso_pims, iso_gest_lab):
    X_train, X_test = iso_pims[trn_idx], iso_pims[tst_idx]
    y_train, y_test = iso_gest_lab[trn_idx], iso_gest_lab[tst_idx]
    log_reg_iso.fit(X_train, y_train)
    iso_acc.append(log_reg_iso.score(X_test, y_test))

print(np.array(iso_acc).mean() * 100)


log_reg_snf = LogisticRegression(
    penalty="l2",
    C=1e6,
    solver="newton-cg",
    fit_intercept=True,
    max_iter=5000,
    multi_class="multinomial",
    random_state=2,
    warm_start=True)

snf_acc = []
for trn_idx, tst_idx in skf.split(snf_pims, snf_gest_lab):
    X_train, X_test = snf_pims[trn_idx], snf_pims[tst_idx]
    y_train, y_test = snf_gest_lab[trn_idx], snf_gest_lab[tst_idx]
    log_reg_snf.fit(X_train, y_train)
    snf_acc.append(log_reg_snf.score(X_test, y_test))

print(np.array(snf_acc).mean() * 100)

################################################################################
############################ Model 3: Random Forest ############################
################################################################################

print("Random Forest:")

from sklearn.ensemble import RandomForestClassifier

n_est = 100
raw_rf = RandomForestClassifier(
    n_estimators=n_est,
    criterion="entropy",
    random_state=2)

raw_acc = []
for trn_idx, tst_idx in skf.split(raw_pims, raw_gest_lab):
    X_train, X_test = raw_pims[trn_idx], raw_pims[tst_idx]
    y_train, y_test = raw_gest_lab[trn_idx], raw_gest_lab[tst_idx]
    raw_rf.fit(X_train, y_train)
    raw_acc.append(raw_rf.score(X_test, y_test))

print(np.array(raw_acc).mean() * 100)


iso_rf = RandomForestClassifier(
    n_estimators=n_est,
    criterion="entropy",
    random_state=2)

iso_acc = []
for trn_idx, tst_idx in skf.split(iso_pims, iso_gest_lab):
    X_train, X_test = iso_pims[trn_idx], iso_pims[tst_idx]
    y_train, y_test = iso_gest_lab[trn_idx], iso_gest_lab[tst_idx]
    iso_rf.fit(X_train, y_train)
    iso_acc.append(iso_rf.score(X_test, y_test))

print(np.array(iso_acc).mean() * 100)


snf_rf = RandomForestClassifier(
    n_estimators=n_est,
    criterion="entropy",
    random_state=2)

snf_acc = []
for trn_idx, tst_idx in skf.split(snf_pims, snf_gest_lab):
    X_train, X_test = snf_pims[trn_idx], snf_pims[tst_idx]
    y_train, y_test = snf_gest_lab[trn_idx], snf_gest_lab[tst_idx]
    snf_rf.fit(X_train, y_train)
    snf_acc.append(snf_rf.score(X_test, y_test))

print(np.array(snf_acc).mean() * 100)


################################################################################
###################### Model 4: Support Vector Classifier ######################
################################################################################

from sklearn.svm import SVC

print("SVC:")


raw_svc = SVC(
    C=1.0,
    kernel='rbf',
    gamma='scale',
    random_state=2)

raw_acc = []
for trn_idx, tst_idx in skf.split(raw_pims, raw_gest_lab):
    X_train, X_test = raw_pims[trn_idx], raw_pims[tst_idx]
    y_train, y_test = raw_gest_lab[trn_idx], raw_gest_lab[tst_idx]
    raw_svc.fit(X_train, y_train)
    raw_acc.append(raw_svc.score(X_test, y_test))

print(np.array(raw_acc).mean() * 100)


iso_svc = SVC(
    C=1.0,
    kernel='rbf',
    gamma='scale',
    random_state=2)

iso_acc = []
for trn_idx, tst_idx in skf.split(iso_pims, iso_gest_lab):
    X_train, X_test = iso_pims[trn_idx], iso_pims[tst_idx]
    y_train, y_test = iso_gest_lab[trn_idx], iso_gest_lab[tst_idx]
    iso_svc.fit(X_train, y_train)
    iso_acc.append(iso_svc.score(X_test, y_test))

print(np.array(iso_acc).mean() * 100)


snf_svc = SVC(
    C=1.0,
    kernel='rbf',
    gamma='scale',
    random_state=1)

snf_acc = []
for trn_idx, tst_idx in skf.split(snf_pims, snf_gest_lab):
    X_train, X_test = snf_pims[trn_idx], snf_pims[tst_idx]
    y_train, y_test = snf_gest_lab[trn_idx], snf_gest_lab[tst_idx]
    snf_svc.fit(X_train, y_train)
    snf_acc.append(snf_svc.score(X_test, y_test))

print(np.array(snf_acc).mean() * 100)


################################################################################
######################## Model 5: Multi-Layer Perceptron #######################
################################################################################

from sklearn.neural_network import MLPClassifier

print("MLP:")


raw_mlp = MLPClassifier(
    hidden_layer_sizes=(100, 100, 100, 50, 25),
    activation='relu',
    solver='adam',
    alpha=0.0001,
    batch_size='auto',
    power_t=0.5,
    max_iter=200,
    shuffle=True,
    random_state=2,
    warm_start=False,
    momentum=0.9,
    nesterovs_momentum=True,
    early_stopping=True)

raw_acc = []
for trn_idx, tst_idx in skf.split(raw_pims, raw_gest_lab):
    X_train, X_test = raw_pims[trn_idx], raw_pims[tst_idx]
    y_train, y_test = raw_gest_lab[trn_idx], raw_gest_lab[tst_idx]
    raw_mlp.fit(X_train, y_train)
    raw_acc.append(raw_mlp.score(X_test, y_test))

print(np.array(raw_acc).mean() * 100)


iso_mlp = MLPClassifier(
    hidden_layer_sizes=(100, 100, 100, 50, 25),
    activation='relu',
    solver='adam',
    alpha=0.0001,
    batch_size='auto',
    power_t=0.5,
    max_iter=200,
    shuffle=True,
    random_state=2,
    warm_start=False,
    momentum=0.9,
    nesterovs_momentum=True,
    early_stopping=True)

iso_acc = []
for trn_idx, tst_idx in skf.split(iso_pims, iso_gest_lab):
    X_train, X_test = iso_pims[trn_idx], iso_pims[tst_idx]
    y_train, y_test = iso_gest_lab[trn_idx], iso_gest_lab[tst_idx]
    iso_mlp.fit(X_train, y_train)
    iso_acc.append(iso_mlp.score(X_test, y_test))

print(np.array(iso_acc).mean() * 100)


snf_mlp = MLPClassifier(
    hidden_layer_sizes=(100, 100, 100, 50, 25),
    activation='relu',
    solver='adam',
    alpha=0.0001,
    batch_size='auto',
    power_t=0.5,
    max_iter=200,
    shuffle=True,
    random_state=2,
    warm_start=False,
    momentum=0.9,
    nesterovs_momentum=True,
    early_stopping=True)

snf_acc = []
for trn_idx, tst_idx in skf.split(snf_pims, snf_gest_lab):
    X_train, X_test = snf_pims[trn_idx], snf_pims[tst_idx]
    y_train, y_test = snf_gest_lab[trn_idx], snf_gest_lab[tst_idx]
    snf_mlp.fit(X_train, y_train)
    snf_acc.append(snf_mlp.score(X_test, y_test))

print(np.array(snf_acc).mean() * 100)
