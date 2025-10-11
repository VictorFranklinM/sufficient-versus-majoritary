import os
import copy

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

MAX_RANDOM_FOREST_DEPTH = 8


def rf_cross_validation(data, n_trees, cv, n_forests=None):
    if not n_forests: n_forests = cv
    kf = KFold(n_splits=cv, shuffle=True)  # Creates cross validation groups and shuffles them
    forests = []
    score = 0  # Avg score of all the trees

    for train_idx, test_idx in kf.split(data):
        train_set, test_set = data.iloc[train_idx], data.iloc[test_idx]
        X_train, y_train = train_set.iloc[:, :-1], train_set.iloc[:, -1]
        X_test, y_test = test_set.iloc[:, :-1], test_set.iloc[:, -1]
        rf = RandomForestClassifier(max_depth=MAX_RANDOM_FOREST_DEPTH, n_estimators=n_trees)
        rf.fit(X_train, y_train)
        y_predict = rf.predict(X_test)

        score += (accuracy_score(y_test, y_predict)) * 100
        forests.append((copy.deepcopy(rf), train_idx, test_idx))
    score /= n_forests # calc avg
    return score, forests


def main():
    fichiers = os.listdir('../datasets')
    if not fichiers:
        raise FileNotFoundError("Datasets not found! Extract datasets.zip into /datasets/")
    database = {}
    tree_amount_info = {"ad-data": 53, "adult": 89, "bank": 21, "christine": 87, "compas": 21,
                        "gina": 55, "gisette": 85, "dorothea": 71, "farm-ads": 41, "mnist49": 81,
                        "mnist38": 85, "dexter": 101, "recidivism": 25, "higgs-boson": 95, "placement": 25}

    for f in fichiers:
        database[f.split('.')[0]] = pd.read_csv(f"../datasets/{f}")

    for dataset in database.keys():
        if dataset not in tree_amount_info.keys(): continue
        print(f"Work on {dataset}")
        tree_amount = tree_amount_info[dataset]
        fold = 10
        ds = database[dataset].copy()

        score, forests = rf_cross_validation(ds, tree_amount, fold)
        print(f"{score=}")


if __name__ == '__main__':
    main()
