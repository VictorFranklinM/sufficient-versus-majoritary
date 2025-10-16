import os
import copy

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from src.wrapper import *

MAX_RANDOM_FOREST_DEPTH = 8

sign = lambda x,predicate: x if predicate else -x

def get_x_y(dataset):
    X = dataset.iloc[:, :-1] # everything except last column
    y = dataset.iloc[:, -1] # last column
    return X, y

def tree_explain_instance(t_clf, x):
    tree = t_clf.tree_
    features = tree.feature
    thresholds = tree.threshold

    node = 0
    explanation = []

    while tree.children_left[node] != tree.children_right[node]:
        feat_idx = features[node]
        feat_value = x.iloc[feat_idx]
        thres = thresholds[node]
        explanation.append(sign(node+1, feat_value <= thres))
        node = tree.children_left[node] if feat_value <= thres else tree.children_right[node]
    explanation.append(node)

    pred = t_clf.classes_[np.argmax(tree.value[node])]
    return np.array(explanation[:-1]), pred

def forest_explain_instance(f_clf, x):
    estimators = f_clf.estimators_
    explanations = []
    predictions = []
    classes = estimators[0].classes_

    add_count = lambda x: x+node_count if x >= 0 else x-node_count

    node_count = 0

    for t_clf in estimators:
        expl, pred = tree_explain_instance(t_clf, x)
        explanations.append([add_count(e) for e in expl])
        predictions.append(pred)
        node_count += len([f for f in t_clf.tree_.feature if f != -2])

    forest_class = max(classes, key=lambda c: sum(predictions == c))

    forest_direct_reason = []

    for i in range(len(explanations)):
        if predictions[i] == forest_class:
            print(i, np.array(explanations[i]))
            forest_direct_reason += explanations[i]

    return np.array(forest_direct_reason)


def rf_cross_validation(data, n_trees, cv, n_forests=None):
    if not n_forests: n_forests = cv
    kf = KFold(n_splits=cv, shuffle=True, random_state=None)  # Creates cross validation groups and shuffles them
    forests = []
    scores = []  # Avg score of all the trees

    for train_idx, test_idx in kf.split(data):
        train_set, test_set = data.iloc[train_idx], data.iloc[test_idx]

        X_train, y_train = get_x_y(train_set)
        X_test, y_test = get_x_y(test_set)

        # X_train, y_train = train_set.iloc[:, :-1], train_set.iloc[:, -1]
        # X_test, y_test = test_set.iloc[:, :-1], test_set.iloc[:, -1]

        rf = RandomForestClassifier(max_depth=MAX_RANDOM_FOREST_DEPTH, n_estimators=n_trees, random_state=None)
        rf.fit(X_train, y_train)
        y_predict = rf.predict(X_test)

        acc = accuracy_score(y_test, y_predict) * 100
        scores.append(acc)
        forests.append((copy.deepcopy(rf), train_idx, test_idx, acc))
    avg_score = np.mean(scores) # calc avg
    return avg_score, forests


def main():
    # Use placement or compas for testing purposes
    fichier = "placement"
    dataset = pd.read_csv(f"datasets/{fichier}.csv")
    print(dataset)
    avg_score, forests = rf_cross_validation(dataset, 25, 10)
    print(avg_score)
    first_forest: RandomForestClassifier = forests[0][0]
    first_train, first_test = forests[0][1], forests[0][2]
    first_clf: DecisionTreeClassifier = first_forest.estimators_[0]
    tree = first_clf.tree_

    print(tree.max_depth)
    print(tree.node_count)
    print(tree.children_left)
    print(tree.children_right)
    print()
    print(tree.feature)
    print()

    X, y = get_x_y(dataset)
    instance = X.iloc[0]
    print(X.iloc[0])
    print(y.iloc[0])
    for t_clf in first_forest.estimators_:
        expl, pred = tree_explain_instance(t_clf, instance)
        print(expl, pred)

    print()
    f_expl = forest_explain_instance(first_forest, instance)
    print()
    print(f_expl)
    print(len(f_expl))
    
    print("\nTree Direct Reason:\n")

    forest_map = ForestFeatureThresholdMap(first_forest, feature_names=X.columns)
    # print(forest_map.get_mapping())
    # print(forest_map.get_tuples())
    print(forest_map.get_human_readable())

if __name__ == "__main__":
    main()

