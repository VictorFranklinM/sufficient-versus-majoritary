import copy

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

from src.wrappers import DecisionTreeWrapper, RandomForestWrapper

MAX_RANDOM_FOREST_DEPTH = 8

sign = lambda x, predicate: x if predicate else -x


def get_x_y(dataset):
    X = dataset.iloc[:, :-1]  # everything except last column
    y = dataset.iloc[:, -1]  # last column
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
        explanation.append(sign(node + 1, feat_value <= thres))
        node = tree.children_left[node] if feat_value <= thres else tree.children_right[node]
    explanation.append(node)

    pred = t_clf.classes_[np.argmax(tree.value[node])]
    return np.array(explanation[:-1]), pred


def forest_explain_instance(f_clf, x):
    estimators = f_clf.estimators_
    explanations = []
    predictions = []
    classes = estimators[0].classes_

    add_count = lambda x: x + node_count if x >= 0 else x - node_count

    node_count = 0

    for t_clf in estimators:
        expl, pred = tree_explain_instance(t_clf, x)
        explanations.append([add_count(e) for e in expl])
        predictions.append(pred)
        node_count += len([f for f in t_clf.tree_.feature if f != -2])

    forest_class = max(classes, key=lambda c: sum(predictions == c))
    print("TYPE OF CLASSES", type(classes))

    forest_direct_reason = []

    for i in range(len(explanations)):
        if predictions[i] == forest_class:
            # print(i, np.array(explanations[i]))
            forest_direct_reason += explanations[i]

    return np.array(sorted(forest_direct_reason, key=abs))


def rf_cross_validation(data, n_trees, cv, n_forests=None):
    if not n_forests: n_forests = cv
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)  # Creates cross validation groups and shuffles them
    forests = []
    scores = []  # Avg score of all the trees

    for train_idx, test_idx in kf.split(data):
        train_set, test_set = data.iloc[train_idx], data.iloc[test_idx]

        X_train, y_train = get_x_y(train_set)
        X_test, y_test = get_x_y(test_set)

        # X_train, y_train = train_set.iloc[:, :-1], train_set.iloc[:, -1]
        # X_test, y_test = test_set.iloc[:, :-1], test_set.iloc[:, -1]

        rf = RandomForestClassifier(max_depth=MAX_RANDOM_FOREST_DEPTH, n_estimators=n_trees, random_state=42)
        rf.fit(X_train, y_train)
        y_predict = rf.predict(X_test)

        acc = accuracy_score(y_test, y_predict) * 100
        scores.append(acc)
        forests.append((copy.deepcopy(rf), train_idx, test_idx, acc))
    avg_score = np.mean(scores)  # calc avg
    return avg_score, forests


def main():
    # Use placement or compas for testing purposes
    fichier = "placement"
    dataset = pd.read_csv(f"../datasets/{fichier}.csv")
    print("Dataset: ")
    print(dataset)
    print()
    avg_score, forests = rf_cross_validation(dataset, 25, 10)
    print("avg_score: ")
    print(avg_score)
    print()
    first_forest: RandomForestClassifier = forests[0][0]
    first_train, first_test = forests[0][1], forests[0][2]
    first_clf: DecisionTreeClassifier = first_forest.estimators_[0]
    tree = first_clf.tree_

    print("################################")
    print()

    print("Max depth: ")
    print(tree.max_depth)
    print()

    print("Node count: ")
    print(tree.node_count)
    print()

    print("Children_left: ")
    print(tree.children_left)
    print(len(tree.children_left))
    print()

    print("children_right: ")
    print(tree.children_right)
    print(len(tree.children_right))
    print()

    print("features: ")
    print(tree.feature)
    print()

    print("thresholds: ")
    print(tree.threshold)
    print()

    print("values: ")
    print([int(np.argmax(tree.value[i])) if tree.feature[i] == -2 else None for i in range(len(tree.feature))])
    print([bool(np.argmax(tree.value[i])) if tree.feature[i] == -2 else None for i in range(len(tree.feature))])
    print()

    print("################################")
    print()

    X, y = get_x_y(dataset)
    instance = X.iloc[0]
    clazz = y.iloc[0]
    print("Instance: ")
    print(instance)
    print(clazz)
    print()

    print("Tree Explain instance: ")
    for t_clf in first_forest.estimators_:
        expl, pred = tree_explain_instance(t_clf, instance)
        print(expl, pred)
    print()

    print("Forest Explain instance: ")
    f_expl = forest_explain_instance(first_forest, instance)
    print(f_expl)
    print(len(f_expl))

    first_tree_map = DecisionTreeWrapper(first_clf)
    print("\nTree Direct Reason:")
    for t_clf in first_forest.estimators_:
        wrapper = DecisionTreeWrapper(t_clf)
        t_expl, t_pred = wrapper.find_direct_reason(instance, z3=True)
        # pred = float(pred)
        print(type(t_expl[0]))
        print(t_expl, t_pred)
    # print(first_tree_map.get_direct_reason(X.iloc[0]))

    print("\nDecision Tree Sufficient Reason")
    for t_clf in first_forest.estimators_:
        wrapper = DecisionTreeWrapper(t_clf)
        t_expl = wrapper.find_sufficient_reason(instance, int(clazz))
        t_expl_z3 = wrapper.find_sufficient_reason(instance, int(clazz), z3=True)
        # pred = float(pred)
        print(t_expl)
        print(t_expl_z3)

    # suff_reason = wrapped_forest.find_sufficient_reason(instance, clazz)
    # print(np.array(suff_reason), len(suff_reason))

    print("\nForest Direct Reason: ")
    wrapped_forest = RandomForestWrapper(first_forest)
    f_expl, f_pred, votes = wrapped_forest.find_direct_reason(instance)
    print(f_expl, f_pred, votes, len(f_expl))
    print(len(wrapped_forest.binarize_instance(instance)))

    print("\nTree CNF Encoding: ")
    print(first_tree_map.to_cnf(negate_tree=True))
    print(first_tree_map.to_z3_formula())

    # print(len(wrapped_forest.binarization))

    # wrapped_forest.calc_cnf_h()

    print("\nRandom Forest Sufficient Reason")
    suff_reason = wrapped_forest.find_sufficient_reason(instance)
    print(np.array(suff_reason), len(suff_reason))

    # print("###############################")
    # test_count = 0
    # for forest in forests:
    #     test_wrapped_forest = RandomForestWrapper(forest[0])
    #     f_test_X, f_test_y = get_x_y(dataset.iloc[forest[2]])
    #     test_count += 1
    #     print(f"\nWork on forest {test_count}")
    #     for i in range(len(f_test_X)):
    #         test_suff_reason = test_wrapped_forest.find_sufficient_reason(f_test_X.iloc[i])
    #         print(np.array(test_suff_reason), len(test_suff_reason))

    # forest_map = ForestFeatureThresholdMap(first_forest, feature_names=X.columns)
    # print(forest_map.get_mapping())
    # print(forest_map.get_tuples())
    # # print(forest_map.get_human_readable())
    #
    # instance = X.iloc[0]
    # valuation = forest_map.evaluate_instance(instance)
    #
    # print("\nValoração da floresta para a instância:")
    # print(valuation)


if __name__ == "__main__":
    main()
