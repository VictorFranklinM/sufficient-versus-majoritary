import copy
import random
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

MAX_RANDOM_FOREST_DEPTH = 8
sign = lambda x, predicate: x if predicate else -x

def get_x_y(dataset):
    X = dataset.iloc[:, :-1]
    y = dataset.iloc[:, -1]
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

def rf_cross_validation(data, n_trees, cv, n_forests=None):
    if not n_forests:
        n_forests = cv
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    forests = []
    scores = []
    for train_idx, test_idx in kf.split(data):
        train_set, test_set = data.iloc[train_idx], data.iloc[test_idx]
        X_train, y_train = get_x_y(train_set)
        X_test, y_test = get_x_y(test_set)
        rf = RandomForestClassifier(max_depth=MAX_RANDOM_FOREST_DEPTH, n_estimators=n_trees, random_state=42)
        rf.fit(X_train, y_train)
        y_predict = rf.predict(X_test)
        acc = accuracy_score(y_test, y_predict) * 100
        scores.append(acc)
        forests.append((copy.deepcopy(rf), train_idx, test_idx, acc))
    avg_score = np.mean(scores)
    return avg_score, forests

class MinimumMajoritaryExplainer:
    def __init__(self, forest: RandomForestClassifier):
        self.forest = forest
        self.n_trees = len(forest.estimators_)
        self.n_features = forest.n_features_in_

    def _predict_majority(self, x):
        preds = np.array([t.predict([x])[0] for t in self.forest.estimators_])
        values, counts = np.unique(preds, return_counts=True)
        majority_class = values[np.argmax(counts)]
        return majority_class

    def explain(self, x, nb_iterations=50, random_state=0):
        random.seed(random_state)
        x = np.array(x).astype(int)
        original_class = self._predict_majority(x)
        best_explanation = None
        for _ in range(nb_iterations):
            features = list(range(len(x)))
            if _ > 0:
                random.shuffle(features)
            candidate = x.copy()
            explanation = []
            for idx in features:
                saved_value = candidate[idx]
                candidate[idx] = 1 - candidate[idx]
                new_class = self._predict_majority(candidate)
                if new_class != original_class:
                    candidate[idx] = saved_value
                    explanation.append(idx)
            if best_explanation is None or len(explanation) < len(best_explanation):
                best_explanation = explanation
        return best_explanation

def forest_binarization(forest: RandomForestClassifier, X: pd.DataFrame):
    binarization_rules = {}
    counter = 1
    for t_clf in forest.estimators_:
        tree = t_clf.tree_
        for feature, threshold in zip(tree.feature, tree.threshold):
            if feature >= 0:
                k = (feature, threshold)
                if k not in binarization_rules:
                    binarization_rules[k] = counter
                    counter += 1
    X_bin = pd.DataFrame(index=X.index)
    for (feat_idx, threshold), new_feat_idx in binarization_rules.items():
        feature_name = X.columns[feat_idx]
        X_bin[f"{feature_name}>{threshold:.3f}"] = (X.iloc[:, feat_idx] > threshold).astype(int)
    return X_bin, binarization_rules

def main():
    fichier = "bank"
    dataset = pd.read_csv(f"datasets/{fichier}.csv")
    print("Dataset carregado:", fichier)
    print("Shape:", dataset.shape)
    print()
    X, y = get_x_y(dataset)
    rf_temp = RandomForestClassifier(max_depth=MAX_RANDOM_FOREST_DEPTH, n_estimators=25, random_state=42)
    rf_temp.fit(X, y)
    X_bin, binarization_rules = forest_binarization(rf_temp, X)
    dataset_bin = pd.concat([X_bin, y], axis=1)
    avg_score, forests = rf_cross_validation(dataset_bin, n_trees=25, cv=10)
    print("Acurácia média:", avg_score)
    print()
    first_forest: RandomForestClassifier = forests[0][0]
    print(dataset_bin)
    print()
    print(first_forest.estimators_[0].tree_.feature)
    print()
    print(first_forest.estimators_[0].tree_.threshold)
    print()
    explainer = MinimumMajoritaryExplainer(first_forest)
    instance = X_bin.iloc[3].values
    print("Instância selecionada:")
    print(instance)
    print()
    explanation = explainer.explain(instance)
    print("Explicação mínima majoritária (índices):", explanation)
    print("Features essenciais:", [X.columns[i] for i in explanation])
    print()
    print("Predições individuais das árvores:")
    preds = []
    for t_clf in first_forest.estimators_:
        expl, pred = tree_explain_instance(t_clf, X_bin.iloc[0])
        print(expl)
        preds.append(pred)
    print(preds)
    print("Classe majoritária:", max(set(preds), key=preds.count))

if __name__ == "__main__":
    main()
