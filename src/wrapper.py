import numpy as np
from pandas import Series
from pysat.card import CardEnc
from pysat.formula import IDPool, CNF
from pysat.solvers import Solver
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


class DecisionNodeWrapper:
    def __init__(self, var, left=None, right=None, value=None):
        self.var = var
        self.left = left
        self.right = right
        self.value = value


class DecisionTreeWrapper:
    def __init__(self, clf: DecisionTreeClassifier):
        self.tree = clf.tree_
        self.root = None
        self.binarization = {}
        self.n_nodes = 0
        self._wrap_clf()

    def _wrap_clf(self):
        # maybe do with clf.classes_
        values = [bool(np.argmax(self.tree.value[i])) for i in range(self.tree.node_count)]

        nodes = []
        count = 0  # count for unique feat_thres pairs
        for i in range(self.tree.node_count):
            if self.tree.children_left[i] == self.tree.children_right[i]:  # is a leaf
                node = DecisionNodeWrapper(None, value=values[i])
            else:  # not a leaf
                feat_thres = (self.tree.feature[i], self.tree.threshold[i])
                if feat_thres not in self.binarization:
                    count += 1
                    self.binarization[feat_thres] = count
                node = DecisionNodeWrapper(feat_thres)
            nodes.append(node)

        self.n_nodes = len(nodes)
        self.root = nodes[0]
        for i in range(len(nodes)):
            if self.tree.children_left[i] == self.tree.children_right[i]: continue  # is a leaf
            nodes[i].right = nodes[self.tree.children_left[i]]
            nodes[i].left = nodes[self.tree.children_right[i]]

    def binarize_instance(self, instance, hash_bin=None, reversed=False, invert_condition= False):
        if not hash_bin:
            hash_bin = self.binarization

        output = []

        for feat, thres in self.binarization.keys():
            p =  instance.iloc[feat] <= thres
            p = (not p) if invert_condition else p
            if p:
                output.append(hash_bin[(feat, thres)])
            else:
                output.append(-hash_bin[(feat, thres)])
        output.sort(key=abs, reverse=reversed)
        return np.array(output)

    def take_decision(self, instance) -> bool:
        node = self.root
        while node.var:
            if instance.iloc[node.var[0]] <= node.var[1]:
                node = node.left
            else:
                node = node.right
        return node.value

    def find_direct_reason(self, instance: Series, hash_bin=None):
        if not hash_bin:
            hash_bin = self.binarization

        node = self.root
        explanation = []

        while node.var:
            feat = instance.iloc[node.var[0]]
            thres = node.var[1]
            sign = 1 if feat <= thres else -1
            explanation.append(sign * hash_bin[node.var])
            node = node.left if feat <= thres else node.right

        pred = node.value
        return np.array(explanation), pred

    def to_cnf(self, hash_bin=None, negate_tree=False):
        if hash_bin is None:
            hash_bin = self.binarization
        return [sorted(clause) for clause in self._to_cnf(self.root, hash_bin, negate_tree)]

    def _to_cnf(self, node: DecisionNodeWrapper, hash_bin: dict[tuple[np.int64, np.float64], int], negate_tree: bool):
        # Base case: leaf node
        if node is None: raise ValueError("Provide a root")
        if node.value is not None:
            if node.value is not None:
                if negate_tree:
                    # Swap True/False for negation
                    return [] if not node.value else [[]]
                else:
                    return [] if node.value else [[]]

        # Recursive case: internal decision node
        left_cnf = self._to_cnf(node.left, hash_bin, negate_tree)
        right_cnf = self._to_cnf(node.right, hash_bin, negate_tree)

        # Combine using Shannon expansion:
        # CNF = (¬var ∨ clauses from left) ∧ (var ∨ clauses from right)
        cnf = []

        for clause in left_cnf:
            cnf.append(clause + [hash_bin[node.var]])  # if left was False, var=True fixes it
        for clause in right_cnf:
            cnf.append(clause + [-hash_bin[node.var]])  # if right was False, var=False fixes it

        return cnf

    def __len__(self):
        return self.n_nodes


class RandomForestWrapper:
    def __init__(self, clf: RandomForestClassifier):
        self.forest = clf
        self.binarization = {}
        self.n_bin_features = 0
        self.trees: list[DecisionTreeWrapper] = []
        self.n_trees = 0
        self._wrap_forest()

    def _wrap_forest(self):
        count = 0
        for t_clf in self.forest.estimators_:
            wrapped = DecisionTreeWrapper(t_clf)
            self.trees.append(wrapped)
            for feat_thres in wrapped.binarization.keys():
                if feat_thres not in self.binarization.keys():
                    count += 1
                    self.binarization[feat_thres] = count
        self.n_trees = len(self.trees)
        self.n_bin_features = len(self.binarization)

    def binarize_instance(self, instance, hash_bin=None, reverse=False, invert_condition=False):
        if not hash_bin:
            hash_bin = self.binarization

        output = []
        for tree in self.trees:
            output.extend(list(tree.binarize_instance(instance, hash_bin=hash_bin, invert_condition=invert_condition)))

        output = list(set(output))
        output.sort(key=abs, reverse=reverse)
        return np.array(output)

    def find_direct_reason(self, instance: Series):
        explanations = []
        predictions = []

        for tree in self.trees:
            expl, pred = tree.find_direct_reason(instance, self.binarization)
            explanations.append(list(expl))
            predictions.append(pred)

        # print(explanations)
        forest_class = max(np.array([True, False]), key=lambda c: sum(predictions == c))

        forest_direct_reason = []

        for i in range(len(explanations)):
            if predictions[i] == forest_class:
                # print(i, np.array(explanations[i]))
                forest_direct_reason += explanations[i]

        forest_direct_reason = list(set(forest_direct_reason))

        # return np.array(forest_direct_reason)
        return np.array(sorted(forest_direct_reason, key=abs)), bool(forest_class)

    def calc_cnf_h(self) -> CNF:
        # count = 1000
        vpool = IDPool(occupied=[[0, (((self.n_bin_features // 100) + 1) * 100)]])

        # create fresh variables y1,...,ym
        y_vars = [vpool.id(f'y{i}') for i in range(self.n_trees)]

        cnf_implicant = []
        for yi, Ti in zip(y_vars, self.trees):
            tree_cnf = Ti.to_cnf(hash_bin=self.binarization, negate_tree=True)
            for clause in tree_cnf:
                cnf_implicant.append([-yi] + clause)

        k = (self.n_trees // 2) + 1
        card_cnf = CardEnc.atleast(lits=y_vars, bound=k, vpool=vpool)
        combined_cnf = CNF()
        combined_cnf.extend(cnf_implicant)
        combined_cnf.extend(card_cnf.clauses)
        return combined_cnf

    def is_sufficient_reason(self, candidate, h: CNF):
        term_clause = [[l] for l in candidate]
        combined = CNF()
        combined.extend(h.clauses)
        combined.extend(term_clause)

        with Solver(bootstrap_with=combined.clauses) as solver:
            return not solver.solve()

    def find_sufficient_reason(self, instance, binarized_instance=False):
        if not binarized_instance:
            implicant = list(self.binarize_instance(instance, reverse=True)) # invert true and false direction to go
        else:
            implicant = list(instance)
        implicant = [int(item) for item in implicant]
        h_cnf = self.calc_cnf_h()
        i = 0
        while i < len(implicant):
            candidate = implicant.copy()
            candidate.pop(i)
            if self.is_sufficient_reason(candidate, h_cnf):
                implicant = candidate
            else:
                i += 1
        return sorted(implicant, key=abs)

    def __len__(self):
        return sum([len(tree) for tree in self.trees])
