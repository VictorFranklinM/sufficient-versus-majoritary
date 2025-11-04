import numpy as np
from pandas import Series
from pysat.formula import CNF
from pysat.solvers import Solver as PySatSolver
from sklearn.tree import DecisionTreeClassifier
from z3 import Solver as Z3Solver, BoolRef
from z3 import Real, Int, And, Or, Not, Implies, simplify, RealVal, IntVal, sat, unsat, BoolVal
from z3 import Bools, Bool


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
            nodes[i].left = nodes[self.tree.children_left[i]]
            nodes[i].right = nodes[self.tree.children_right[i]]

    def binarize_instance(self, instance, hash_bin=None, reversed=False, invert_condition=False):
        if not hash_bin:
            hash_bin = self.binarization

        output = []

        for feat, thres in self.binarization.keys():
            p = instance.iloc[feat] <= thres
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

    def find_direct_reason(self, instance: Series, hash_bin=None, z3=False):
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
        if not z3:
            return np.array(explanation), pred
        else:
            z3_vars = Bools([f"x_{abs(e)}" for e in explanation])
            constraints = []
            for val, z3_var in zip(explanation, z3_vars):
                constraints.append(z3_var == (val > 0))
            return constraints, [Bool("y") == pred]



    def is_sufficient_reason(self, candidate, h: CNF):
        term_clause = [[l] for l in candidate]
        combined = CNF()
        combined.extend(h.clauses)
        combined.extend(term_clause)

        with PySatSolver(bootstrap_with=combined.clauses) as solver:
            return not solver.solve()

    def find_sufficient_reason(self, instance: Series, target, hash_bin=None, binarized_instance=False, z3=False):
        """
        Find a minimal subset of features (sufficient reason) that guarantees
        the tree predicts True for the given instance.
        """
        if hash_bin is None:
            hash_bin = self.binarization

        # Step 1: Get the full implicant (binarized literals)
        if not binarized_instance:
            implicant = list(self.binarize_instance(instance, hash_bin=hash_bin))
        else:
            implicant = list(instance)

        if z3:
            z3_vars, y, formula = self.to_z3_formula(binarized=False)
            x, classe_alvo = instance, target

            solver = Z3Solver()
            solver.add(formula)

            # atributos e valores da entrada: uma equação para cada variável
            atributos = [
                var == RealVal(x.iloc[nome])  for nome, var in z3_vars.items()
            ]
            # print(atributos)

            explicacao = atributos.copy()

            for atributo in atributos:
                # Testa se é possível garantir y == classe_alvo com os outros atributos
                # print(atributos)
                outras_features = [h2 for h2 in atributos if not h2.eq(atributo)]
                # print(outras_features)
                solver.push()
                solver.add(And(outras_features))
                solver.add(y != classe_alvo)  # queremos refutar isso
                resultado = solver.check()
                solver.pop()

                if resultado == unsat:
                    # print('entrou')
                    # é garantido ser igual a classe alvo mesmo sem o atributo, logo ele cai fora
                    explicacao.remove(atributo)
                    # explicacao.append(atributo)
                # else Pode dar outra classe sem o atributo. Logo, o atributo é necessário dadas as outras features
                # ou seja, mantemos ele em

            return explicacao




        # Sort literals by absolute value (optional, for deterministic removal)
        implicant.sort(key=abs, reverse=True)
        implicant = [int(x) for x in implicant]

        # Step 2: Build CNF from tree
        tree_cnf = CNF(from_clauses=self.to_cnf(hash_bin=hash_bin, negate_tree=not self.take_decision(instance)))

        # Step 3: Greedy removal of literals
        i = 0
        while i < len(implicant):
            candidate = implicant.copy()
            candidate.pop(i)  # try removing this literal

            if self.is_sufficient_reason(candidate, tree_cnf):
                # Removal is safe → keep literal removed
                implicant = candidate
            else:
                # Removal breaks sufficiency → keep literal
                i += 1

        # Step 4: Return minimized sufficient reason
        return np.array(sorted(implicant, key=abs))

    def to_z3_formula(self, binarized=True):
        if binarized:
            # Cria variáveis Z3 para cada feature binarizada
            z3_vars = {key: Bool(f"x_{abs(value)}") for key, value in self.binarization.items()}
            # classe_para_inteiro = {nome: i for i, nome in enumerate(clf.classes_)}
            paths = []

            def construir_path(node: DecisionNodeWrapper=self.root, condicoes=[]):
                # Se é folha
                if not node.var:
                    # classe_index = values[node].argmax()
                    classe_val = node.value
                    paths.append((And(condicoes), node.value))
                    return

                thres = node.var[1]
                var = z3_vars[node.var]

                # Ramificação à esquerda: x <= threshold
                cond_esq = condicoes + [var == True]
                construir_path(node.left, cond_esq)

                # Ramificação à direita: x > threshold
                cond_dir = condicoes + [var == False]
                construir_path(node.right, cond_dir)

            construir_path()

            # Predição final como variável: y
            y = Bool("y")
            formulas = [Implies(cond, y == BoolVal(classe)) for cond, classe in paths]
            return z3_vars, y, And(*formulas)

        # Cria variáveis Z3 para cada feature
        z3_vars = {idx: Real(f"f_{idx}") for idx in range(self.tree.n_features)}
        # classe_para_inteiro = {nome: i for i, nome in enumerate(clf.classes_)}
        paths = []

        def construir_path(node: DecisionNodeWrapper = self.root, condicoes=[]):
            # Se é folha
            if not node.var:
                paths.append((And(condicoes), int(node.value)))
                return

            feat_idx = node.var[0]
            thres = node.var[1]
            var = z3_vars[feat_idx]

            # Ramificação à esquerda: x <= threshold
            cond_esq = condicoes + [var <= thres]
            construir_path(node.left, cond_esq)

            # Ramificação à direita: x > threshold
            cond_dir = condicoes + [var > thres]
            construir_path(node.right, cond_dir)

        construir_path()

        # Predição final como variável: y
        y = Real("y")
        formulas = [Implies(cond, y == RealVal(classe)) for cond, classe in paths]
        return z3_vars, y, And(*formulas)

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
