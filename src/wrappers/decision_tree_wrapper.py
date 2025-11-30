import numpy as np
from pandas import Series
from pysat.formula import CNF
from pysat.solvers import Solver as PySatSolver
from sklearn.tree import DecisionTreeClassifier
from z3 import Solver as Z3Solver, BoolRef, Real, Int, And, Or, Not, Implies, simplify, RealVal, IntVal, sat, unsat, BoolVal, Bools, Bool, Z3_REAL_SORT, Z3_BOOL_SORT
from collections import defaultdict


class DecisionNodeWrapper:
    def __init__(self, var, left=None, right=None, value=None):
        self.var = var
        self.left = left
        self.right = right
        self.value = value

    def __repr__(self):
        return f"{self.var} | [[{self.left} , {self.right}]] | {self.value}"


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
        Retorna sempre uma string no formato pedido, ex: "[64 == f_11, 19 == f_12]".
        No modo z3: executa a ablação apenas sobre os atributos do caminho da instância.
        No modo não-z3: usa o fluxo CNF/guloso e mapeia o resultado para o mesmo formato.
        """

        def _fmt_pairs_as_z3(pairs):
            if not pairs:
                return "[]"
            items = []
            for feat_idx, val in pairs:
                try:
                    fval = float(val)
                    if float(fval).is_integer():
                        sval = str(int(fval))
                    else:
                        sval = str(fval)
                except Exception:
                    sval = str(val)
                items.append(f"{sval} == f_{int(feat_idx)}")
            return "[" + ", ".join(items) + "]"

        if hash_bin is None:
            hash_bin = self.binarization

        # passo inicial: implicant binarizado (usado no fallback / modo não-z3)
        if not binarized_instance:
            implicant = list(self.binarize_instance(instance, hash_bin=hash_bin))
        else:
            implicant = list(instance)

        # reverse map hash_bin id -> (feat_idx, threshold)
        rev_hash = {v: k for k, v in hash_bin.items()}

        # função auxiliar: percorre a árvore com a instância e retorna a lista de features do caminho
        def _path_features_from_instance(x_instance):
            node = self.root
            path_feats = []
            while node and node.var:
                feat_idx, th = node.var
                path_feats.append(int(feat_idx))
                # seguir ramificação conforme a instância
                try:
                    val = x_instance.iloc[feat_idx]
                except Exception:
                    val = x_instance.loc[feat_idx]
                if val <= th:
                    node = node.left
                else:
                    node = node.right
            return path_feats

        # -----------------------
        # MODO Z3: trabalhar apenas com features do caminho
        # -----------------------
        if z3:

            z3_vars, y, formula = self.to_z3_formula(binarized=False)
            x = instance

            solver = Z3Solver()
            solver.add(formula)

            # extrair apenas features do caminho (pode haver repetições; usamos set)
            path_feat_list = _path_features_from_instance(x)
            path_feats = sorted(set(path_feat_list))  # vantagens: único por índice

            # 1) construir atributo_constraints somente para essas features do caminho
            atributo_constraints = {}
            for feat_idx in path_feats:
                var = z3_vars[feat_idx]
                try:
                    val = float(x.iloc[feat_idx])
                except Exception:
                    val = float(x.loc[feat_idx])

                if var.sort().kind() == Z3_REAL_SORT:
                    atributo_constraints[int(feat_idx)] = (var == RealVal(val))
                else:
                    atributo_constraints[int(feat_idx)] = (var == BoolVal(bool(val)))

            # 2) bounds razoáveis a partir dos thresholds conhecidos (apenas para manter segurança)
            thresholds_by_feat = defaultdict(list)
            for (feat, th) in hash_bin.keys():
                thresholds_by_feat[int(feat)].append(float(th))

            feature_bounds = {}
            for feat_idx in path_feats:
                if feat_idx in thresholds_by_feat and len(thresholds_by_feat[feat_idx]) > 0:
                    ths = sorted(thresholds_by_feat[feat_idx])
                    feature_bounds[int(feat_idx)] = (ths[0] - 1.0, ths[-1] + 1.0)
                else:
                    try:
                        base = float(x.iloc[feat_idx])
                    except Exception:
                        base = float(x.loc[feat_idx])
                    feature_bounds[int(feat_idx)] = (base - 1.0, base + 1.0)

            # 3) explicacao_constraints inicia com todas as features do caminho fixadas
            explicacao_constraints = dict(atributo_constraints)

            # 4) ablação: remover cada atributo do caminho e testar
            for feat_idx in list(atributo_constraints.keys()):
                outras = [c for f, c in atributo_constraints.items() if f is not feat_idx]

                solver.push()
                if outras:
                    solver.add(And(*outras))

                # para a feature removida, adicionamos apenas bounds (se Real)
                var = z3_vars[feat_idx]
                if var.sort().kind() == Z3_REAL_SORT:
                    lb, ub = feature_bounds[feat_idx]
                    solver.add(var >= RealVal(lb))
                    solver.add(var <= RealVal(ub))

                # adicionar y != target e checar
                if y.sort().kind() == Z3_REAL_SORT:
                    solver.add(y != RealVal(float(target)))
                elif y.sort().kind() == Z3_BOOL_SORT:
                    solver.add(y != BoolVal(bool(target)))
                else:
                    solver.add(y != RealVal(float(target)))

                res = solver.check()
                solver.pop()

                if res == unsat:
                    explicacao_constraints.pop(feat_idx, None)
                else:
                    pass

            # 5) converte explicacao_constraints -> lista (feat_idx, valor_inst)
            explicacao_legivel = []
            for feat_idx in sorted(explicacao_constraints.keys()):
                try:
                    val = float(instance.iloc[feat_idx])
                except Exception:
                    val = float(instance.loc[feat_idx])
                explicacao_legivel.append((int(feat_idx), val))

            # 6) fallback CNF/guloso caso fique vazia (mapeando apenas literais do caminho)
            if len(explicacao_legivel) == 0:
                # usar o implicant binarizado original (ele contém literais do caminho)
                implicant.sort(key=abs, reverse=True)
                implicant = [int(x) for x in implicant]
                tree_cnf = CNF(
                    from_clauses=self.to_cnf(hash_bin=hash_bin, negate_tree=not self.take_decision(instance)))

                i = 0
                while i < len(implicant):
                    candidate = implicant.copy()
                    candidate.pop(i)
                    if self.is_sufficient_reason(candidate, tree_cnf):
                        implicant = candidate
                    else:
                        i += 1

                resultado = []
                for lit in implicant:
                    lit_abs = abs(int(lit))
                    if lit_abs in rev_hash:
                        feat_idx, _ = rev_hash[lit_abs]
                        try:
                            val = float(instance.iloc[int(feat_idx)])
                        except Exception:
                            val = float(instance.loc[int(feat_idx)])
                        resultado.append((int(feat_idx), val))
                return _fmt_pairs_as_z3(resultado)

            return _fmt_pairs_as_z3(explicacao_legivel)

        # -----------------------
        # MODO NÃO-Z3: usa CNF/guloso mas retorna no mesmo formato pedido
        # -----------------------
        implicant.sort(key=abs, reverse=True)
        implicant = [int(x) for x in implicant]

        tree_cnf = CNF(from_clauses=self.to_cnf(hash_bin=hash_bin, negate_tree=not self.take_decision(instance)))

        i = 0
        while i < len(implicant):
            candidate = implicant.copy()
            candidate.pop(i)
            if self.is_sufficient_reason(candidate, tree_cnf):
                implicant = candidate
            else:
                i += 1

        resultado = []
        for lit in implicant:
            lit_abs = abs(int(lit))
            if lit_abs in rev_hash:
                feat_idx, _ = rev_hash[lit_abs]
                try:
                    val = float(instance.iloc[int(feat_idx)])
                except Exception:
                    val = float(instance.loc[int(feat_idx)])
                resultado.append((int(feat_idx), val))

        return _fmt_pairs_as_z3(resultado)

    def to_z3_formula(self, binarized=False):
        # Caso binarizado
        if binarized:
            # Cria variáveis Z3 para cada feature binarizada
            z3_vars = {key: Bool(f"x_{abs(value)}") for key, value in self.binarization.items()}
            # variaveis bool criadas a partir dos thresholds
            bool_vars = {}
            # classe_para_inteiro = {nome: i for i, nome in enumerate(clf.classes_)}
            paths = []

            # nome das variaveis booleanas no Z3
            def var_name(feat_idx, thres, op):
                return f"f{feat_idx} {op} {thres:.3f}"

            def construir_path(node, condicoes=[]):
                # Se é folha
                if not node.var:
                    # valor da classe já vem binarizado pelo wrapper
                    classe = bool(node.value)
                    paths.append((And(condicoes), classe))
                    return

                feat_idx, thres = node.var
                left_name = var_name(feat_idx, thres, "<=")
                right_name = var_name(feat_idx, thres, ">")

                if left_name not in bool_vars:
                    bool_vars[left_name] = Bool(left_name)
                if right_name not in bool_vars:
                    bool_vars[right_name] = Not(bool_vars[left_name])

                # Ramificação à esquerda: x <= threshold
                cond_esq = condicoes + [bool_vars[left_name]]
                construir_path(node.left, cond_esq)

                # Ramificação à direita: x > threshold
                cond_dir = condicoes + [Not(bool_vars[left_name])]
                construir_path(node.right, cond_dir)

            construir_path(self.root)

            # Predição final como variável: y
            y = Bool("y")
            formulas = []

            # Coerência entre thresholds
            thresholds_by_feat = defaultdict(list)

            for var_name in bool_vars.keys():
                if " <=" in var_name:
                    feat_name, thres_str = var_name.split(" <=")
                    thresholds_by_feat[feat_name].append(float(thres_str))

            # constraints de monotonicidade
            for feat, ths in thresholds_by_feat.items():
                ths.sort()
                for i in range(len(ths) - 1):
                    v1 = bool_vars[f"{feat} <= {ths[i]:.3f}"]
                    v2 = bool_vars[f"{feat} <= {ths[i + 1]:.3f}"]
                    formulas.append(Implies(v1, v2))

            # caminho -> classe (binarizado)
            for cond, classe in paths:
                formulas.append(Implies(cond, y == BoolVal(classe)))

            # lista apenas dos caminhos
            conds_only = [cond for cond, _ in paths]

            # ao menos um caminho verdadeiro
            formulas.append(Or(*conds_only))

            # monocidade
            for i in range(len(conds_only) - 1):
                for j in range(i + 1, len(conds_only)):
                    formulas.append(Not(And(conds_only[i], conds_only[j])))

            return bool_vars, y, And(*formulas)

        # Cria variáveis Z3 para cada feature (Caso não binarizado)
        z3_vars = {idx: Real(f"f_{idx}") for idx in range(self.tree.n_features)}
        # variaveis bool criadas a partir dos thresholds
        bool_vars = {}
        # classe_para_inteiro = {nome: i for i, nome in enumerate(clf.classes_)}
        paths = []

        def cond_name_real(feat_idx, thres, op):
            return f"f{feat_idx} {op} {thres:.3f}"

        def construir_path(node, condicoes=[]):
            # Se é folha
            if not node.var:
                classe = int(node.value)
                paths.append((And(condicoes), classe))
                return

            feat_idx, thres = node.var
            var = z3_vars[feat_idx]

            nome_esq = cond_name_real(feat_idx, thres, "<=")
            nome_dir = cond_name_real(feat_idx, thres, ">")

            if nome_esq not in bool_vars:
                bool_vars[nome_esq] = Bool(nome_esq)
            if nome_dir not in bool_vars:
                bool_vars[nome_dir] = Not(bool_vars[nome_esq])

            # Ramificação à esquerda: x <= threshold
            cond_esq = condicoes + [var <= thres, bool_vars[nome_esq]]
            construir_path(node.left, cond_esq)

            # Ramificação à direita: x > threshold
            cond_dir = condicoes + [var > thres, Not(bool_vars[nome_esq])]
            construir_path(node.right, cond_dir)

        construir_path(self.root)

        # Predição final como variável: y
        y = Real("y")
        formulas = []

        # Coerência entre thresholds
        thresholds_by_feat = defaultdict(list)

        for var_name in bool_vars.keys():
            if " <=" in var_name:
                feat_name, thres_str = var_name.split(" <=")
                thresholds_by_feat[feat_name].append(float(thres_str))

        # constraints de monotonicidade
        for feat, ths in thresholds_by_feat.items():
            ths.sort()
            for i in range(len(ths) - 1):
                v1 = bool_vars[f"{feat} <= {ths[i]:.3f}"]
                v2 = bool_vars[f"{feat} <= {ths[i + 1]:.3f}"]
                formulas.append(Implies(v1, v2))

        # caminho -> classe (nao binarizado)
        for cond, classe in paths:
            formulas.append(Implies(cond, y == RealVal(classe)))

        # lista apenas dos caminhos
        conds_only = [cond for cond, _ in paths]

        # ao menos um caminho verdadeiro
        formulas.append(Or(*conds_only))

        # monocidade
        for i in range(len(conds_only) - 1):
            for j in range(i + 1, len(conds_only)):
                formulas.append(Not(And(conds_only[i], conds_only[j])))

        return z3_vars, y, And(*formulas)
    
    def predict_z3(self, x_vars):
        """
        Dado um vetor de variáveis Z3 x_vars (ex.: [Real('x_0'), Real('x_1'), ...]),
        retorna uma Z3 expression que avalia ao rótulo da árvore (0 ou 1) para essas variáveis.

        Usa If(...) recursivamente para codificar as decisões dos nós.
        """

        from z3 import If, RealVal, IntVal

        def _rec(node):
            # se nó folha, retornar o valor da classe como IntVal(0/1)
            if node is None:
                # deveria não acontecer
                return IntVal(0)
            if node.value is not None:
                # node.value no seu wrapper é bool — converte para 0/1
                return IntVal(1) if bool(node.value) else IntVal(0)

            # nó interno: node.var é (feat_idx, threshold)
            feat_idx, th = node.var
            # pegar var Z3 correspondente — assumimos x_vars indexável por feat_idx
            xv = x_vars[int(feat_idx)]
            # comparação: xv <= th  (use RealVal para o threshold)
            cond = xv <= RealVal(float(th))
            # construir expressões para as duas ramificações
            left_expr = _rec(node.left)
            right_expr = _rec(node.right)
            # retornar If(cond, left_expr, right_expr)
            return If(cond, left_expr, right_expr)

        return _rec(self.root)


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
