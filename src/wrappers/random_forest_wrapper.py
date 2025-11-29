import numpy as np
from pandas import Series
from pysat.card import CardEnc
from pysat.formula import CNF, IDPool
from pysat.solvers import Solver
from sklearn.ensemble import RandomForestClassifier
from z3 import Solver as Z3Solver, Real, Bool, Int, If, Sum, And, Or, Not, RealVal, IntVal, BoolVal, substitute, sat, unsat, Z3_REAL_SORT, Z3_BOOL_SORT
from collections import defaultdict

from src.wrappers import DecisionTreeWrapper

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

    def find_direct_reason(self, instance: Series, z3=False):
        explanations = []
        predictions = []

        for tree in self.trees:
            expl, pred = tree.find_direct_reason(instance, self.binarization)
            explanations.append(list(expl))
            predictions.append(pred)

        # print(explanations)
        forest_class = max(np.array([True, False]), key=lambda c: sum(predictions == c))

        forest_direct_reason = []
        vote_count = 0

        for i in range(len(explanations)):
            if predictions[i] == forest_class:
                # print(i, np.array(explanations[i]))
                forest_direct_reason += explanations[i]
                vote_count += 1

        forest_direct_reason = list(set(forest_direct_reason))

        # return np.array(forest_direct_reason)
        return np.array(sorted(forest_direct_reason, key=abs)), bool(forest_class), vote_count

    def calc_cnf_h(self, instance) -> CNF:
        # count = 1000
        vpool = IDPool(occupied=[[0, (((self.n_bin_features // 100) + 1) * 100)]])

        # create fresh variables y1,...,ym
        y_vars = [vpool.id(f'y{i}') for i in range(self.n_trees)]

        cnf_implicant = []
        for yi, Ti in zip(y_vars, self.trees):
            tree_cnf = Ti.to_cnf(hash_bin=self.binarization, negate_tree=True)
            decision = Ti.take_decision(instance)
            for clause in tree_cnf:
                cnf_implicant.append([-yi] + clause)
                # cnf_implicant.append([yi if decision else -yi] + clause)

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
        # combined.extend([-target])

        with Solver(bootstrap_with=combined.clauses) as solver:
            return not solver.solve()

    def find_sufficient_reason(self, instance, target=None, hash_bin=None, binarized_instance=False, z3=False):
        """
        Retorna uma string no formato: "[64 == f_11, 19 == f_12]".
        Parâmetros:
        - instance: pandas.Series ou array-like da instância
        - target: classe alvo (se None, calcula usando o modelo)
        - hash_bin: mapeamento binário (feat,threshold) -> bin_id
        - binarized_instance: se True, instance já é vetor de literais
        - z3: se True, usa verificação SMT combinada para a RandomForest (modo "formal")
        """

        # ---- helper formatting ----
        def _fmt_pairs_as_z3(pairs):
            if not pairs:
                return "[]"
            items = []
            for feat_idx, val in pairs:
                try:
                    fval = float(val)
                    sval = str(int(fval)) if float(fval).is_integer() else str(fval)
                except Exception:
                    sval = str(val)
                items.append(f"{sval} == f_{int(feat_idx)}")
            return "[" + ", ".join(items) + "]"

        # ----------------------------
        if hash_bin is None:
            hash_bin = self.binarization

        # determinar target se não foi passado
        if target is None:
            try:
                pred = self.forest.predict([instance])
                # pred pode ser array-like
                target = bool(pred[0]) if (pred.shape == () or len(pred) == 1) else bool(pred[0])
            except Exception:
                # fallback: usar direct reason para inferir target
                _, target, _ = self.find_direct_reason(instance)

        # implicant inicial (literais binarizadas) para fallback CNF/guloso
        if not binarized_instance:
            implicant = list(self.binarize_instance(instance, reverse=True))
        else:
            implicant = list(instance)
        implicant = [int(x) for x in implicant]
        implicant.sort(key=abs, reverse=True)

        # rev_hash: id -> (feat_idx, threshold)
        rev_hash = {v: k for k, v in hash_bin.items()}

        # ---------- MODO Z3 (SMT) ----------
        if z3:
            try:
                # 1) pegar literais diretos das árvores que votam para a classe alvo
                direct_bin, forest_class, vote_count = self.find_direct_reason(instance)
                # se forest_class discordar do target pedido, usar target (o usuário pediu)
                # mas direct_bin é útil para extrair features do caminho
                # map id -> (feat,th)
                rev_hash_local = {v: k for k, v in self.binarization.items()}

                # path_feats: features presentes nas direct reasons (convertidas)
                path_feats = set()
                for lit in list(direct_bin):
                    lit_abs = abs(int(lit))
                    if lit_abs in rev_hash_local:
                        feat_idx, _ = rev_hash_local[lit_abs]
                        path_feats.add(int(feat_idx))

                # se direct_bin vazio, use implicant para extrair features (fallback)
                if len(path_feats) == 0:
                    for lit in implicant:
                        lit_abs = abs(int(lit))
                        if lit_abs in rev_hash:
                            feat_idx, _ = rev_hash[lit_abs]
                            path_feats.add(int(feat_idx))

                path_feats = sorted(path_feats)

                # 2) número de features e variáveis globais f_i
                n_feats = getattr(self.forest, "n_features_in_", None)
                if n_feats is None:
                    max_feat = 0
                    for (f, t) in self.binarization.keys():
                        if int(f) > max_feat:
                            max_feat = int(f)
                    n_feats = max_feat + 1
                global_f_vars = {i: Real(f"f_{i}") for i in range(n_feats)}

                # 3) construir fórmulas por árvore substituindo as f_i locais por global_f_vars
                tree_y_vars = []
                tree_formulas = []
                for ti, tree in enumerate(self.trees):
                    z3_vars_tree, y_tree, formula_tree = tree.to_z3_formula(binarized=False)
                    # coletar substituições (old_var -> global_f_vars[idx]) e (y -> y_ti)
                    subs = []
                    # z3_vars_tree keys podem ser ints (feat_idx)
                    for idx, old_var in z3_vars_tree.items():
                        if int(idx) in global_f_vars:
                            subs.append((old_var, global_f_vars[int(idx)]))
                    # renomear y para y_ti
                    if y_tree.sort().kind() == Z3_BOOL_SORT:
                        new_y = Bool(f"y_{ti}")
                    else:
                        new_y = Real(f"y_{ti}")
                    subs.append((y_tree, new_y))
                    formula_sub = substitute(formula_tree, *subs)
                    tree_y_vars.append(new_y)
                    tree_formulas.append(formula_sub)

                # 4) thresholds_by_feat e bounds para variáveis liberadas
                thresholds_by_feat = defaultdict(list)
                for (feat, th) in self.binarization.keys():
                    thresholds_by_feat[int(feat)].append(float(th))
                feature_bounds = {}
                for feat_idx in path_feats:
                    if feat_idx in thresholds_by_feat and len(thresholds_by_feat[feat_idx]) > 0:
                        ths = sorted(thresholds_by_feat[feat_idx])
                        feature_bounds[int(feat_idx)] = (ths[0] - 1.0, ths[-1] + 1.0)
                    else:
                        try:
                            base = float(instance.iloc[feat_idx])
                        except Exception:
                            base = float(instance.loc[feat_idx])
                        feature_bounds[int(feat_idx)] = (base - 1.0, base + 1.0)

                # 5) montar solver global com todas as fórmulas
                solver = Z3Solver()
                for fml in tree_formulas:
                    solver.add(fml)

                # majority threshold k
                k = (self.n_trees // 2) + 1

                # construir soma de indicadores de voto == target
                # Vamos criar b_i (Int) onde b_i == 1 iff y_i == target
                b_vars = []
                for i, yvar in enumerate(tree_y_vars):
                    b_i = Int(f"b_{i}")
                    # comparar yvar com target, observando o tipo
                    if yvar.sort().kind() == Z3_BOOL_SORT:
                        solver.add(b_i == If(yvar == BoolVal(bool(target)), IntVal(1), IntVal(0)))
                    else:
                        solver.add(b_i == If(yvar == RealVal(float(target)), IntVal(1), IntVal(0)))
                    b_vars.append(b_i)
                major_sum = Sum(*b_vars)
                neg_major = (major_sum < k)

                # 6) constraints iniciais: fixar features do path
                fixed_constraints = {}
                for feat_idx in path_feats:
                    v = global_f_vars[feat_idx]
                    try:
                        ival = float(instance.iloc[feat_idx])
                    except Exception:
                        ival = float(instance.loc[feat_idx])
                    fixed_constraints[feat_idx] = (v == RealVal(ival))

                # auxiliar: testa existência de contra-exemplo (MAJORIA < k) com um conjunto de constraints
                def exists_counterexample_with_constraints(constraints_dict):
                    solver.push()
                    if constraints_dict:
                        solver.add(And(*constraints_dict.values()))
                    solver.add(neg_major)
                    res = solver.check()
                    solver.pop()
                    return res == sat

                # checar consistência: se já existe contra-exemplo com tudo fixado -> fallback CNF/guloso
                if exists_counterexample_with_constraints(fixed_constraints):
                    # fallback para CNF/guloso
                    # (não tentamos remoção via Z3)
                    pass
                else:
                    # tentar remover cada feature do conjunto fixado (greedy)
                    for feat_idx in list(sorted(fixed_constraints.keys())):
                        other_constraints = {f: c for f, c in fixed_constraints.items() if f != feat_idx}
                        # para feat removida, colocamos apenas bounds
                        lb, ub = feature_bounds[feat_idx]
                        v = global_f_vars[feat_idx]
                        other_with_bounds = dict(other_constraints)
                        other_with_bounds[feat_idx] = And(v >= RealVal(lb), v <= RealVal(ub))
                        # se não existe contra-exemplo com essa feature "liberada" (i.e., Sum<k é UNSAT), então a feature NÃO é necessária
                        if not exists_counterexample_with_constraints(other_with_bounds):
                            fixed_constraints.pop(feat_idx, None)
                        else:
                            # precisa permanecer
                            pass

                # montar explicacao_legivel a partir do fixed_constraints restante
                explicacao_legivel = []
                for feat_idx in sorted(fixed_constraints.keys()):
                    try:
                        val = float(instance.iloc[feat_idx])
                    except Exception:
                        val = float(instance.loc[feat_idx])
                    explicacao_legivel.append((int(feat_idx), val))

                # se vazio -> fallback CNF/guloso
                if len(explicacao_legivel) == 0:
                    # iremos cair no fallback logo abaixo (código de CNF/guloso)
                    pass
                else:
                    return _fmt_pairs_as_z3(explicacao_legivel)
            except Exception:
                # se qualquer erro acontecer na rotina Z3, caímos no fallback
                pass

        # ---------- MODO NÃO-Z3 / FALLBACK CNF-GREEDY ----------
        # já temos implicant binarizado no começo
        implicant.sort(key=abs, reverse=True)
        implicant = [int(x) for x in implicant]

        # H: CNF construído para a floresta (prop. que testa se majority can be inverted)
        h_cnf = self.calc_cnf_h(instance)

        # greedy removal: remover literais do implicant mantendo suficiência
        i = 0
        while i < len(implicant):
            candidate = implicant.copy()
            candidate.pop(i)
            if self.is_sufficient_reason(candidate, h_cnf):
                implicant = candidate
            else:
                i += 1

        # converter literais binários em (feat_idx, valor_original_da_instância)
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

        # remover duplicatas mantendo a primeira ocorrência
        resultado_unico = []
        vistos = set()
        for feat_idx, val in resultado:
            if feat_idx not in vistos:
                vistos.add(feat_idx)
                resultado_unico.append((feat_idx, val))

        return _fmt_pairs_as_z3(resultado_unico)
    
    def is_majoritary_reason(self, candidate, h: CNF):
        term_clause = [[l] for l in candidate]

        combined = h.copy()   # ← importante!!!
        combined.extend(term_clause)

        with Solver(bootstrap_with=combined.clauses) as solver:
            return not solver.solve()


    def find_majoritary_reason(self, instance, target=None, hash_bin=None, binarized_instance=False, z3=True):
        """
        Retorna a menor Majoritary Reason:
        conjunto mínimo de features que, quando liberadas, permite que
        exista alguma instância que inverta a maioria da RandomForest.

        Estrutura e estilo EXATAMENTE iguais à find_sufficient_reason,
        para facilitar entendimento e comparação.
        """

        # ---- helper formatting (igual ao da sufficient_reason) ----
        def _fmt_pairs_as_z3(pairs):
            if not pairs:
                return "[]"
            items = []
            for feat_idx, val in pairs:
                try:
                    fval = float(val)
                    sval = str(int(fval)) if float(fval).is_integer() else str(fval)
                except Exception:
                    sval = str(val)
                items.append(f"{sval} == f_{int(feat_idx)}")
            return "[" + ", ".join(items) + "]"

        # ----------------------- PREPARAÇÃO -----------------------
        if hash_bin is None:
            hash_bin = self.binarization

        # predição original
        pred = self.forest.predict([instance])
        forest_class = bool(pred[0])
        target_flipped = 1 - int(forest_class)

        # implicant binarizado (igual ao sufficient reason)
        if not binarized_instance:
            implicant = list(self.binarize_instance(instance, reverse=True))
        else:
            implicant = list(instance)
        implicant = [int(x) for x in implicant]
        implicant.sort(key=abs, reverse=True)

        # rev_hash: id -> (feat_idx, threshold)
        rev_hash = {v: k for k, v in hash_bin.items()}

        # -------------------------------------------------------
        # ------------------------- MODO Z3 ---------------------
        # -------------------------------------------------------
        if z3:
            try:
                from z3 import (
                    Real, Bool, Int, Sum, And, Or, If,
                    RealVal, IntVal, BoolVal, substitute,
                    Solver as Z3Solver
                )
                from z3.z3util import Z3_BOOL_SORT

                # (1) Direct reason — igual ao sufficient_reason
                direct_bin, forest_class_dr, vote_count = self.find_direct_reason(instance)
                rev_hash_local = {v: k for k, v in self.binarization.items()}

                # extrair features dos literais diretos
                path_feats = set()
                for lit in list(direct_bin):
                    lit_abs = abs(int(lit))
                    if lit_abs in rev_hash_local:
                        feat_idx, _ = rev_hash_local[lit_abs]
                        path_feats.add(int(feat_idx))

                # fallback se vazio (igual sufficient)
                if len(path_feats) == 0:
                    for lit in implicant:
                        lit_abs = abs(int(lit))
                        if lit_abs in rev_hash:
                            feat_idx, _ = rev_hash[lit_abs]
                            path_feats.add(int(feat_idx))

                path_feats = sorted(path_feats)

                # (2) criar variáveis globais f_i
                n_feats = getattr(self.forest, "n_features_in_", None)
                if n_feats is None:
                    max_feat = 0
                    for (f, t) in self.binarization.keys():
                        max_feat = max(max_feat, int(f))
                    n_feats = max_feat + 1

                global_f_vars = {i: Real(f"f_{i}") for i in range(n_feats)}

                # (3) converter cada árvore para fórmula Z3 (idêntico)
                tree_y_vars = []
                tree_formulas = []
                for ti, tree in enumerate(self.trees):
                    z3_vars_tree, y_tree, formula_tree = tree.to_z3_formula(binarized=False)

                    subs = []
                    for idx, old_var in z3_vars_tree.items():
                        if int(idx) in global_f_vars:
                            subs.append((old_var, global_f_vars[int(idx)]))

                    if y_tree.sort().kind() == Z3_BOOL_SORT:
                        new_y = Bool(f"y_{ti}")
                    else:
                        new_y = Real(f"y_{ti}")
                    subs.append((y_tree, new_y))

                    formula_sub = substitute(formula_tree, *subs)
                    tree_y_vars.append(new_y)
                    tree_formulas.append(formula_sub)

                # (4) bounds por feature (idêntico)
                from collections import defaultdict
                thresholds_by_feat = defaultdict(list)
                for (f, t) in self.binarization.keys():
                    thresholds_by_feat[int(f)].append(float(t))

                feature_bounds = {}
                for feat_idx in path_feats:
                    if thresholds_by_feat[feat_idx]:
                        ths = sorted(thresholds_by_feat[feat_idx])
                        feature_bounds[feat_idx] = (ths[0] - 1.0, ths[-1] + 1.0)
                    else:
                        try:
                            base = float(instance.iloc[feat_idx])
                        except Exception:
                            base = float(instance.loc[feat_idx])
                        feature_bounds[feat_idx] = (base - 1.0, base + 1.0)

                # (5) construir solver
                solver = Z3Solver()
                for fml in tree_formulas:
                    solver.add(fml)

                # threshold de maioria
                k = (self.n_trees // 2) + 1

                # (6) indicadores b_i (mas agora para o target invertido)
                b_vars = []
                for i, y in enumerate(tree_y_vars):
                    b_i = Int(f"b_{i}")
                    if y.sort().kind() == Z3_BOOL_SORT:
                        solver.add(b_i == If(y == BoolVal(bool(target_flipped)), IntVal(1), IntVal(0)))
                    else:
                        solver.add(b_i == If(y == RealVal(float(target_flipped)), IntVal(1), IntVal(0)))
                    b_vars.append(b_i)

                major_sum = Sum(b_vars)
                majority_flip = (major_sum >= k)

                # -----------------------------------------------------------
                #                 GREEDY PARA MAJORITARY REASON
                # -----------------------------------------------------------

                # No sufficient reason você fixa tudo e tenta remover.
                # Aqui é o oposto: você libera tudo e tenta FIXAR apenas o necessário.

                fixadas = set()         # features que NÃO podem variar
                liberadas = set()       # features que podem variar

                # inicialmente: nada é fixado -> tudo pode variar
                # mas vamos testar uma a uma para ver se a variação é necessária

                for feat_idx in path_feats:
                    solver.push()

                    # liberar: essa variável pode variar dentro do intervalo
                    lb, ub = feature_bounds[feat_idx]
                    solver.add(global_f_vars[feat_idx] >= RealVal(lb))
                    solver.add(global_f_vars[feat_idx] <= RealVal(ub))

                    # porém fixamos as outras (que ainda não foram liberadas)
                    for ffix in fixadas:
                        try:
                            ival = float(instance.iloc[ffix])
                        except Exception:
                            ival = float(instance.loc[ffix])
                        solver.add(global_f_vars[ffix] == RealVal(ival))

                    solver.add(majority_flip)
                    res = solver.check()

                    solver.pop()

                    if res == True:   # SAT => liberar essa feature permite flip
                        liberadas.add(feat_idx)
                    else:             # UNSAT => essa feature deve ser fixada
                        fixadas.add(feat_idx)

                # construir explicação legível
                explicacao = []
                for feat_idx in sorted(liberadas):
                    try:
                        val = float(instance.iloc[feat_idx])
                    except Exception:
                        val = float(instance.loc[feat_idx])
                    explicacao.append((feat_idx, val))

                if len(explicacao) > 0:
                    return _fmt_pairs_as_z3(explicacao)

            except Exception:
                # qualquer erro → fallback CNF/guloso
                pass

        # -------------------------------------------------------
        # ------------- MODO FALLBACK CNF/GULOSO ----------------
        # -------------------------------------------------------

        #igual à sufficient_reason, só que invertendo regra

        h_cnf = self.calc_cnf_h(instance)

        # greedy: features cujo valor altera possibilidade de inverter maioria
        resultado = []
        teste = implicant.copy()

        i = 0
        while i < len(teste):
            cand = teste.copy()
            # remover literal = liberar aquela feature
            cand.pop(i)
            if not self.is_sufficient_reason(cand, h_cnf):
                # se deixar de fixar esse literal permite inverter → relevante
                resultado.append(teste[i])
                teste = cand
            else:
                i += 1

        # converter para pares (feat_idx, valor original)
        final = []
        vistos = set()
        for lit in resultado:
            lit_abs = abs(int(lit))
            if lit_abs in rev_hash:
                feat_idx, _ = rev_hash[lit_abs]
                if feat_idx not in vistos:
                    vistos.add(feat_idx)
                    try:
                        val = float(instance.iloc[feat_idx])
                    except Exception:
                        val = float(instance.loc[feat_idx])
                    final.append((feat_idx, val))

        return _fmt_pairs_as_z3(final)


    def __len__(self):
        return sum([len(tree) for tree in self.trees])
