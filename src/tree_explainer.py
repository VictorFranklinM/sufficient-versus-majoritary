import copy

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from z3 import *
from z3 import sat
from z3 import Solver as Z3Solver
from collections import defaultdict

from src.wrappers import DecisionTreeWrapper, RandomForestWrapper

MAX_RANDOM_FOREST_DEPTH = 8

sign = lambda x, predicate: x if predicate else -x

def verify_sufficiency_random_perturbation(rf_clf, rf_wrapper, instance, explan_pairs, n_trials=1000, feature_bounds=None, rng=None):
    """
    Teste empírico: fixa os pares explan_pairs e sorteia as demais features dentro de bounds.
    Retorna (ok True/False, counterexample or None).
    """
    if rng is None:
        rng = np.random.RandomState(42)
    # determinar target pela floresta se não passado
    target = safe_rf_predict(rf_clf, instance)[0]

    # transformar explan_pairs em set de índices
    expl_feats = set(int(f) for f, _ in explan_pairs)

    # montar bounds se não vierem
    if feature_bounds is None:
        feature_bounds = {}
        try:
            # usar thresholds do wrapper para bounds razoáveis
            thresholds_by_feat = {}
            for (feat, th) in rf_wrapper.binarization.keys():
                thresholds_by_feat.setdefault(int(feat), []).append(float(th))
            n_feats = getattr(rf_clf, "n_features_in_", instance.shape[0])
            for feat in range(n_feats):
                if feat in thresholds_by_feat:
                    ths = sorted(thresholds_by_feat[feat])
                    feature_bounds[feat] = (ths[0] - 1.0, ths[-1] + 1.0)
                else:
                    v = float(instance.iloc[feat])
                    delta = max(abs(v) * 0.1, 1e-3)
                    feature_bounds[feat] = (v - delta, v + delta)
        except Exception:
            # fallback simples baseado na instância
            for feat in range(len(instance)):
                v = float(instance.iloc[feat])
                delta = max(abs(v) * 0.1, 1e-3)
                feature_bounds[feat] = (v - delta, v + delta)

    # executar perturbações
    for t in range(n_trials):
        inst2 = instance.astype(float).copy(deep=True)
        for feat in range(len(instance)):
            if feat in expl_feats:
                # fixa
                continue
            low, high = feature_bounds[feat]
            inst2.iloc[feat] = rng.uniform(low, high)
        pred = safe_rf_predict(rf_clf, inst2)[0]
        if pred != target:
            return False, inst2
    return True, None
# ----------------------------
def safe_rf_predict(rf_clf, series_or_df):
    """
    Usa rf_clf.predict com segurança, criando um DataFrame com os nomes de feature
    se possível. Aceita como series_or_df:
      - pd.Series (uma instância)
      - pd.DataFrame (uma ou várias instâncias)
      - np.ndarray ou lista (uma instância posicional)
    Retorna o array de predições do sklearn.
    """
    # caso seja Series (uma instância)
    if isinstance(series_or_df, pd.Series):
        values = series_or_df.values
        # tentar criar DataFrame com feature names do modelo
        try:
            cols = rf_clf.feature_names_in_
            df = pd.DataFrame([values], columns=cols)
        except Exception:
            df = pd.DataFrame([values])
        return rf_clf.predict(df)

    # caso seja DataFrame (pode ser 1xN ou MxN)
    if isinstance(series_or_df, pd.DataFrame):
        df = series_or_df.copy()
        # se o modelo sabe os nomes, ajusta as colunas posicionalmente (evita warnings)
        try:
            cols = rf_clf.feature_names_in_
            if df.shape[1] == len(cols) and list(df.columns) != list(cols):
                df.columns = cols
        except Exception:
            pass
        return rf_clf.predict(df)

    # caso seja lista/np.ndarray (uma instância posicional)
    if isinstance(series_or_df, (list, tuple, np.ndarray)):
        vals = np.array(series_or_df).ravel()
        try:
            cols = rf_clf.feature_names_in_
            df = pd.DataFrame([vals], columns=cols)
        except Exception:
            df = pd.DataFrame([vals])
        return rf_clf.predict(df)

    # fallback: tentar passar diretamente (sklearn aceitará lista posicional)
    return rf_clf.predict([series_or_df])

def evaluate_explanations(rf_clf, rf_wrapper, X_test, n_instances=100, timeout_ms=3000, trials=500):
    rng = np.random.RandomState(0)
    results = {"idx": [], "is_suff_z3": [], "empirical_ok": [], "is_minimal": []}
    N = min(n_instances, len(X_test))
    for i in range(N):
        inst = X_test.iloc[i]
        expl_str = rf_wrapper.find_sufficient_reason(inst, z3=True)  # prefer Z3 explanation
        expl_pairs = parse_expl_string(expl_str)
        # formal check
        is_suff, ce = verify_sufficiency_z3_forest(rf_wrapper, inst, expl_pairs, target=safe_rf_predict(rf_clf, inst)[0], timeout_ms=timeout_ms)
        # empirical check
        emp_ok, emp_ce = verify_sufficiency_random_perturbation(rf_clf, rf_wrapper, inst, explan_pairs=expl_pairs, n_trials=trials, rng=rng)
        # minimality (formal, per-feature)
        is_minimal, necessity = verify_minimality_z3(rf_wrapper, inst, expl_pairs, target=safe_rf_predict(rf_clf, inst)[0], timeout_ms=timeout_ms)
        results["idx"].append(i)
        results["is_suff_z3"].append(is_suff)
        results["empirical_ok"].append(emp_ok)
        results["is_minimal"].append(is_minimal)
    # sumariza
    import pandas as pd
    df = pd.DataFrame(results)
    print("Z3 pass rate:", df["is_suff_z3"].mean())
    print("Empirical pass rate:", df["empirical_ok"].mean())
    print("Minimality pass rate:", df["is_minimal"].mean())
    return df

def parse_expl_string(s):
    """
    Converte string do formato "[val == f_i, ...]" para [(i,val),...].
    Aceita também lista/np.array já no formato.
    """
    import numpy as np
    if s is None:
        return []
    if isinstance(s, (list, tuple, np.ndarray)):
        return list(s)
    s = s.strip()
    if s == "[]" or s == "":
        return []
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1].strip()
    if s == "":
        return []
    pairs = []
    for item in s.split(","):
        item = item.strip()
        if "==" not in item:
            continue
        left, right = item.split("==")
        left = left.strip(); right = right.strip()
        if not right.startswith("f_"):
            continue
        try:
            feat_idx = int(right[2:])
            val = float(left.replace("'", "").replace('"', ""))
            pairs.append((feat_idx, val))
        except Exception:
            continue
    return pairs

def _model_eval_to_float(m, var):
    """
    Tenta extrair float do model.eval(var).
    """
    try:
        v = m.eval(var, model_completion=True)
        # as_decimal pode vir com ? if repeating; fallback para float(str)
        try:
            s = v.as_decimal(20)
            if s.endswith('?'):
                # repetir: converte via str
                return float(str(v))
            return float(s)
        except Exception:
            return float(str(v))
    except Exception:
        return None

def verify_sufficiency_z3_forest(rf_wrapper, instance, explan_pairs, target=None, timeout_ms=3000):
    """
    Verifica com Z3 se explan_pairs ([(feat_idx,val),...]) garante a predição target da RandomForest.
    Retorna (is_sufficient:bool, counterexample:dict or None)
    - is_sufficient == True -> prova (UNSAT) que não existe contra-exemplo (dentro dos bounds)
    - is_sufficient == False -> existe contra-exemplo; retorna model como dict feat->value
    Observações:
    - usa DecisionTreeWrapper.to_z3_formula para montar as fórmulas por árvore
    - timeout_ms: milissegundos para Z3 (passa solver.set("timeout", timeout_ms))
    """
    # inferir target se não foi dado
    if target is None:
        rf = getattr(rf_wrapper, "forest", None)
        if rf is None:
            raise ValueError("target required if rf_wrapper has no underlying sklearn forest")
        # garantir DataFrame com nomes (evita warning)
        try:
            target = rf.predict(pd.DataFrame([instance]))[0]
        except Exception:
            target = rf.predict([instance])[0]

    # extrair lista de features da explicacao
    path_feats = sorted([int(f) for f, _ in explan_pairs])

    # montar variaveis globais f_i (Real)
    n_feats = getattr(rf_wrapper.forest, "n_features_in_", None)
    if n_feats is None:
        max_feat = 0
        for (f, t) in rf_wrapper.binarization.keys():
            if int(f) > max_feat:
                max_feat = int(f)
        n_feats = max_feat + 1
    global_f_vars = {i: Real(f"f_{i}") for i in range(n_feats)}

    # construir fórmulas por árvore (substitute variables locais -> global_f_vars, y -> y_i)
    tree_y_vars = []
    tree_formulas = []
    for ti, tree in enumerate(rf_wrapper.trees):
        z3_vars_tree, y_tree, formula_tree = tree.to_z3_formula(binarized=False)
        subs = []
        for idx, old_var in z3_vars_tree.items():
            if int(idx) in global_f_vars:
                subs.append((old_var, global_f_vars[int(idx)]))
        # renomear y
        if y_tree.sort().kind() == Z3_BOOL_SORT:
            new_y = Bool(f"y_{ti}")
        else:
            new_y = Real(f"y_{ti}")
        subs.append((y_tree, new_y))
        formula_sub = substitute(formula_tree, *subs)
        tree_y_vars.append(new_y)
        tree_formulas.append(formula_sub)

    solver = Z3Solver()
    solver.set("timeout", int(timeout_ms))
    for f in tree_formulas:
        solver.add(f)

    # majority threshold
    k = (len(tree_y_vars) // 2) + 1

    # criar b_i onde b_i == 1 iff y_i == target
    b_vars = []
    for i, yvar in enumerate(tree_y_vars):
        bi = Int(f"b_{i}")
        if yvar.sort().kind() == Z3_BOOL_SORT:
            solver.add(bi == If(yvar == BoolVal(bool(target)), IntVal(1), IntVal(0)))
        else:
            solver.add(bi == If(yvar == RealVal(float(target)), IntVal(1), IntVal(0)))
        b_vars.append(bi)
    major_sum = Sum(*b_vars)

    # bounds por feature (usar thresholds do wrapper)
    thresholds_by_feat = defaultdict(list)
    for (feat, th) in rf_wrapper.binarization.keys():
        thresholds_by_feat[int(feat)].append(float(th))
    feature_bounds = {}
    for feat in path_feats:
        if feat in thresholds_by_feat and len(thresholds_by_feat[feat]) > 0:
            ths = sorted(thresholds_by_feat[feat])
            feature_bounds[int(feat)] = (ths[0] - 1.0, ths[-1] + 1.0)
        else:
            try:
                base = float(instance.iloc[feat])
            except Exception:
                base = float(instance.loc[feat])
            feature_bounds[int(feat)] = (base - 1.0, base + 1.0)

    # constraints que fixam as features de explan_pairs
    fixed_constraints = []
    for feat_idx, val in explan_pairs:
        v = global_f_vars[int(feat_idx)]
        fixed_constraints.append(v == RealVal(float(val)))

    # queremos saber se EXISTE atribuição que faz major_sum < k (i.e., majority invertida)
    solver.push()
    if fixed_constraints:
        solver.add(And(*fixed_constraints))
    solver.add(major_sum < k)
    res = solver.check()
    if res == sat:
        model = solver.model()
        # constuir contra-exemplo: tentar extrair valor para cada f_i
        ce = {}
        for i in range(n_feats):
            v = global_f_vars[i]
            try:
                val = _model_eval_to_float(model, v)
                if val is None:
                    # model might not assign it; try model_completion
                    eval_v = model.eval(v, model_completion=True)
                    try:
                        val = float(str(eval_v))
                    except Exception:
                        val = None
                ce[i] = val
            except Exception:
                ce[i] = None
        solver.pop()
        return False, ce
    else:
        solver.pop()
        return True, None
    
def verify_majoritary_z3_forest(rf_wrapper, instance, explan_pairs, target=None, timeout_ms=3000):
    """
    Versão reforçada / robusta de verificação Z3.
    Retorna (is_sufficient: bool, counterexample: dict or None)
    """
    # --- inferir target se necessário (sklearn predict) ---
    if target is None:
        rf = getattr(rf_wrapper, "forest", None)
        if rf is None:
            raise ValueError("target required if rf_wrapper has no underlying sklearn forest")
        try:
            target = rf.predict(pd.DataFrame([instance]))[0]
        except Exception:
            target = rf.predict([instance])[0]

    # converte target para boolean e int (0/1) usados em Z3
    target_bool = bool(target)
    target_as_boolval = BoolVal(target_bool)
    target_as_realval = RealVal(float(int(target_bool)))  # 0.0 ou 1.0

    # features list (vindas da explicação)
    path_feats = sorted([int(f) for f, _ in explan_pairs])

    # número de features e vars globais
    n_feats = getattr(rf_wrapper.forest, "n_features_in_", None)
    if n_feats is None:
        max_feat = -1
        for (f, t) in rf_wrapper.binarization.keys():
            if int(f) > max_feat:
                max_feat = int(f)
        n_feats = max_feat + 1
    global_f_vars = {i: Real(f"f_{i}") for i in range(n_feats)}

    # montar fórmulas por árvore (substituir var_local -> global_f_vars, y->y_i)
    tree_y_vars = []
    tree_formulas = []
    for ti, tree in enumerate(rf_wrapper.trees):
        z3_vars_tree, y_tree, formula_tree = tree.to_z3_formula(binarized=False)
        subs = []
        for idx, old_var in z3_vars_tree.items():
            idx_int = int(idx)
            if idx_int in global_f_vars:
                subs.append((old_var, global_f_vars[idx_int]))
        # renomear y para y_{ti}
        if y_tree.sort().kind() == Z3_BOOL_SORT:
            new_y = Bool(f"y_{ti}")
        else:
            new_y = Real(f"y_{ti}")
        subs.append((y_tree, new_y))
        formula_sub = substitute(formula_tree, *subs)
        tree_y_vars.append(new_y)
        tree_formulas.append(formula_sub)

    # criar solver e adicionar fórmulas das árvores
    solver = Z3Solver()
    solver.set("timeout", int(timeout_ms))
    for f in tree_formulas:
        solver.add(f)

    # majority threshold k
    k = (len(tree_y_vars) // 2) + 1

    # criar b_i onde b_i == 1 iff y_i == target_bool (compativel com y sort)
    b_vars = []
    for i, yvar in enumerate(tree_y_vars):
        bi = Int(f"b_{i}")
        if yvar.sort().kind() == Z3_BOOL_SORT:
            solver.add(bi == If(yvar == target_as_boolval, IntVal(1), IntVal(0)))
        else:
            solver.add(bi == If(yvar == target_as_realval, IntVal(1), IntVal(0)))
        b_vars.append(bi)
    major_sum = Sum(*b_vars)

    # --- bounds por feature (usar thresholds do wrapper quando existir) ---
    thresholds_by_feat = defaultdict(list)
    for (feat, th) in rf_wrapper.binarization.keys():
        thresholds_by_feat[int(feat)].append(float(th))

    feature_bounds = {}
    # garanto bounds para todas as features (não apenas path_feats)
    for feat in range(n_feats):
        if feat in thresholds_by_feat and len(thresholds_by_feat[feat]) > 0:
            ths = sorted(thresholds_by_feat[feat])
            feature_bounds[feat] = (ths[0] - 1.0, ths[-1] + 1.0)
        else:
            # usar valor da instância se possível; caso contrário usar (-1e6, +1e6)
            try:
                base = float(instance.iloc[feat])
            except Exception:
                try:
                    base = float(instance.loc[feat])
                except Exception:
                    base = 0.0
            delta = max(abs(base) * 0.1, 1e-3)
            feature_bounds[feat] = (base - delta, base + delta)

    # constraints que FIXAM as features da explicação (explan_pairs)
    fixed_constraints = []
    for feat_idx, val in explan_pairs:
        idx = int(feat_idx)
        v = global_f_vars[idx]
        fixed_constraints.append(v == RealVal(float(val)))

    # --- testar existência de contra-exemplo: existe x tal que (fixed_constraints) & (major_sum < k) ? ---
    solver.push()
    if fixed_constraints:
        solver.add(And(*fixed_constraints))

    # garantir que cada variável não-fixada esteja dentro dos bounds para que solver ache soluções válidas
    bounds_constraints = []
    for feat in range(n_feats):
        # se a feature está fixada em fixed_constraints já coberta; mesmo assim bounds são seguros
        lb, ub = feature_bounds[feat]
        v = global_f_vars[feat]
        bounds_constraints.append(And(v >= RealVal(lb), v <= RealVal(ub)))
    if bounds_constraints:
        solver.add(And(*bounds_constraints))

    # adicionar condição de existência de contra-exemplo: majority < k
    solver.add(major_sum < k)

    res = solver.check()
    if res == sat:
        model = solver.model()
        # construir contra-exemplo (feature -> valor)
        ce = {}
        for i in range(n_feats):
            v = global_f_vars[i]
            try:
                val = _model_eval_to_float(model, v)
                if val is None:
                    eval_v = model.eval(v, model_completion=True)
                    try:
                        val = float(str(eval_v))
                    except Exception:
                        val = None
                ce[i] = val
            except Exception:
                ce[i] = None
        solver.pop()
        return False, ce
    elif res == unsat:
        solver.pop()
        return True, None
    else:
        # unknown / timeout -> inconclusivo: tratar como não provado (retorna False, None)
        solver.pop()
        return False, None


def verify_minimality_z3(rf_wrapper, instance, explan_pairs, target=None, timeout_ms=3000):
    """
    Checa minimalidade formal: para cada feature em explan_pairs testa se E \\ {f} ainda é suficiente.
    Retorna (is_minimal:bool, necessity: dict feat -> (is_necessary(bool), counterexample_or_None))
    """
    # inferir target se necessario
    if target is None:
        rf = getattr(rf_wrapper, "forest", None)
        if rf is None:
            raise ValueError("target required if rf_wrapper has no underlying sklearn forest")
        try:
            target = rf.predict(pd.DataFrame([instance]))[0]
        except Exception:
            target = rf.predict([instance])[0]

    expl_feats = [int(f) for f, _ in explan_pairs]
    necessity = {}
    for feat in expl_feats:
        subset = [(f, v) for f, v in explan_pairs if int(f) != int(feat)]
        is_suff, ce = verify_sufficiency_z3_forest(rf_wrapper, instance, subset, target=target, timeout_ms=timeout_ms)
        # if subset is sufficient -> feat not necessary (is_necessary False)
        necessity[feat] = (not is_suff, ce)
    is_minimal = all(v[0] for v in necessity.values())
    return is_minimal, necessity


def verify_majoritary_minimality_z3(
    rf_wrapper,
    instance,
    majoritary_pairs,
    target=None,
    timeout_ms=3000
):
    """
    Checa minimalidade majoritária via Z3.
    Para cada feature f em M:
        testa se existe x que satisfaz M\\{f} E faz a maioria inverter.
    Retorna (is_minimal, necessity_dict) onde necessity_dict[feat] = (is_necessary, counterexample_or_None)
    (is_necessary == True significa: a feature é necessária para impedir a inversão).
    """
    if target is None:
        rf = getattr(rf_wrapper, "forest", None)
        if rf is None:
            raise ValueError("target required if rf_wrapper has no underlying sklearn forest")
        try:
            target = rf.predict(pd.DataFrame([instance]))[0]
        except Exception:
            target = rf.predict([instance])[0]

    necessity = {}
    M = [(int(f), v) for f, v in majoritary_pairs]

    for feat, val in M:
        # M sem feat
        M_reduced = [(f, v) for f, v in M if int(f) != int(feat)]

        # verify_majoritary_z3_forest retorna:
        #   (True, None)  -> M_reduced NÃO admite contra-exemplo (UNSAT)
        #   (False, ce)   -> existe contra-exemplo (SAT)
        is_suff, ce = verify_majoritary_z3_forest(
            rf_wrapper, instance, M_reduced, target=target, timeout_ms=timeout_ms
        )

        # agora: se is_suff == True -> M_reduced não permite flip => feat é necessária
        is_necessary = bool(is_suff)
        counterexample = ce  # pode ser None ou dict
        necessity[feat] = (is_necessary, counterexample)

    is_minimal = all(v[0] for v in necessity.values())
    return is_minimal, necessity




# ----------------------------
# Example usage (paste into main(), replace your previous verify call)
# ----------------------------
# after you computed:
# wrapped_forest = RandomForestWrapper(first_forest)
# suff_reason = wrapped_forest.find_sufficient_reason(instance, z3=True)  # prefer z3=True to get Z3-flavored string
#
# do:
#
# expl_pairs = parse_expl_string(suff_reason)
# is_suff, ce = verify_sufficiency_z3_forest(wrapped_forest, instance, expl_pairs, timeout_ms=3000)
# if is_suff:
#     print("Z3 prova: explicação é suficiente.")
# else:
#     print("Z3 encontrou contra-exemplo (features->value):")
#     print(ce)
#
# is_minimal, necessity = verify_minimality_z3(wrapped_forest, instance, expl_pairs, timeout_ms=3000)
# print("É minimal?", is_minimal)
# print("Necessidade por feature (feat_idx: (is_necessary, counterexample)):")
# print(necessity)
#
# Observação: ao chamar rf.predict em código acima, usamos pd.DataFrame([instance]) para evitar aviso
# "X does not have valid feature names".

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
    fichier = "bank"
    dataset = pd.read_csv(f"datasets/{fichier}.csv")
    print("Dataset loaded, shape:", dataset.shape)
    print()

    avg_score, forests = rf_cross_validation(dataset, 21, 10)
    print("avg_score: ", avg_score)
    print()

    for i in range(10):
        # escolha manual do índice da floresta
        idx = i   # <-- coloque aqui o índice desejado

        # pegar a floresta escolhida (tupla: (clf, train_idx, test_idx, acc))
        chosen_forest: RandomForestClassifier = forests[idx][0]
        chosen_train, chosen_test = forests[idx][1], forests[idx][2]

        # pegar a primeira árvore interna da floresta escolhida
        chosen_clf: DecisionTreeClassifier = chosen_forest.estimators_[0]
        tree = chosen_clf.tree_

        
        #print("Example tree: max_depth", tree.max_depth, " node_count", tree.node_count)
        #print()

        # preparar instância de teste (primeira da base)
        X, y = get_x_y(dataset)
        instance = X.iloc[0]
        #print("Instance (positional values):")
        #print(instance)
        #print()

        # wrapper
        wrapped_forest = RandomForestWrapper(chosen_forest)

        # obter target de forma robusta (usa safe_rf_predict)
        try:
            target = safe_rf_predict(chosen_forest, instance)[0]
        except Exception:
            # fallback direto
            target = chosen_forest.predict(pd.DataFrame([instance.values]))[0]
        #print("Model target for instance:", target)
        #print()

        # calcular sufficient reason (prefira z3=True se você tem o modo z3)
        #suff_reason = wrapped_forest.find_sufficient_reason(instance, target=target, z3=True)
        #print("Sufficient reason (string):", suff_reason)
        #print(suff_reason)
        #expl_pairs = parse_expl_string(suff_reason)
        #print("Parsed explanation pairs:", expl_pairs)
        #print()

        # calcular majoritary reason (prefira z3=True se você tem o modo z3)
        maj_reason = wrapped_forest.find_majoritary_reason(instance, target=target, z3=False)
        print("Majoritary reason (string):", maj_reason)
        print(maj_reason)
        maj_expl_pairs = parse_expl_string(maj_reason)
        print("Parsed explanation pairs:", maj_expl_pairs)
        print()

        """
        
        # ---------- 1) verificação empírica (perturbações aleatórias) ----------
        print("1) Empirical random perturbation check (n_trials=200)...")
        emp_ok, emp_ce = verify_sufficiency_random_perturbation(chosen_forest, wrapped_forest, instance, explan_pairs=expl_pairs, n_trials=200)
        if emp_ok:
            print("  -> Empirical: no counterexample found in random trials.")
        else:
            print("  -> Empirical: counterexample found (inst):")
            print(emp_ce)
            try:
                pred_ce = safe_rf_predict(chosen_forest, emp_ce)[0]
                print("     model prediction for counterexample:", pred_ce)
            except Exception:
                print("     unable to re-predict counterexample with safe_rf_predict.")
        print()

        # ---------- 2) verificação formal Z3 ----------
        print("2) Formal Z3 sufficiency check (timeout_ms=3000)...")
        try:
            is_suff, z3_ce = verify_sufficiency_z3_forest(wrapped_forest, instance, expl_pairs, target=target, timeout_ms=3000)
            if is_suff:
                print("  -> Z3: proved sufficient (no counterexample within bounds).")
            else:
                print("  -> Z3: counterexample found (dict feat_idx -> value):")
                print(z3_ce)
                # montar Series para testar predição (substitui None por valor original)
                n_feats = getattr(wrapped_forest.forest, "n_features_in_", len(instance))
                vals = []
                for i in range(n_feats):
                    v = z3_ce.get(i, None) if isinstance(z3_ce, dict) else None
                    if v is None:
                        vals.append(float(instance.iloc[i]))
                    else:
                        vals.append(float(v))
                ce_series = pd.Series(vals)
                # tentar predizer o contra-exemplo
                try:
                    pred_ce = safe_rf_predict(chosen_forest, ce_series)[0]
                    print("     model prediction for Z3 counterexample (constructed):", pred_ce)
                except Exception:
                    print("     could not predict constructed z3 counterexample.")
        except Exception as e:
            print("  -> Z3 check failed / timeout / exception:", repr(e))
            print("     treat as inconclusive or increase timeout_ms.")
        print()

        # ---------- 3) checagem de minimalidade (formal) ----------
        print("3) Minimality check (per-feature, Z3)...")
        try:
            is_minimal, necessity = verify_minimality_z3(wrapped_forest, instance, expl_pairs, target=target, timeout_ms=3000)
            print("  -> Is minimal?", is_minimal)
            print("  -> Necessity per feature:")
            for feat_idx, info in necessity.items():
                is_req, ce_feat = info
                print(f"     feat {feat_idx}: necessary={is_req}, counterexample={ce_feat}")
        except Exception as e:
            print("  -> Minimality check failed / timeout / exception:", repr(e))

        print("\nDone.")

        print("\n====================")
        print(" MAJORITY REASON CHECK")
        print("====================\n")

        # ---------- 1) Empirical check ----------
        print("1) Empirical random perturbation check (Majoritary)...")
        emp_ok, emp_ce = verify_sufficiency_random_perturbation(
            chosen_forest, wrapped_forest, instance,
            explan_pairs=maj_expl_pairs, n_trials=200
        )
        if emp_ok:
            print("  -> Empirical: no counterexample found for majoritary reason.")
        else:
            print("  -> Empirical: counterexample found for majoritary reason:")
            print(emp_ce)
        print()

        # ---------- 2) Formal Z3 check ----------
        print("2) Formal Z3 sufficiency check for Majoritary Reason...")
        is_suff, z3_ce = verify_majoritary_z3_forest(
            wrapped_forest, instance, maj_expl_pairs,
            target=target, timeout_ms=3000
        )
        if is_suff:
            print("  -> Z3: majoritary reason is sufficient (no CE).")
        else:
            print("  -> Z3: counterexample found for majoritary reason:")
            print(z3_ce)
        print()

        # ---------- 3) Minimality check ----------
        print("3) Minimality check for Majoritary Reason...")
        is_minimal, necessity = verify_majoritary_minimality_z3(
            wrapped_forest, instance, maj_expl_pairs,
            target=target, timeout_ms=3000
        )
        print("  -> Is minimal?", is_minimal)
        print("  -> Necessity per feature:")
        for feat_idx, info in necessity.items():
            print(f"     feat {feat_idx}: necessary={info[0]}, counterexample={info[1]}")
        print()
        """


if __name__ == "__main__":
    main()
