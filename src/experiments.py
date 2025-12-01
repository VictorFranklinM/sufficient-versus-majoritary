import time
import pandas as pd
import numpy as np
from tqdm import tqdm
from src.wrappers import RandomForestWrapper

def calculate_size_from_string(expl_str):
    if not expl_str or expl_str == "[]":
        return 0
    content = expl_str.strip("[]")
    if not content:
        return 0
    return len(content.split(","))

def benchmark_explanations(rf_clf, X_test, n_samples=25):
    wrapper = RandomForestWrapper(rf_clf)
    
    if len(X_test) > n_samples:
        samples = X_test.sample(n=n_samples, random_state=42)
    else:
        samples = X_test

    results = []
    print(f"--- Starting benchmark in {len(samples)} instances ---")

    for i, (idx, instance) in enumerate(tqdm(samples.iterrows(), total=len(samples))):
        
        # 1. Direct Reason (Baseline)
        start = time.time()
        direct_expl_arr, _, _ = wrapper.find_direct_reason(instance) 
        end = time.time()
        results.append({
            'Instance_ID': idx,
            'Method': 'Direct', 
            'Time': end - start,
            'Size': len(direct_expl_arr)
        })

        # 2. Majoritary Reason (Greedy)
        start = time.time()
        try:
            maj_str = wrapper.find_majoritary_reason(instance, z3=False)
            size_maj = calculate_size_from_string(maj_str)
        except Exception:
            size_maj = 0
        end = time.time()
        results.append({
            'Instance_ID': idx,
            'Method': 'Majoritary (Greedy)',
            'Time': end - start,
            'Size': size_maj
        })

        # 3. Majoritary (Z3)
        start = time.time()
        try:
            maj_z3_str = wrapper.find_majoritary_reason(instance, z3=True)
            size_maj_z3 = calculate_size_from_string(maj_z3_str)
        except Exception:
            size_maj_z3 = None
        end = time.time()
        
        if size_maj_z3 is not None:
            results.append({
                'Instance_ID': idx,
                'Method': 'Majoritary (Z3)',
                'Time': end - start,
                'Size': size_maj_z3
            })

        # 4. Sufficient Reason (Z3 - Formal)
        start = time.time()
        try:
            suff_str = wrapper.find_sufficient_reason(instance, z3=True)
            size_suff = calculate_size_from_string(suff_str)
        except Exception:
            size_suff = None
        end = time.time()
        
        if size_suff is not None:
            results.append({
                'Instance_ID': idx,
                'Method': 'Sufficient (Z3)',
                'Time': end - start,
                'Size': size_suff
            })

        # 5. Sufficient Reason (Greedy - Fallback)
        start = time.time()
        try:
            # z3=False força o uso do algoritmo guloso com CNF
            suff_greedy_str = wrapper.find_sufficient_reason(instance, z3=False)
            size_suff_greedy = calculate_size_from_string(suff_greedy_str)
        except Exception:
            size_suff_greedy = 0
        end = time.time()
        
        results.append({
            'Instance_ID': idx,
            'Method': 'Sufficient (Greedy)',
            'Time': end - start,
            'Size': size_suff_greedy
        })

    return pd.DataFrame(results)
