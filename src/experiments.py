import time
import pandas as pd
import numpy as np
from tqdm import tqdm
from src.wrappers import RandomForestWrapper


def calculate_size_from_string(expl_str):
    if not expl_str or expl_str.strip() == "[]":
        return 0
    content = expl_str.strip()[1:-1].strip()  # remove [ ]
    if not content:
        return 0
    return len([p for p in content.split(",") if p.strip()])


def benchmark_explanations(rf_clf, X_test, n_samples=25):
    wrapper = RandomForestWrapper(rf_clf)

    if len(X_test) > n_samples:
        samples = X_test.sample(n=n_samples, random_state=42)
    else:
        samples = X_test

    results = []
    print(f"--- Starting benchmark in {len(samples)} instances ---")

    for i, (idx, instance) in enumerate(tqdm(samples.iterrows(), total=len(samples))):

        # 1. Direct Reason
        start = time.time()
        direct_expl_arr, _, _ = wrapper.find_direct_reason(instance)
        end = time.time()

        results.append({
            'Instance_ID': idx,
            'Method': 'Direct',
            'Time': end - start,
            'Size': len(direct_expl_arr)
        })

        # 2. Majoritary Reason (SAT, fallback)
        start = time.time()
        try:
            maj_sat_str = wrapper.find_majoritary_reason(instance, z3=False)
            size_maj_sat = calculate_size_from_string(maj_sat_str)
        except Exception:
            size_maj_sat = 0
        end = time.time()

        results.append({
            'Instance_ID': idx,
            'Method': 'Majoritary (SAT)',
            'Time': end - start,
            'Size': size_maj_sat
        })

        # 3. Majoritary Reason (SMT)
        start = time.time()
        try:
            maj_z3_str = wrapper.find_majoritary_reason(instance, z3=True)
            size_maj_z3 = calculate_size_from_string(maj_z3_str)
        except Exception:
            try:
                # fallback for safety
                maj_z3_str = wrapper.find_majoritary_reason(instance, z3=False)
                size_maj_z3 = calculate_size_from_string(maj_z3_str)
            except Exception:
                size_maj_z3 = None
        end = time.time()

        if size_maj_z3 is not None:
            results.append({
                'Instance_ID': idx,
                'Method': 'Majoritary (SMT)',
                'Time': end - start,
                'Size': size_maj_z3
            })

        # 4. Sufficient Reason (SAT)
        start = time.time()
        try:
            suff_sat_str = wrapper.find_sufficient_reason(instance, z3=False)
            size_suff_sat = calculate_size_from_string(suff_sat_str)
        except Exception:
            size_suff_sat = 0
        end = time.time()

        results.append({
            'Instance_ID': idx,
            'Method': 'Sufficient (SAT)',
            'Time': end - start,
            'Size': size_suff_sat
        })

        # 5. Sufficient Reason (SMT)
        start = time.time()
        try:
            suff_z3_str = wrapper.find_sufficient_reason(instance, z3=True)
            size_suff_z3 = calculate_size_from_string(suff_z3_str)
        except Exception:
            try:
                # fallback
                suff_z3_str = wrapper.find_sufficient_reason(instance, z3=False)
                size_suff_z3 = calculate_size_from_string(suff_z3_str)
            except Exception:
                size_suff_z3 = None
        end = time.time()

        if size_suff_z3 is not None:
            results.append({
                'Instance_ID': idx,
                'Method': 'Sufficient (SMT)',
                'Time': end - start,
                'Size': size_suff_z3
            })

    return pd.DataFrame(results)
