import time
import pandas as pd
import numpy as np
from tqdm import tqdm
from src.wrappers import RandomForestWrapper

def calculate_size_from_string(expl_str):

    if not expl_str or expl_str == "[]":
        return 0

    content = expl_str.strip("[]")
    return len(content.split(","))

def benchmark_explanations(rf_clf, X_test, n_samples=25):
    
    wrapper = RandomForestWrapper(rf_clf)
    
    # Amostragem, mudar depois
    if len(X_test) > n_samples:
        samples = X_test.sample(n=n_samples, random_state=42)
    else:
        samples = X_test

    results = []
    print(f"--- Beginning benchmark in {len(samples)} instances ---")

    for i, (idx, instance) in enumerate(tqdm(samples.iterrows(), total=len(samples))):
        
        print(f"\nProcessing instance {i+1}/{len(samples)} (ID: {idx})...")

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

        # # 2. Majoritary Reason
        # start = time.time()
        # maj_str = wrapper.find_majoritary_reason(instance)
        # size_maj = calculate_size_from_string(maj_str)
        # end = time.time()
        
        # results.append({
        #     'Instance_ID': idx,
        #     'Method': 'Majoritary',
        #     'Time': end - start,
        #     'Size': size_maj
        # })

        # 3. Sufficient Reason (Z3)
        start = time.time()
        suff_str = wrapper.find_sufficient_reason(instance, z3=True)
        size_suff = calculate_size_from_string(suff_str)
        end = time.time()
        
        results.append({
            'Instance_ID': idx,
            'Method': 'Sufficient (Z3)',
            'Time': end - start,
            'Size': size_suff
        })

    return pd.DataFrame(results)