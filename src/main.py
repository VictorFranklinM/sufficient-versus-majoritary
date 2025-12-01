import os
import copy
import shutil
from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

from src.experiments import benchmark_explanations
from src.plots import generate_plots

MAX_RANDOM_FOREST_DEPTH = 8


def rf_cross_validation(data, n_trees, cv, n_forests=None):
    if not n_forests: n_forests = cv
    kf = KFold(n_splits=cv, shuffle=True)  # Creates cross validation groups and shuffles them
    forests = []
    score = 0  # Avg score of all the trees

    for train_idx, test_idx in kf.split(data):
        train_set, test_set = data.iloc[train_idx], data.iloc[test_idx]
        X_train, y_train = train_set.iloc[:, :-1], train_set.iloc[:, -1]
        X_test, y_test = test_set.iloc[:, :-1], test_set.iloc[:, -1]
        rf = RandomForestClassifier(max_depth=MAX_RANDOM_FOREST_DEPTH, n_estimators=n_trees)
        rf.fit(X_train, y_train)
        y_predict = rf.predict(X_test)

        score += (accuracy_score(y_test, y_predict)) * 100
        forests.append((copy.deepcopy(rf), train_idx, test_idx))
    score /= n_forests # calc avg
    return score, forests


def main():
    fichiers = os.listdir('datasets')
    if not fichiers:
        raise FileNotFoundError("Datasets not found! Extract datasets.zip into /datasets/")
    database = {}
    tree_amount_info = {"ad-data": 53, "adult": 89, "bank": 21, "christine": 87, "compas": 21,
                        "gina": 55, "gisette": 85, "dorothea": 71, "farm-ads": 41, "mnist49": 81,
                        "mnist38": 85, "dexter": 101, "recidivism": 25, "higgs-boson": 95, "placement": 25}

    for f in fichiers:
        if f not in ["placement.csv", "bank.csv", "compas.csv", "compas-col.csv", "recidivism.csv"]:
            continue
        database[f.split('.')[0]] = pd.read_csv(f"datasets/{f}")


    for dataset_name in database.keys():
        if dataset_name not in tree_amount_info.keys(): continue
        print(f"Working on {dataset_name}")
        tree_amount = tree_amount_info[dataset_name]
        fold = 10
        ds = database[dataset_name].copy()

        # Training
        print(f"Training Random Forest ({tree_amount} trees)...")
        score, forests = rf_cross_validation(ds, tree_amount, fold)
        print(f"Accuracy Score: {score:.2f}%")

        # Benchmark
        first_forest_model = forests[0][0]
        test_indices = forests[0][2]
        test_data = ds.iloc[test_indices]
        X_test = test_data.iloc[:, :-1] # Remove a coluna target

        print(f"Explaining samples...")
        df_results = benchmark_explanations(first_forest_model, X_test, n_samples=25)
        
        # Saving results
        results_dir = os.path.join("plots", "results")
        os.makedirs(results_dir, exist_ok=True)
        csv_path = os.path.join(results_dir, f"{dataset_name}_results.csv")
        df_results.to_csv(csv_path, index=False)
        print(f"Data saved in: {csv_path}")

        # Plotting graphs
        print("Generating graphs...")
        generate_plots(df_results, dataset_name)
        print(f"Finished: {dataset_name}\n")

def clean_pycache():
    root_dir = Path(".")
    for cache_dir in root_dir.rglob("__pycache__"):
        if cache_dir.is_dir():
            shutil.rmtree(cache_dir)
            print(f"Cleaning: Cache removed from {cache_dir}")

if __name__ == '__main__':
    try:
        main()
    finally:
        print("\nCleaning temp archives...")
        clean_pycache()
