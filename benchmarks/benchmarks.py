import pandas as pd
import numpy as np

import lightgbm as lgb
from polynomial_boosting import PolynomialBoostingModel

import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.datasets import fetch_california_housing, load_diabetes
from sklearn.model_selection import KFold 

from ucimlrepo import fetch_ucirepo

N_TREES = 100
LEARNING_RATE = 0.25
LINEAR_LAMBDA = 5.0
MIN_SAMPLES_LEAF = 10
GOSS_ALPHA = 0.2
GOSS_BETA = 0.025
SEED = 123

#-------------------Datasets-------------------#

def get_california_housing():
    data = fetch_california_housing()
    X = data.data
    y = data.target
    return X, y

def get_diabetes():
    data = load_diabetes()
    X = data.data
    y = data.target
    return X, y

def get_wine_quality():
    data = fetch_ucirepo("Wine Quality")
    X = data.data.features.values
    y = data.data.targets.values.reshape(-1)
    return X, y

#-------------------Models-------------------#

def fit_predict_polyboost_1(X_train, y_train, X_test):
    model = PolynomialBoostingModel(1, LEARNING_RATE, LINEAR_LAMBDA, N_TREES, MIN_SAMPLES_LEAF, GOSS_ALPHA, GOSS_BETA, SEED)
    model.fit_fast(X_train, y_train.reshape(-1, 1))
    y_pred = model.predict_fast(X_test).reshape(-1)
    return y_pred

def fit_predict_polyboost_2(X_train, y_train, X_test):
    model = PolynomialBoostingModel(2, LEARNING_RATE, LINEAR_LAMBDA, N_TREES, MIN_SAMPLES_LEAF, GOSS_ALPHA, GOSS_BETA, SEED)
    model.fit_fast(X_train, y_train.reshape(-1, 1))
    y_pred = model.predict_fast(X_test).reshape(-1)
    return y_pred

def fit_predict_lgb(X_train, y_train, X_test):
    lgd_dataset_train = lgb.Dataset(X_train, label=y_train, params={'linear_tree': True})
    lgb_params = lgb_params = {
                "objective": "regression",
                "metric": "l2",
                "num_iterations": N_TREES,
                "num_leaves": 2,
                "learning_rate": LEARNING_RATE,
                "linear_lambda": LINEAR_LAMBDA,
                "min_data_in_leaf": MIN_SAMPLES_LEAF,
                "data_sampling_strategy": "goss",
                "top_rate": GOSS_ALPHA,
                "other_rate": GOSS_BETA,
                "deterministic": True,
                "seed": SEED,
                "verbosity": -1
            }

    model = lgb.train(lgb_params, lgd_dataset_train)
    y_pred = model.predict(X_test).reshape(-1)

    return y_pred

def fit_predict_sklearn(X_train, y_train, X_test):
    model = GradientBoostingRegressor(max_depth=1, n_estimators=N_TREES, learning_rate=LEARNING_RATE, min_samples_leaf=MIN_SAMPLES_LEAF, random_state=SEED)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test).reshape(-1)

    return y_pred

#-------------------Benchmarking-------------------#

DATASETS = {"California Housing": get_california_housing, "Diabetes": get_diabetes, "Wine Quality": get_wine_quality}
MODELS = {"PolyBoost (linear, p=1)": fit_predict_polyboost_1, "PolyBoost (quadratic, p=2)": fit_predict_polyboost_2, "LightGBM": fit_predict_lgb, "Sklearn": fit_predict_sklearn}

def full_benchmark():
    scores_df = pd.DataFrame(columns=MODELS.keys(), index=DATASETS.keys())

    for dataset_name in DATASETS.keys():
        print("Starting benchmark for dataset: {}".format(dataset_name))
        for model_name in MODELS.keys():
            print("Starting benchmark for model: {}".format(model_name))
            scores = run_benchmark(dataset_name, model_name)
        
            mean_score = np.mean(scores)
            std_score = np.std(scores)

            mean_std = "{:.4f} +/- {:.4f}".format(mean_score, std_score)

            scores_df.loc[dataset_name, model_name] = mean_std
            print("Finished benchmark for model: {}".format(model_name))
            print("Mean score: {}".format(mean_score))
            print("Std score: {}".format(std_score))

            print("\n")
        print("Finished benchmark for dataset: {}".format(dataset_name))
        print("------------------\n\n\n")


    #save mean_scores_df as table image
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.axis('off')
    ax.axis('tight')
    ax.table(cellText=scores_df.values, colLabels=scores_df.columns, rowLabels=scores_df.index, loc='center')
    fig.tight_layout()
    fig.savefig("results.png")

def run_benchmark(dataset_name, model_name):
    X, y = DATASETS[dataset_name]()
    model = MODELS[model_name]
    
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

    scores = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index,:], X[test_index,:]
        y_train, y_test = np.array(y[train_index]), np.array(y[test_index])

        #normalize data (z-score)
        X_mean = np.mean(X_train, axis=0)
        X_std = np.std(X_train, axis=0)
        X_train = (X_train - X_mean) / X_std
        X_test = (X_test - X_mean) / X_std

        y_mean = np.mean(y_train)
        y_std = np.std(y_train)
        y_train = (y_train - y_mean) / y_std
        y_test = (y_test - y_mean) / y_std

        y_pred = model(X_train, y_train, X_test)

        scores.append(np.mean((y_pred - y_test)**2))

    return scores

if __name__ == "__main__":
    full_benchmark()
