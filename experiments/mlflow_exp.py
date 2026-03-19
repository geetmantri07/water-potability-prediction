import pandas as pd
import numpy as np
import random
import mlflow
import dagshub
import mlflow.sklearn

from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

dagshub.init(
    repo_owner="geetmantri07",
    repo_name="water-potability-prediction",
    mlflow=True
)


#########################################
# DATA PREPROCESSING
#########################################

def preprocess_data(df):

    X = df.drop(columns=["Potability"])
    y = df["Potability"]
    

    imputer = KNNImputer(n_neighbors=5,weights="uniform")
    X = imputer.fit_transform(X)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    return X_train, X_test, y_train, y_test


#########################################
# MODEL EVALUATION
#########################################

def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    return accuracy, precision, recall, f1


#########################################
# EXPERIMENT RUNNER
#########################################



def run_experiment(model_class, param_grid, experiment_name,
                   X_train, X_test, y_train, y_test, max_runs=None):

    mlflow.set_experiment(experiment_name)

    grid = list(ParameterGrid(param_grid))

    # Limit number of runs if specified
    if max_runs and len(grid) > max_runs:
        grid = random.sample(grid, max_runs)

    print(f"Running {len(grid)} runs for {experiment_name}")

    for params in grid:

        with mlflow.start_run():

            model = model_class(**params)

            model.fit(X_train, y_train)

            accuracy, precision, recall, f1 = evaluate_model(
                model, X_test, y_test
            )

            mlflow.log_params(params)

            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)

            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model"
            )
#########################################
# MAIN FUNCTION
#########################################

def main(df):

    X_train, X_test, y_train, y_test = preprocess_data(df)

    #########################################
    # LOGISTIC REGRESSION
    #########################################

    logistic_param_grid = {
        "C": [0.001, 0.01, 0.1, 1, 10, 100],
        "solver": ["lbfgs", "liblinear"],
        "penalty": ["l2"],
        "max_iter": [100, 200, 500, 1000]
    }

    run_experiment(
        LogisticRegression,
        logistic_param_grid,
        "Logistic_Regression_v2",
        X_train,
        X_test,
        y_train,
        y_test
    )

    #########################################
    # DECISION TREE
    #########################################

    decision_tree_param_grid = {
        "max_depth": [3, 5, 7, 10, 15, 20, None],
        "min_samples_split": [2, 5, 10, 20],
        "min_samples_leaf": [1, 2, 4, 8],
        "criterion": ["gini", "entropy"]
    }

    run_experiment(
        DecisionTreeClassifier,
        decision_tree_param_grid,
        "Decision_Tree",
        X_train,
        X_test,
        y_train,
        y_test,
        max_runs=80
    )

    #########################################
    # RANDOM FOREST
    #########################################

    random_forest_param_grid = {
        "n_estimators": [50, 100, 200, 300],
        "max_depth": [5, 10, 20, 30, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"]
    }

    run_experiment(
        RandomForestClassifier,
        random_forest_param_grid,
        "Random_Forest",
        X_train,
        X_test,
        y_train,
        y_test,
        max_runs=80
    )


#########################################
# EXECUTION
#########################################

if __name__ == "__main__":

    # Load dataset (replace with your path if needed)
    df = pd.read_csv("experiments/water_potability.csv")

    main(df)