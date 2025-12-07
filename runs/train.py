# %%

import mlflow

mlflow.set_tracking_uri('http://localhost:5000/')
mlflow.is_tracking_uri_set()

# %%

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# %%

with mlflow.start_run(1):
    mlflow.sklearn.autolog()

    X, y = load_breast_cancer(return_X_y=True)

    cv = 

