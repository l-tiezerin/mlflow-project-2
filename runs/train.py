# %%

import mlflow

mlflow.set_tracking_uri('http://localhost:5000/')
mlflow.is_tracking_uri_set()

# %%

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

# %%

with mlflow.start_run(experiment_id=1):
    mlflow.sklearn.autolog()

    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=.2, random_state=42
    )

    params = {
        'criterion': 'gini',
        'max_depth': None,
        'min_samples_split': 2,
        'class_weight': None,
    }
    model = DecisionTreeClassifier(**params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = {
        'test_accuracy': accuracy_score(y_test, y_pred),
        'test_precision': precision_score(y_test, y_pred),
        'test_recall': recall_score(y_test, y_pred)
    }

    mlflow.log_metrics(metrics)

# %%


