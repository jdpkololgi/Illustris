import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from Model import Model
import torch

def get_data(model_type):
    print(f"Loading data for {model_type}...")
    # Initialize Model to get data loader
    # We use 'random_forest' as a dummy to initialize, the data loading part is common or similar enough for this purpose
    # or we can just use the specific model type if it matters for data loading in Model.py
    # Looking at Model.py, pipeline() is called in __init__, which loads data.
    # It seems independent of model_type for the data loading part (pipeline call).
    
    model_wrapper = Model(model_type=model_type)
    
    # Access train_loader
    train_loader = model_wrapper.train_loader
    
    # Extract data from loader
    features_list = []
    labels_list = []
    for features, labels in train_loader:
        features_list.append(features.numpy())
        labels_list.append(labels.numpy())
    
    X_train = np.concatenate(features_list, axis=0)
    y_train = np.concatenate(labels_list, axis=0)
    
    print(f"Data loaded. Shape: {X_train.shape}")
    return X_train, y_train

def tune_random_forest():
    print("\n--- Tuning Random Forest ---")
    X_train, y_train = get_data('random_forest')
    
    param_dist = {
        'n_estimators': [100, 200, 300, 400, 500],
        'max_depth': [None, 10, 20, 30, 40, 50],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }
    
    rf = RandomForestClassifier(class_weight='balanced', random_state=42)
    
    random_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_dist,
        n_iter=20,
        cv=3,
        verbose=2,
        random_state=42,
        n_jobs=-1
    )
    
    random_search.fit(X_train, y_train)
    
    print(f"Best Parameters: {random_search.best_params_}")
    print(f"Best Score: {random_search.best_score_}")
    return random_search.best_params_

def tune_xgboost():
    print("\n--- Tuning XGBoost ---")
    X_train, y_train = get_data('xgboost')
    
    param_dist = {
        'n_estimators': [100, 200, 300, 400, 500],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 4, 5, 6, 8, 10],
        'min_child_weight': [1, 3, 5],
        'gamma': [0, 0.1, 0.2, 0.3, 0.4],
        'colsample_bytree': [0.3, 0.4, 0.5, 0.7]
    }
    
    # XGBoost handles class weights differently, often via scale_pos_weight for binary
    # For multi-class, we might need to pass sample_weights to fit, or use a specific objective.
    # The current Model_classes.py implementation passes sample_weights to fit.
    # RandomizedSearchCV doesn't easily support passing sample_weights to fit for each split unless we wrap it or pass it as fit_params (which assumes same split).
    # However, XGBClassifier has a class_weight parameter in recent versions or we can rely on the model learning it if the data is balanced or if we don't provide it.
    # The original code manually balances classes or passes weights.
    # Let's try to use sample weights if possible, or just run without explicit weights for tuning to find structural params.
    # But wait, Model_classes.py uses `sample_weight` in `fit`.
    # Let's compute sample weights for the whole training set and pass it to fit_params.
    # Note: RandomizedSearchCV splits data, so passing a fixed array of sample_weights for X_train might be tricky if indices don't align.
    # Actually, sklearn's cross_val_score/GridSearchCV/RandomizedSearchCV handles `fit_params` but it expects the parameters to be indexable if they are data-dependent (like sample_weight).
    # Let's compute sample weights.
    
    from sklearn.utils import compute_class_weight
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    sample_weights = class_weights[y_train.astype(int)]
    
    xgb_model = xgb.XGBClassifier(objective='multi:softprob', num_class=len(np.unique(y_train)), eval_metric='mlogloss', random_state=42)
    
    random_search = RandomizedSearchCV(
        estimator=xgb_model,
        param_distributions=param_dist,
        n_iter=20,
        cv=3,
        verbose=2,
        random_state=42,
        n_jobs=-1
    )
    
    # We pass sample_weight to fit. RandomizedSearchCV will slice it correctly if it's passed as a fit_param and is an array of length n_samples.
    random_search.fit(X_train, y_train, sample_weight=sample_weights)
    
    print(f"Best Parameters: {random_search.best_params_}")
    print(f"Best Score: {random_search.best_score_}")
    return random_search.best_params_

if __name__ == "__main__":
    tune_random_forest()
    tune_xgboost()
