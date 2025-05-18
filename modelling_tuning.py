import pandas as pd
import numpy as np
import os
import mlflow
import optuna
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import f1_score, classification_report, accuracy_score, confusion_matrix
import lightgbm as lgb
import dagshub
dagshub.init(repo_owner='mpfordreamer', repo_name='Cryptonews-analysis', mlflow=True)

# MLflow setup
EXPERIMENT_NAME = "crypto_sentiment_lgbm"
mlflow.set_experiment(EXPERIMENT_NAME)

def load_and_preprocess_data():
    """Load and preprocess the dataset"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, 'cryptonews_preprocessing', 'preprocessed_cryptonews.csv')
    
    df = pd.read_csv(file_path)
    X_text = df['text_clean']
    y = df['sentiment_encoded']
    
    vectorizer = TfidfVectorizer(max_features=2000)
    X = vectorizer.fit_transform(X_text)
    
    return train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

def objective(trial):
    """Optuna objective function with MLflow tracking"""
    # Define parameters for this trial
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'num_leaves': trial.suggest_int('num_leaves', 10, 150),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'random_state': 42,
        'verbose': -1
    }
    
    with mlflow.start_run(nested=True):
        # Log parameters
        mlflow.log_params(params)
        
        # Train and evaluate model
        model = lgb.LGBMClassifier(**params)
        scores = cross_val_score(model, X_train, y_train, cv=3, scoring='f1_weighted', n_jobs=-1)
        mean_f1 = np.mean(scores)
        
        # Log metrics
        mlflow.log_metric("mean_cv_f1", mean_f1)
        
        return mean_f1

def train_best_model(study, X_train, X_test, y_train, y_test):
    """Train and evaluate the best model"""
    with mlflow.start_run(run_name="best_model"):
        # Log best parameters
        mlflow.log_params(study.best_params)
        
        # Train model with best parameters
        best_model = lgb.LGBMClassifier(**study.best_params, random_state=42)
        best_model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = best_model.predict(X_test)
        test_f1 = f1_score(y_test, y_pred, average='weighted')
        accuracy = accuracy_score(y_test, y_pred)
        
        # Log metrics
        mlflow.log_metric("test_f1", test_f1)
        mlflow.log_metric("accuracy", accuracy)
        
        # Log model parameters
        mlflow.log_param("model_type", "LGBMClassifier")
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("n_samples", X_train.shape[0])
        
        # Log model
        mlflow.lightgbm.log_model(best_model, "model")
        
        # Save locally
        best_model.booster_.save_model('best_lgbm_model.txt')
        
        return best_model, test_f1, accuracy

if __name__ == "__main__":
    # Load and prepare data
    print("[INFO] Loading and preparing data...")
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    
    # Optimization
    print("[INFO] Starting Optuna optimization...")
    study = optuna.create_study(direction='maximize')
    
    # Run optimization with MLflow tracking
    with mlflow.start_run(run_name="optimization") as run:
        study.optimize(objective, n_trials=50)
        
        # Log best trial info and parameters
        mlflow.log_params({
            "best_trial_number": study.best_trial.number,
            "best_value": study.best_value,
            "n_trials": 50,
            "optimization_direction": "maximize"
        })
    
    # Train and evaluate best model
    print("\n[INFO] Training best model...")
    best_model, test_f1, accuracy = train_best_model(study, X_train, X_test, y_train, y_test)
    
    # Print results
    print("\n[RESULTS] Best Parameters:")
    print(study.best_params)
    print(f"\n[RESULTS] Best CV F1 Score: {study.best_value:.4f}")
    print(f"[Test] F1 Score: {test_f1:.4f}")
    print(f"[Test] Accuracy: {accuracy:.4f}")
    print("\n[INFO] Best model saved as 'best_lgbm_model.txt'")