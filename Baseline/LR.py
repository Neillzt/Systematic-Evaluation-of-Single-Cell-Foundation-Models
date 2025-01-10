
import sys
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegressionCV
from tqdm import tqdm
import matplotlib.pyplot as plt
import joblib
import os

def prepare_data(adata, label_encoder=None, scaler=None):
    """
    Prepare training data:
    1. Extract features and labels
    2. Standardize features
    3. Encode labels
    """
    X = adata.X
    labels = adata.obs['Celltype']
    
    # Feature standardization
    if scaler is None:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    else:
        X = scaler.transform(X)
    
    # Label encoding
    if label_encoder is None:
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(labels)
    else:
        y = label_encoder.transform(labels)
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Label vector shape: {y.shape}")
    
    return X, y, label_encoder, scaler

def evaluate(model, X, y):
    """
    Evaluate model performance using multiple metrics
    """
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred, average='macro')
    precision = precision_score(y, y_pred, average='macro')
    recall = recall_score(y, y_pred, average='macro')
    
    return accuracy * 100, precision, recall, f1

def train(X_train, y_train, params, log_file='training_log.txt'):
    """
    Train logistic regression model and record training process
    """
    model = LogisticRegressionCV(
        cv=5,  # 5-fold cross validation
        max_iter=1000,  # Maximum iterations
        n_jobs=-1,  # Use all CPU cores
        verbose=0,
        **params
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate training results
    train_accuracy, train_precision, train_recall, train_f1 = evaluate(model, X_train, y_train)
    log_message = f"Training Results:\n"
    log_message += f"Training Accuracy: {train_accuracy:.2f}%, F1 Score: {train_f1:.4f}\n"
    
    print(log_message)
    with open(log_file, 'w') as f:
        f.write(log_message)
    
    return model

def save_model(model, params, filename):
    """Save model and parameters to file"""
    model_data = {
        'model': model,
        'params': params
    }
    joblib.dump(model_data, filename)
    print(f"Model saved to: {filename}")

def load_model(filename):
    """Load model and parameters from file"""
    model_data = joblib.load(filename)
    print(f"Model loaded from: {filename}")
    return model_data['model'], model_data['params']

def parameter_search(X_train, y_train, X_test, y_test, param_grid):
    """
    Parameter search function to find best model configuration
    """
    best_score = 0
    best_params = None
    best_model = None
    
    for params in tqdm(param_grid):
        model = train(X_train, y_train, params, log_file=f'training_log_LR_{params["multi_class"]}.txt')
        train_accuracy, _, _, train_f1 = evaluate(model, X_train, y_train)
        score = (train_accuracy + train_f1 * 100) / 2  # Use average of accuracy and F1 score as metric
        
        test_accuracy, _, _, test_f1 = evaluate(model, X_test, y_test)
        
        print(f"Parameters: {params}")
        print(f"Training Score: {score:.2f}")
        print(f"Test Score: {(test_accuracy + test_f1 * 100) / 2:.2f}")
        print("--------------------")
        
        if score > best_score:
            best_score = score
            best_params = params
            best_model = model
    
    # Save best model
    save_model(best_model, best_params, 'best_lr_model.joblib')
    
    return best_model, best_params, best_score

def main():
    # Load data
    train_adata = sc.read_h5ad('demo_train_2.h5ad')
    test_adata = sc.read_h5ad('demo_test_new.h5ad')
    
    # Prepare global label encoder
    all_cell_types = pd.concat([train_adata.obs['Celltype'], test_adata.obs['Celltype']])
    global_label_encoder = LabelEncoder()
    global_label_encoder.fit(all_cell_types)
    
    # Prepare training and test sets
    X_train, y_train, _, scaler = prepare_data(train_adata, global_label_encoder)
    X_test, y_test, _, _ = prepare_data(test_adata, global_label_encoder, scaler)
    
    print("Classes in training set:", global_label_encoder.classes_)
    print("Classes in test set:", global_label_encoder.transform(test_adata.obs['Celltype'].unique()))
    
    # Define parameter search grid
    param_grid = [
        {
            'multi_class': 'ovr',  # one-vs-rest strategy
            'solver': 'lbfgs',
        },
        {
            'multi_class': 'multinomial',  # softmax strategy
            'solver': 'lbfgs',
        }
    ]
    
    # Check for existing saved model
    if os.path.exists('best_lr_model.joblib'):
        print("Found saved model. Do you want to load it? (y/n)")
        choice = input().lower()
        if choice == 'y':
            best_model, best_params = load_model('best_lr_model.joblib')
        else:
            best_model, best_params, best_score = parameter_search(X_train, y_train, X_test, y_test, param_grid)
    else:
        best_model, best_params, best_score = parameter_search(X_train, y_train, X_test, y_test, param_grid)
    
    print(f"Best parameters: {best_params}")
    
    # Evaluate final model performance
    train_accuracy, train_precision, train_recall, train_f1 = evaluate(best_model, X_train, y_train)
    print(f"Training Results - Accuracy: {train_accuracy:.2f}%, F1 Score: {train_f1:.4f}, "
          f"Precision: {train_precision:.4f}, Recall: {train_recall:.4f}")
    
    test_accuracy, test_precision, test_recall, test_f1 = evaluate(best_model, X_test, y_test)
    print(f"Test Results - Accuracy: {test_accuracy:.2f}%, F1 Score: {test_f1:.4f}, "
          f"Precision: {test_precision:.4f}, Recall: {test_recall:.4f}")

if __name__ == '__main__':
    main()
