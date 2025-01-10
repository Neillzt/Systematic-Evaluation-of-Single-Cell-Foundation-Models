# Ziteng #
import sys
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from xgboost import XGBClassifier
from tqdm import tqdm
import matplotlib.pyplot as plt
import joblib
import os

def prepare_data(adata, label_encoder=None):
    """
    Prepare data for training by extracting features and labels
    """
    X = adata.X
    labels = adata.obs['celltype']
    
    if label_encoder is None:
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(labels)
    else:
        y = label_encoder.transform(labels)
    
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    
    return X, y, label_encoder

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
    Train XGBoost model with given parameters
    """
    model = XGBClassifier(
        use_label_encoder=False,
        tree_method='gpu_hist',  # Use GPU acceleration, change to 'hist' if no GPU available
        **params
    )
    
    eval_set = [(X_train, y_train)]
    
    model.fit(
        X_train, y_train,
        eval_set=eval_set,
        eval_metric='mlogloss',
        verbose=False
    )
    
    results = model.evals_result()
    train_loss = results['validation_0']['mlogloss']
    
    with open(log_file, 'w') as f:
        for epoch in range(len(train_loss)):
            train_accuracy, train_precision, train_recall, train_f1 = evaluate(model, X_train, y_train)
            
            log_message = f"Epoch {epoch+1}/{len(train_loss)}:\n"
            log_message += f"Train Loss: {train_loss[epoch]:.4f}, Train Accuracy: {train_accuracy:.2f}%, Train F1 Score: {train_f1:.4f}\n"
            
            print(log_message)
            f.write(log_message + '\n')
    
    # Plot training metrics
    plt.figure(figsize=(12, 4))
    plt.plot(train_loss, label='Train Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'xgboost_training_metrics_{params["n_estimators"]}.png')
    plt.close()
    
    return model

def save_model(model, params, filename):
    """Save model and parameters to file"""
    model_data = {
        'model': model,
        'params': params
    }
    joblib.dump(model_data, filename)
    print(f"Model saved to {filename}")

def load_model(filename):
    """Load model and parameters from file"""
    model_data = joblib.load(filename)
    print(f"Model loaded from {filename}")
    return model_data['model'], model_data['params']

def parameter_search(X_train, y_train, X_test, y_test, param_grid):
    """
    Perform parameter search to find the best model configuration
    """
    best_score = 0
    best_params = None
    best_model = None
    
    for params in tqdm(param_grid):
        model = train(X_train, y_train, params, log_file=f'training_log_n_estimators_{params["n_estimators"]}.txt')
        train_accuracy, _, _, train_f1 = evaluate(model, X_train, y_train)
        score = (train_accuracy + train_f1 * 100) / 2  # Use average of accuracy and F1 score as evaluation metric
        
        test_accuracy, _, _, test_f1 = evaluate(model, X_test, y_test)
        
        print(f"Parameters: {params}")
        print(f"Train Score: {score:.2f}")
        print(f"Test Score: {(test_accuracy + test_f1 * 100) / 2:.2f}")
        print("--------------------")
        
        if score > best_score:
            best_score = score
            best_params = params
            best_model = model
    
    # Save best model
    save_model(best_model, best_params, 'best_xgboost_model_Ms.joblib')
    
    return best_model, best_params, best_score

def main():
    # Load data
    train_adata = sc.read_h5ad('train_val_0.h5ad')
    test_adata = sc.read_h5ad('test_0.h5ad')
    
    # Prepare global label encoder
    all_cell_types = pd.concat([train_adata.obs['celltype'], test_adata.obs['celltype']])
    global_label_encoder = LabelEncoder()
    global_label_encoder.fit(all_cell_types)
    
    # Prepare datasets
    X_train, y_train, _ = prepare_data(train_adata, global_label_encoder)
    X_test, y_test, _ = prepare_data(test_adata, global_label_encoder)
    
    print("Classes in training set:", global_label_encoder.classes_)
    print("Classes in test set:", global_label_encoder.transform(test_adata.obs['celltype'].unique()))
    
    # Define parameter grid for search
    param_grid = [
        {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 7,
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'objective': 'multi:softprob',
            'num_class': len(global_label_encoder.classes_)
        },
        {
            'n_estimators': 200,
            'learning_rate': 0.05,
            'max_depth': 9,
            'min_child_weight': 2,
            'subsample': 0.7,
            'colsample_bytree': 0.7,
            'objective': 'multi:softprob',
            'num_class': len(global_label_encoder.classes_)
        },
        {
            'n_estimators': 300,
            'learning_rate': 0.03,
            'max_depth': 8,
            'min_child_weight': 3,
            'subsample': 0.75,
            'colsample_bytree': 0.75,
            'objective': 'multi:softprob',
            'num_class': len(global_label_encoder.classes_)
        }
        # Add more parameter combinations as needed...
    ]
    
    # Check for existing model
    if os.path.exists('best_xgboost_model_Ms_nor.joblib'):
        print("Found saved model. Do you want to load it? (y/n)")
        choice = input().lower()
        if choice == 'y':
            best_model, best_params = load_model('best_xgboost_model_ms.joblib')
        else:
            best_model, best_params, best_score = parameter_search(X_train, y_train, X_test, y_test, param_grid)
    else:
        best_model, best_params, best_score = parameter_search(X_train, y_train, X_test, y_test, param_grid)
    
    print(f"Best parameters: {best_params}")
    
    # Evaluate final model performance
    train_accuracy, train_precision, train_recall, train_f1 = evaluate(best_model, X_train, y_train)
    print(f"Training - Accuracy: {train_accuracy:.2f}%, F1 Score: {train_f1:.4f}, "
          f"Precision: {train_precision:.4f}, Recall: {train_recall:.4f}")
    
    test_accuracy, test_precision, test_recall, test_f1 = evaluate(best_model, X_test, y_test)
    print(f"Final Model - Test Accuracy: {test_accuracy:.2f}%, Test F1 Score: {test_f1:.4f}, "
          f"Test Precision: {test_precision:.4f}, Test Recall: {test_recall:.4f}")

if __name__ == '__main__':
    main()
