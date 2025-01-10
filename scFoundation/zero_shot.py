# Ziteng #
import sys 
import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import scanpy as sc
import scipy.sparse
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from tqdm import tqdm
import matplotlib.pyplot as plt
sys.path.append("../model/")
from load import *
import pandas as pd
import scanpy as sc
from sklearn.preprocessing import LabelEncoder
from scipy.spatial.distance import cdist
import umap

class LinearProbingClassifier(nn.Module):
    def __init__(self, ckpt_path, frozenmore=True):
        super().__init__()
        self.ckpt_path = ckpt_path
        self.frozenmore = frozenmore

    def build(self):
        model, model_config = load_model_frommmf(self.ckpt_path)
        self.token_emb = model.token_emb
        self.pos_emb = model.pos_emb
        self.encoder = model.encoder
        
        if self.frozenmore:
            for _, p in self.token_emb.named_parameters():
                p.requires_grad = False
            for _, p in self.pos_emb.named_parameters():
                p.requires_grad = False
            print('self.pos_emb and self.token_emb also frozen')
        
        for na, param in self.encoder.named_parameters():
            param.requires_grad = False
        for na, param in self.encoder.transformer_encoder[-2].named_parameters():
            print('self.encoder.transformer_encoder ', na, ' have grad')
            param.requires_grad = True

        self.norm = torch.nn.BatchNorm1d(model_config['encoder']['hidden_dim'], affine=False, eps=1e-6)
        self.model_config = model_config
        
    def forward(self, sample_list, *args, **kwargs):
        label = sample_list['targets']

        x = sample_list['x']  # (B, L)
        value_labels = x > 0
        x, x_padding = gatherData(x, value_labels, self.model_config['pad_token_id'])
        data_gene_ids = torch.arange(19264, device=x.device).repeat(x.shape[0], 1)
        position_gene_ids, _ = gatherData(data_gene_ids, value_labels,
                                        self.model_config['pad_token_id'])
        
        x = self.token_emb(torch.unsqueeze(x, 2).float(), output_weight=0)
        position_emb = self.pos_emb(position_gene_ids)
        x += position_emb

        logits = self.encoder(x, x_padding)

        embeddings, _ = torch.max(logits, dim=1)  # b,dim
        embeddings = self.norm(embeddings)

        return embeddings
      
def predict_cell_types(train_embeddings, train_labels, test_embeddings, k=5):
    """
    Predict cell types using KNN method based on cosine similarity
    k: number of nearest neighbors to consider for majority voting
    """
    # Calculate cosine similarity
    similarities = 1 - cdist(test_embeddings, train_embeddings, metric='cosine')
    
    # Find k most similar training samples for each test sample
    top_k_indices = np.argsort(similarities, axis=1)[:, -k:]
    
    # Get predictions based on majority voting of k nearest neighbors
    predictions = []
    for indices in top_k_indices:
        # Get labels of k nearest neighbors
        neighbor_labels = train_labels[indices]
        # Select most common label as prediction
        pred = np.bincount(neighbor_labels).argmax()
        predictions.append(pred)
    
    return np.array(predictions)

def evaluate_predictions(true_labels, predicted_labels, label_encoder):
    """
    Evaluate prediction results with multiple metrics
    """
    accuracy = accuracy_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels, average='macro')
    precision = precision_score(true_labels, predicted_labels, average='macro')
    recall = recall_score(true_labels, predicted_labels, average='macro')
    
    # Get specific class names
    class_names = label_encoder.classes_
    
    # Calculate F1 score for each class
    class_f1 = f1_score(true_labels, predicted_labels, average=None)
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'per_class_f1': dict(zip(class_names, class_f1))
    }

def prepare_data(expression_csv, adata, label_encoder=None):
    """
    Prepare data by loading expression matrix and labels
    """
    # Load expression matrix
    X = pd.read_csv(expression_csv, index_col=0).values
    
    # Get cell type labels
    labels = adata.obs['cell_type']
    
    # Encode labels
    if label_encoder is None:
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(labels)
    else:
        y = label_encoder.transform(labels)
    
    # Convert to tensors
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y)
    
    print(f"X_tensor shape: {X_tensor.shape}")
    print(f"y_tensor shape: {y_tensor.shape}")
    
    return TensorDataset(X_tensor, y_tensor), label_encoder

def save_embeddings(model, data_loader, save_path, device):
    """Save embeddings to file"""
    model.eval()
    all_embeddings = []
    all_labels = []
    
    with torch.no_grad():
        for data, target in tqdm(data_loader, desc="Extracting embeddings"):
            data, target = data.to(device), target.to(device)
            embeddings = model({'x': data, 'targets': target})
            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.append(target.cpu().numpy())
    
    embeddings = np.concatenate(all_embeddings, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    
    np.savez(save_path, embeddings=embeddings, labels=labels)
    
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Saved embeddings to {save_path}")
    
    return embeddings, labels

def main():
    # Load test dataset
    test_adata = sc.read_h5ad('./data/Lung/sample_proc_lung_test.h5ad')
    
    # Store experiment results
    embeddings_results = {}
    
    # Process each seed
    for seed in range(5):
        print(f"\nProcessing seed {seed}")
        
        # Load current seed's train and validation sets
        train_adata = sc.read_h5ad(f'./data/Lung/Lung_train_seed_{seed}.h5ad')
        val_adata = sc.read_h5ad(f'./data/Lung/Lung_val_seed_{seed}.h5ad')
        
        # Combine all cell types for encoding
        all_cell_types = pd.concat([
            train_adata.obs['cell_type'],
            val_adata.obs['cell_type'],
            test_adata.obs['cell_type']
        ])
        global_label_encoder = LabelEncoder()
        global_label_encoder.fit(all_cell_types)
        
        # Prepare data loaders
        test_dataset, _ = prepare_data('./data/Lung/Lung_test_new.csv', test_adata, global_label_encoder)
        train_dataset, _ = prepare_data(f'./data/Lung/Lung_train_seed_{seed}.csv', train_adata, global_label_encoder)
        val_dataset, _ = prepare_data(f'./data/Lung/Lung_val_seed_{seed}.csv', val_adata, global_label_encoder)
        
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)
        
        # Initialize model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = LinearProbingClassifier(ckpt_path='./models/models.ckpt')
        model.build()
        model = model.to(device)
        
        # Extract embeddings
        train_embeddings, train_labels = save_embeddings(model, train_loader, f'./embeddings/Lung_train_embeddings_seed_{seed}.npz', device)
        val_embeddings, val_labels = save_embeddings(model, val_loader, f'./embeddings/Lung_val_embeddings_seed_{seed}.npz', device)
        test_embeddings, test_labels = save_embeddings(model, test_loader, f'./embeddings/Lung_test_embeddings_seed_{seed}.npz', device)
        
        embeddings_results[seed] = {
            'train': (train_embeddings, train_labels),
            'val': (val_embeddings, val_labels),
            'test': (test_embeddings, test_labels)
        }
        
        # Try different k values
        k_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 25, 30, 40, 50]
        results = {}
        
        for k in k_values:
            print(f"\nTesting with k={k} for seed {seed}")
            predicted_labels = predict_cell_types(train_embeddings, train_labels, test_embeddings, k=k)
            metrics = evaluate_predictions(test_labels, predicted_labels, global_label_encoder)
            results[k] = metrics
            
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print(f"F1 Score: {metrics['f1']:.4f}")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall: {metrics['recall']:.4f}")
            
            print("\nPer-class F1 scores:")
            for cell_type, f1_score in metrics['per_class_f1'].items():
                print(f"{cell_type}: {f1_score:.4f}")

        # Save results for current seed
        with open(f'./data/Lung/embedding_prediction_results_seed_{seed}.txt', 'w') as f:
            f.write(f"Embedding-based Cell Type Prediction Results for Seed {seed}\n")
            f.write("="*50 + "\n\n")
            
            for k, metrics in results.items():
                f.write(f"\nResults for k={k}\n")
                f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
                f.write(f"F1 Score: {metrics['f1']:.4f}\n")
                f.write(f"Precision: {metrics['precision']:.4f}\n")
                f.write(f"Recall: {metrics['recall']:.4f}\n")
                
                f.write("\nPer-class F1 scores:\n")
                for cell_type, f1_score in metrics['per_class_f1'].items():
                    f.write(f"{cell_type}: {f1_score:.4f}\n")
                f.write("\n" + "-"*50 + "\n")


