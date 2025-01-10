# Ziteng #
import sys 
import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import scanpy as sc
import scipy.sparse
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm
import matplotlib.pyplot as plt
sys.path.append("../model/")
from load import *
import pandas as pd
import scanpy as sc
from sklearn.preprocessing import LabelEncoder
    
    
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
            # Original frozen logic remains but will only execute if frozenmore=True
            for _, p in self.token_emb.named_parameters():
                p.requires_grad = False
            for _, p in self.pos_emb.named_parameters():
                p.requires_grad = False
            print('self.pos_emb and self.token_emb frozen')
            
            for na, param in self.encoder.named_parameters():
                param.requires_grad = False
            for na, param in self.encoder.transformer_encoder[-2].named_parameters():
                print('self.encoder.transformer_encoder', na, 'have grad')
                param.requires_grad = True
        else:
            # All parameters will be trainable
            for _, p in self.token_emb.named_parameters():
                p.requires_grad = True
            for _, p in self.pos_emb.named_parameters():
                p.requires_grad = True
            for _, param in self.encoder.named_parameters():
                param.requires_grad = True
            print('All parameters are trainable')

        self.fc1 = nn.Sequential(
            nn.Linear(model_config['encoder']['hidden_dim'], 512),
            nn.ReLU(),
            nn.Linear(512, 21),

        )
        self.norm = torch.nn.BatchNorm1d(model_config['encoder']['hidden_dim'], affine=False, eps=1e-6)
        self.model_config = model_config

    def forward(self, sample_list, *args, **kwargs):
        label = sample_list['targets']

        x = sample_list['x'] # (B, L)
        value_labels = x > 0
        x, x_padding = gatherData(x, value_labels, self.model_config['pad_token_id'])
        data_gene_ids = torch.arange(19264, device=x.device).repeat(x.shape[0], 1)
        position_gene_ids, _ = gatherData(data_gene_ids, value_labels,
                                        self.model_config['pad_token_id'])
        
        x = self.token_emb(torch.unsqueeze(x, 2).float(), output_weight = 0)
        position_emb = self.pos_emb(position_gene_ids)
        x += position_emb

        logits = self.encoder(x,x_padding)

        # mlp
        logits, _ = torch.max(logits, dim=1)  # b,dim

        logits = self.norm(logits)
        logits = self.fc1(logits)

        return logits

def evaluate(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            outputs = model({'x': data, 'targets': target})
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    accuracy = 100 * correct / total
    f1 = f1_score(all_targets, all_preds, average='macro')
    precision = precision_score(all_targets, all_preds, average='macro')
    recall = recall_score(all_targets, all_preds, average='macro')
    
    return accuracy, precision, recall, f1


def train(model, train_loader, val_loader, test_loader, epochs=100, lr=0.00001, patience=10, log_file='./res/MS_mixed_0.txt'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val_accuracy = 0
    best_test_metrics = None
    best_val_metrics = None
    epochs_no_improve = 0
    
    with open(log_file, 'w') as f:
        f.write("Training started\n")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}")
        for batch_idx, (data, target) in progress_bar:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model({'x': data, 'targets': target})
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            progress_bar.set_postfix({'loss': total_loss / (batch_idx + 1), 'acc': 100. * correct / total})
        
        train_loss = total_loss / len(train_loader)
        train_accuracy = 100. * correct / total
        
        # Validation phase
        val_accuracy, val_precision, val_recall, val_f1 = evaluate(model, val_loader, device)
        
        # Test phase
        test_accuracy, test_precision, test_recall, test_f1 = evaluate(model, test_loader, device)
        
        # Logging
        log_message = f"Epoch {epoch+1}/{epochs}:\n"
        log_message += f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%\n"
        log_message += f"Val Accuracy: {val_accuracy:.2f}%, Val F1 Score: {val_f1:.4f}, Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}\n"
        log_message += f"Test Accuracy: {test_accuracy:.2f}%, Test F1 Score: {test_f1:.4f}, Test Precision: {test_precision:.4f}, Test Recall: {test_recall:.4f}\n"
        
        print(log_message)
        with open(log_file, 'a') as f:
            f.write(log_message)
        
        # Early stopping based on validation accuracy
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_val_metrics = (val_accuracy, val_precision, val_recall, val_f1)
            best_test_metrics = (test_accuracy, test_precision, test_recall, test_f1)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                early_stop_message = f"Early stopping triggered after {epoch+1} epochs\n"
                print(early_stop_message)
                with open(log_file, 'a') as f:
                    f.write(early_stop_message)
                break
    
    return best_val_metrics, best_test_metrics

def prepare_data(expression_csv, adata, label_encoder=None):
    # 
    X = pd.read_csv(expression_csv, index_col=0).values
    
    # 
    labels = adata.obs['cell_type']
    
    # 
    if label_encoder is None:
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(labels)
    else:
        # 
        y = label_encoder.transform(labels)
    
    # 
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y)
    
    # print(f"X_tensor shape: {X_tensor.shape}")
    # print(f"y_tensor shape: {y_tensor.shape}")
    
    
    return TensorDataset(X_tensor, y_tensor), label_encoder



def main():
   # Load test dataset
   test_adata = sc.read_h5ad('query_adata.h5ad')
   
   # Store all experiment results
   all_results = {
       'val': [],
       'test': []
   }
   
   # Process each random seed
   for seed in range(5):
       print(f"\nProcessing seed {seed}")
       
       # Load train and validation data for current seed
       train_adata = sc.read_h5ad(f'./data/MY/MY_train_seed_{seed}.h5ad')
       val_adata = sc.read_h5ad(f'./data/MY/MY_val_seed_{seed}.h5ad')
       
       # Combine all cell types for consistent encoding
       all_cell_types = pd.concat([
           train_adata.obs['cell_type'],
           val_adata.obs['cell_type'],
           test_adata.obs['cell_type']
       ])
       global_label_encoder = LabelEncoder()
       global_label_encoder.fit(all_cell_types)
       
       # Prepare data loaders
       test_dataset, _ = prepare_data('My_test.csv', test_adata, global_label_encoder)
       train_dataset, _ = prepare_data(f'./data/MY/MY_train_seed_{seed}.csv', train_adata, global_label_encoder)
       val_dataset, _ = prepare_data(f'./data/MY/MY_val_seed_{seed}.csv', val_adata, global_label_encoder)
       
       # Create data loaders with batch processing settings
       train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
       val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)
       test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)
       
       # Initialize and train model
       model = LinearProbingClassifier(ckpt_path='./models/models.ckpt')
       model.build()
       model = model.cuda()
       
       # Train and get best metrics
       val_metrics, test_metrics = train(
           model, train_loader, val_loader, test_loader,
           log_file=f'./res/MY_256F2_seed_{seed}.txt'
       )
       
       # Store results
       all_results['val'].append(val_metrics)
       all_results['test'].append(test_metrics)
       
       # Print current seed results
       print(f"\nResults for seed {seed}:")
       print(f"Validation - Accuracy: {val_metrics[0]:.2f}%, F1: {val_metrics[3]:.4f}, "
             f"Precision: {val_metrics[1]:.4f}, Recall: {val_metrics[2]:.4f}")
       print(f"Test - Accuracy: {test_metrics[0]:.2f}%, F1: {test_metrics[3]:.4f}, "
             f"Precision: {test_metrics[1]:.4f}, Recall: {test_metrics[2]:.4f}")
   
   # Calculate statistical metrics
   val_results = np.array(all_results['val'])
   test_results = np.array(all_results['test'])
   
   val_mean = np.mean(val_results, axis=0)
   val_std = np.std(val_results, axis=0)
   test_mean = np.mean(test_results, axis=0)
   test_std = np.std(test_results, axis=0)
   
   # Save final statistics to file
   with open('./res/MY_256F2_final_statistics.txt', 'w') as f:
       f.write("Individual experiment results:\n\n")
       for seed in range(5):
           f.write(f"Seed {seed}:\n")
           f.write(f"Validation - Accuracy: {val_results[seed][0]:.2f}%, F1: {val_results[seed][3]:.4f}, "
                  f"Precision: {val_results[seed][1]:.4f}, Recall: {val_results[seed][2]:.4f}\n")
           f.write(f"Test - Accuracy: {test_results[seed][0]:.2f}%, F1: {test_results[seed][3]:.4f}, "
                  f"Precision: {test_results[seed][1]:.4f}, Recall: {test_results[seed][2]:.4f}\n\n")
       
       f.write("\nFinal Statistics:\n")
       f.write("\nValidation Set (Mean ± Std):\n")
       f.write(f"Accuracy: {val_mean[0]:.2f}% (±{val_std[0]:.2f})\n")
       f.write(f"F1 Score: {val_mean[3]:.4f} (±{val_std[3]:.4f})\n")
       f.write(f"Precision: {val_mean[1]:.4f} (±{val_std[1]:.4f})\n")
       f.write(f"Recall: {val_mean[2]:.4f} (±{val_std[2]:.4f})\n")
       
       f.write("\nTest Set (Mean ± Std):\n")
       f.write(f"Accuracy: {test_mean[0]:.2f}% (±{test_std[0]:.2f})\n")
       f.write(f"F1 Score: {test_mean[3]:.4f} (±{test_std[3]:.4f})\n")
       f.write(f"Precision: {test_mean[1]:.4f} (±{test_std[1]:.4f})\n")
       f.write(f"Recall: {test_mean[2]:.4f} (±{test_std[2]:.4f})\n")

if __name__ == '__main__':
   main()
