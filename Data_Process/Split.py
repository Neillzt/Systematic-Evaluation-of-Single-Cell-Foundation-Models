#Ziteng#
import pandas as pd
import numpy as np
import scanpy as sc
import scipy.sparse
from sklearn.model_selection import train_test_split

def convert_and_split_adata(adata, output_prefix, seeds=[0,1,2,3,4], train_size=0.9):
    """
    Extract data from AnnData object, convert gene IDs to gene names, and split/save as CSV and h5ad files using multiple seeds.
    
    Parameters:
    adata (AnnData): AnnData object containing gene expression data
    output_prefix (str): Prefix for output files
    seeds (list): List of random seeds
    train_size (float): Proportion of training set
    """
    # Extract expression matrix
    expression_matrix = pd.DataFrame(
        adata.X.toarray() if scipy.sparse.issparse(adata.X) else adata.X,
        index=adata.obs_names,
        columns=adata.var_names
    )
    
    # Create mapping dictionary from ENSEMBL ID to gene name
    id_to_name = dict(zip(adata.var_names, adata.var['gene_name']))
    
    # Convert column names
    new_columns = [id_to_name.get(col, col) for col in expression_matrix.columns]
    expression_matrix.columns = new_columns
    
    # Generate splits and save for each seed
    for seed in seeds:
        print(f"\nProcessing seed {seed}:")
        
        # Split data indices
        train_idx, val_idx = train_test_split(
            np.arange(len(expression_matrix)),
            train_size=train_size,
            random_state=seed
        )
        
        # Split DataFrame
        train_data = expression_matrix.iloc[train_idx]
        val_data = expression_matrix.iloc[val_idx]
        
        # Save CSV files
        train_csv = f"{output_prefix}_train_seed_{seed}.csv"
        val_csv = f"{output_prefix}_val_seed_{seed}.csv"
        train_data.to_csv(train_csv)
        val_data.to_csv(val_csv)
        
        # Create and save new AnnData objects
        # Training set
        train_adata = sc.AnnData(
            X=train_data.values,
            obs=adata.obs.iloc[train_idx].copy(),
            var=adata.var.copy()
        )
        train_h5ad = f"{output_prefix}_train_seed_{seed}.h5ad"
        train_adata.write_h5ad(train_h5ad)
        
        # Validation set
        val_adata = sc.AnnData(
            X=val_data.values,
            obs=adata.obs.iloc[val_idx].copy(),
            var=adata.var.copy()
        )
        val_h5ad = f"{output_prefix}_val_seed_{seed}.h5ad"
        val_adata.write_h5ad(val_h5ad)
        
        # Print information
        print(f"Seed {seed} split results:")
        print(f"Train set: {len(train_data)} samples")
        print(f"- Saved as CSV: {train_csv}")
        print(f"- Saved as H5AD: {train_h5ad}")
        print(f"Validation set: {len(val_data)} samples")
        print(f"- Saved as CSV: {val_csv}")
        print(f"- Saved as H5AD: {val_h5ad}")
        print(f"Train/Val split ratio: {len(train_data)}/{len(val_data)} = {len(train_data)/len(val_data):.2f}")

def main():
    # Load data
    print("Loading data...")
    adata = sc.read_h5ad('batch_covid_subsampled_train.h5ad') # If you use Covid-19 dataset
    print(f"Loaded data with shape: {adata.shape}")
    
    # Set output prefix
    output_prefix = 'Cod'
    
    # Set random seed list
    seeds = [0, 1, 2, 3, 4]
    
    # Convert and split data
    print("\nStarting data conversion and splitting...")
    convert_and_split_adata(adata, output_prefix, seeds=seeds, train_size=0.9)
    
    # Verify outputs
    print("\nVerifying data...")
    # Check CSV files
    train_csv = pd.read_csv(f'{output_prefix}_train_seed_0.csv', index_col=0)
    val_csv = pd.read_csv(f'{output_prefix}_val_seed_0.csv', index_col=0)
    
    # Check H5AD files
    train_h5ad = sc.read_h5ad(f'{output_prefix}_train_seed_0.h5ad')
    val_h5ad = sc.read_h5ad(f'{output_prefix}_val_seed_0.h5ad')
    
    print("\nVerification results for seed 0:")
    print(f"CSV files - Train: {train_csv.shape}, Val: {val_csv.shape}")
    print(f"H5AD files - Train: {train_h5ad.shape}, Val: {val_h5ad.shape}")

if __name__ == "__main__":
    main()
