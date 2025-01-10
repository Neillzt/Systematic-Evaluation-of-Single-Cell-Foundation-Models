# Ziteng #
import pandas as pd
import numpy as np
import scanpy as sc
import scipy.sparse
from sklearn.model_selection import train_test_split

def convert_and_split_adata(adata, output_prefix, seeds=[0,1,2,3,4], train_size=0.9):
    """
    Extract data from AnnData object, convert gene IDs to gene names, and split into train/val sets.
    
    Parameters:
    adata (AnnData): AnnData object containing gene expression data
    output_prefix (str): Prefix for output files
    seeds (list): List of random seeds for reproducible splits
    train_size (float): Proportion of training set
    """
    # Extract expression matrix
    expression_matrix = pd.DataFrame(
        adata.X.toarray() if scipy.sparse.issparse(adata.X) else adata.X,
        index=adata.obs_names,
        columns=adata.var_names
    )
    
    # Create mapping from ENSEMBL ID to gene name
    id_to_name = dict(zip(adata.var_names, adata.var['gene_name']))
    
    # Convert column names to gene names
    new_columns = [id_to_name.get(col, col) for col in expression_matrix.columns]
    expression_matrix.columns = new_columns
    
    # Process each seed
    for seed in seeds:
        print(f"\nProcessing seed {seed}:")
        
        # Split indices for train/val
        train_idx, val_idx = train_test_split(
            np.arange(len(expression_matrix)),
            train_size=train_size,
            random_state=seed
        )
        
        # Split data into train/val sets
        train_data = expression_matrix.iloc[train_idx]
        val_data = expression_matrix.iloc[val_idx]
        
        # Save initial CSV files
        train_csv = f"{output_prefix}_train_seed_{seed}.csv"
        val_csv = f"{output_prefix}_val_seed_{seed}.csv"
        train_data.to_csv(train_csv)
        val_data.to_csv(val_csv)
        
        # Create and save train AnnData
        train_adata = sc.AnnData(
            X=train_data.values,
            obs=adata.obs.iloc[train_idx].copy(),
            var=adata.var.copy()
        )
        train_h5ad = f"{output_prefix}_train_seed_{seed}.h5ad"
        train_adata.write_h5ad(train_h5ad)
        
        # Create and save validation AnnData
        val_adata = sc.AnnData(
            X=val_data.values,
            obs=adata.obs.iloc[val_idx].copy(),
            var=adata.var.copy()
        )
        val_h5ad = f"{output_prefix}_val_seed_{seed}.h5ad"
        val_adata.write_h5ad(val_h5ad)
        
        print(f"Seed {seed} split results:")
        print(f"Train set: {len(train_data)} samples")
        print(f"Validation set: {len(val_data)} samples")
        print(f"Train/Val ratio: {len(train_data)}/{len(val_data)} = {len(train_data)/len(val_data):.2f}")

def reorder_genes_by_tsv(csv_file, tsv_file, output_file):
    """
    Reorder genes in CSV file according to the gene order in TSV file.
    
    Parameters:
    csv_file (str): Input CSV file with gene expression data
    tsv_file (str): TSV file containing desired gene order
    output_file (str): Path to save reordered data
    
    Returns:
    tuple: Statistics about gene numbers (original, new, added, removed)
    """
    # Read input files
    expression_data = pd.read_csv(csv_file, index_col=0)
    gene_order = pd.read_csv(tsv_file, sep='\t', header=None, names=['gene_name', 'index'])
    gene_order = gene_order['gene_name'].tolist()[1:]  # Skip header
    
    # Create new DataFrame with ordered genes
    new_data = pd.DataFrame(0.0, index=expression_data.index, columns=gene_order)
    
    # Transfer existing gene data
    existing_genes = list(set(gene_order) & set(expression_data.columns))
    new_data.loc[:, existing_genes] = expression_data[existing_genes]
    
    # Save reordered data
    new_data.to_csv(output_file, index=True, index_label='')
    
    return (len(expression_data.columns), len(new_data.columns),
            len(set(gene_order) - set(expression_data.columns)),
            len(set(expression_data.columns) - set(gene_order)))

def process_all_files(input_prefix, output_prefix, tsv_file, seeds):
    """
    Process and reorder genes for all train/val files across multiple seeds.
    """
    print("Starting gene reordering process...")
    
    for seed in seeds:
        print(f"\nProcessing seed {seed}")
        
        # Process training set
        train_input = f"{input_prefix}_train_seed_{seed}.csv"
        train_output = f"{output_prefix}_train_seed_{seed}.csv"
        
        orig_genes, new_genes, added_genes, removed_genes = reorder_genes_by_tsv(
            train_input, tsv_file, train_output
        )
        print(f"Training set - Original genes: {orig_genes}, Final: {new_genes}")
        print(f"Added: {added_genes}, Removed: {removed_genes}")
        
        # Process validation set
        val_input = f"{input_prefix}_val_seed_{seed}.csv"
        val_output = f"{output_prefix}_val_seed_{seed}.csv"
        
        orig_genes, new_genes, added_genes, removed_genes = reorder_genes_by_tsv(
            val_input, tsv_file, val_output
        )
        print(f"Validation set - Original genes: {orig_genes}, Final: {new_genes}")
        print(f"Added: {added_genes}, Removed: {removed_genes}")

def main():
    # Initial setup
    input_data = 'batch_covid_subsampled_train.h5ad'
    output_prefix = 'Cod'
    tsv_file = 'OS_scRNA_gene_index.19264.tsv'
    seeds = range(5)
    
    # Step 1: Convert and split AnnData
    print("Step 1: Converting and splitting AnnData...")
    adata = sc.read_h5ad(input_data)
    convert_and_split_adata(adata, output_prefix, seeds=seeds, train_size=0.9)
    
    # Step 2: Reorder genes
    print("\nStep 2: Reordering genes according to TSV file...")
    process_all_files(output_prefix, output_prefix, tsv_file, seeds)
    
    # Verify final output
    print("\nVerifying final output (seed 0):")
    final_train = pd.read_csv(f'{output_prefix}_train_seed_0.csv', index_col=0)
    final_val = pd.read_csv(f'{output_prefix}_val_seed_0.csv', index_col=0)
    print(f"Final shapes - Train: {final_train.shape}, Val: {final_val.shape}")

if __name__ == "__main__":
    main()
