"""
Script to inflate data files by adding noise to original samples.

This script supports both .h5ad and .npz file formats. It loads data,
creates additional samples by adding Gaussian noise to the original data,
and saves the inflated dataset with an '_inflated' postfix.
"""

import argparse
import numpy as np
import pandas as pd
import scanpy as sc
from pathlib import Path
from scipy import sparse


def _to_dense_array(data):
    """Convert data to dense numpy array if needed."""
    if sparse.issparse(data):
        return data.toarray()
    return np.asarray(data)


def load_h5ad_data(input_file):
    """
    Load data from .h5ad file format.
    
    Args:
        input_file (str): Path to the input .h5ad file
    
    Returns:
        dict: Dictionary containing loaded data with keys:
            - 'X_pca': PCA components
            - 'obs': Observation metadata
            - 'obsm': Additional observation matrices
            - 'sample_labels': Sample labels (if available)
    """
    print(f"Loading .h5ad data from {input_file}...")
    adata = sc.read_h5ad(input_file)
    
    if 'X_pca' not in adata.obsm:
        raise ValueError("PCA components not found in adata.obsm['X_pca']. Please run PCA first.")
    
    data_dict = {
        'X_pca': _to_dense_array(adata.obsm['X_pca']),
        'obs': adata.obs.copy() if adata.obs is not None else None,
        'obsm': {k: _to_dense_array(v) for k, v in adata.obsm.items() if k != 'X_pca'},
        'sample_labels': adata.obs.get('sample_labels', None) if adata.obs is not None else None,
        'original_adata': adata  # Keep reference for metadata
    }
    
    return data_dict


def load_npz_data(input_file):
    """
    Load data from .npz file format.
    
    Args:
        input_file (str): Path to the input .npz file
    
    Returns:
        dict: Dictionary containing loaded data with keys:
            - 'X_pca': PCA components
            - 'obs': Observation metadata (constructed from available data)
            - 'obsm': Additional observation matrices
            - 'sample_labels': Sample labels
    """
    print(f"Loading .npz data from {input_file}...")
    npz_data = np.load(input_file)
    
    # Check for required keys
    if 'X_pca' not in npz_data:
        # Try alternative names that might contain PCA data
        pca_candidates = ['pcs', 'X_pca', 'pca']
        pca_key = None
        for candidate in pca_candidates:
            if candidate in npz_data:
                pca_key = candidate
                break
        
        if pca_key is None:
            raise ValueError(f"PCA components not found. Available keys: {list(npz_data.keys())}")
        
        X_pca = npz_data[pca_key]
    else:
        X_pca = npz_data['X_pca']
    
    # Get sample labels
    sample_labels = None
    if 'sample_labels' in npz_data:
        sample_labels = npz_data['sample_labels']
    elif 'days' in npz_data:
        sample_labels = npz_data['days']
    
    # Create observation metadata
    n_samples = X_pca.shape[0]
    obs_data = {}
    
    if sample_labels is not None:
        obs_data['sample_labels'] = sample_labels
    
    # Add any other 1D arrays as observation metadata
    for key, value in npz_data.items():
        if key not in ['X_pca', 'pcs', 'sample_labels', 'days'] and value.ndim == 1 and len(value) == n_samples:
            obs_data[key] = value
    
    obs = pd.DataFrame(obs_data, index=[f"cell_{i}" for i in range(n_samples)]) if obs_data else None
    
    # Collect other matrices (2D arrays that match sample count)
    obsm = {}
    for key, value in npz_data.items():
        if (key not in ['X_pca', 'pcs', 'sample_labels', 'days'] and 
            value.ndim == 2 and value.shape[0] == n_samples and 
            key not in obs_data):
            obsm[key] = value
    
    data_dict = {
        'X_pca': X_pca,
        'obs': obs,
        'obsm': obsm,
        'sample_labels': sample_labels,
        'original_npz': npz_data  # Keep reference for metadata
    }
    
    return data_dict


def load_data(input_file):
    """
    Load data from either .h5ad or .npz file format.
    
    Args:
        input_file (str): Path to the input file
    
    Returns:
        dict: Dictionary containing loaded data
    """
    input_path = Path(input_file)
    
    if input_path.suffix == '.h5ad':
        return load_h5ad_data(input_file)
    elif input_path.suffix == '.npz':
        return load_npz_data(input_file)
    else:
        raise ValueError(f"Unsupported file format: {input_path.suffix}. Supported formats: .h5ad, .npz")


def build_inflated_anndata(data_dict, target_samples, noise_std):
    """
    Build an inflated AnnData object from loaded data.
    
    Args:
        data_dict (dict): Dictionary containing loaded data
        target_samples (int): Target number of samples in the inflated dataset
        noise_std (float): Standard deviation of Gaussian noise to add
    
    Returns:
        AnnData: Inflated AnnData object
    """
    X_pca = data_dict['X_pca']
    obs = data_dict['obs']
    obsm = data_dict['obsm']
    
    original_samples = X_pca.shape[0]
    print(f"Original dataset has {original_samples} samples")
    
    if target_samples <= original_samples:
        print(f"Target samples ({target_samples}) <= original samples ({original_samples}). No inflation needed.")
        # Create AnnData with original data
        adata = sc.AnnData(X=np.empty((original_samples, 0)))
        adata.obsm['X_pca'] = X_pca
        if obs is not None:
            adata.obs = obs
        for key, value in obsm.items():
            adata.obsm[key] = value
        return adata
    
    additional_samples = target_samples - original_samples
    print(f"Adding {additional_samples} samples with noise std={noise_std}")
    
    # Generate indices for sampling (with replacement)
    sample_indices = np.random.choice(original_samples, size=additional_samples, replace=True)
    
    # Create noisy copies of the PCA components
    pca_noisy = X_pca[sample_indices] + np.random.normal(0, noise_std, 
                                                         (additional_samples, X_pca.shape[1]))
    
    # Combine original and noisy PCA data
    pca_inflated = np.vstack([X_pca, pca_noisy])
    
    # Create new AnnData object with empty X (discarding gene expression)
    adata_inflated = sc.AnnData(X=np.empty((target_samples, 0)))
    
    # Add PCA components to obsm
    adata_inflated.obsm['X_pca'] = pca_inflated
    
    # Copy original observations metadata and extend with noisy samples
    if obs is not None and len(obs.columns) > 0:
        obs_noisy = obs.iloc[sample_indices].copy()
        obs_noisy.index = [f"noisy_{i}" for i in range(additional_samples)]
        adata_inflated.obs = pd.concat([obs, obs_noisy], ignore_index=False)
    
    # Copy other obsm data
    for key, value in obsm.items():
        # Extend other obsm data by copying for noisy samples
        value_noisy = value[sample_indices]
        adata_inflated.obsm[key] = np.vstack([value, value_noisy])
    
    print(f"Inflated dataset has {adata_inflated.n_obs} samples")
    return adata_inflated


def inflate_data(input_file, target_samples, noise_std):
    """
    Inflate a data file by adding noise to create more samples.
    
    Args:
        input_file (str): Path to the input file (.h5ad or .npz)
        target_samples (int): Target number of samples in the inflated dataset
        noise_std (float): Standard deviation of Gaussian noise to add
    
    Returns:
        AnnData: Inflated AnnData object
    """
    # Load data using appropriate loader
    data_dict = load_data(input_file)
    
    # Build inflated AnnData object
    adata_inflated = build_inflated_anndata(data_dict, target_samples, noise_std)
    
    return adata_inflated


def main():
    parser = argparse.ArgumentParser(description="Inflate data file by adding noise to samples")
    parser.add_argument("input_file", type=str, help="Path to input file (.h5ad or .npz)")
    parser.add_argument("target_samples", type=int, help="Target number of samples")
    parser.add_argument("--noise_std", type=float, default=0.1, 
                       help="Standard deviation of Gaussian noise (default: 0.1)")
    
    args = parser.parse_args()
    
    # Validate input file
    input_path = Path(args.input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file {args.input_file} does not exist")
    
    if input_path.suffix not in ['.h5ad', '.npz']:
        raise ValueError("Input file must have .h5ad or .npz extension")
    
    # Generate output filename
    output_path = input_path.parent / f"{input_path.stem}_inflated.h5ad"
    
    print(f"Input file: {args.input_file}")
    print(f"Target samples: {args.target_samples}")
    print(f"Noise std: {args.noise_std}")
    print(f"Output file: {output_path}")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Inflate data
    adata_inflated = inflate_data(args.input_file, args.target_samples, args.noise_std)
    
    # Save inflated data
    print(f"Saving inflated data to {output_path}...")
    adata_inflated.write_h5ad(output_path)
    print("Done!")


if __name__ == "__main__":
    main()
