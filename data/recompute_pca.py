#!/usr/bin/env python
"""Recompute PCA for an AnnData h5ad file."""

import argparse
from pathlib import Path

import scanpy as sc

def main():
    parser = argparse.ArgumentParser(
        description="Recompute PCA for an AnnData h5ad file."
    )
    parser.add_argument(
        "data_path",
        type=str,
        help="Path to the input h5ad file.",
    )
    parser.add_argument(
        "-n", "--n_components",
        type=int,
        default=50,
        help="Number of PCA components to compute (default: 50).",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output file path (default: <input>_recomputed_pca.h5ad).",
    )
    args = parser.parse_args()

    # Load data
    adata = sc.read_h5ad(args.data_path)
    print(adata)

    # Recompute PCA
    sc.tl.pca(adata, n_comps=args.n_components, random_state=0)

    # Print PCA loadings shape
    pca_directions = adata.varm['PCs']
    print("Shape of PCA loadings:", pca_directions.shape)

    # Save the updated AnnData object
    input_path = Path(args.data_path)
    output_path = args.output or input_path.parent / f"{input_path.stem}_recomputed_pca.h5ad"
    sc.write(output_path, adata)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
