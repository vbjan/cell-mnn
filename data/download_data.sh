#!/usr/bin/env bash

mkdir -p data/ebdata data/citedata data/multidata

# Download Embryoid dataset from TrajectoryNet GitHub repository
wget -nc -O data/ebdata/eb_velocity_v5.npz \
  "https://raw.githubusercontent.com/KrishnaswamyLab/TrajectoryNet/master/data/eb_velocity_v5.npz"

# Download Cite dataset from Mendeley Data
wget -nc --content-disposition -P data/citedata \
  "https://data.mendeley.com/public-files/datasets/hhny5ff7yj/files/1862acf5-6294-4eb1-8644-d1c6d25e4126/file_downloaded"

# Download Multi dataset from Mendeley Data
wget --content-disposition -P data/multidata \
  "https://data.mendeley.com/public-files/datasets/hhny5ff7yj/files/5f4b6e5b-f122-4f5a-8ede-0d188c5cf00c/file_downloaded"


# For GRN discovery experiment: Download less preprocessed Embryoid dataset from Mendeley Data (which retains gene symbols)
wget -nc -O data/ebdata/ebdata_v3.h5ad \
  "https://data.mendeley.com/public-files/datasets/hhny5ff7yj/files/d82698f4-d143-442f-9a41-10be8ad02584/file_downloaded"

# Recompute PCA for this file because ebdata_v3.h5ad does not contain the principle components
python data/recompute_pca.py data/ebdata/ebdata_v3.h5ad 