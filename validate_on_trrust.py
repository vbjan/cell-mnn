import torch
from lib.data.data_preprocessing import get_data
from lib.model import CellMNN
from lib.interpretability import predict_gene_interaction
from lib.utils import fix_seed
import numpy as np
import os
import scanpy as sc
import pandas as pd
import argparse
import sys


# Parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(
        description='Filter for gene interactions in predicted operators')
    parser.add_argument('--gene', type=str, default="POU5F1",
                        help='Name of the source gene to filter interactions for')
    parser.add_argument('--num_batches', type=int, default=100,)
    parser.add_argument('--batch_size', type=int, default=2,)
    parser.add_argument('--models_folder', 
                        type=str, 
                        default="pre_trained_models/cellmnn_1ev0",
                        help='Choose either "cellmnn_1ev0" or "cellmnn" for pretrained models contained in repo.')
    return parser.parse_args()


def filter_for_interaction(models, X_pca, day, W, interaction, num_batches, batch_size, source_gene_mask, target_gene_mask):
    num_corr_sign = 0
    num_source_activations = 0

    for batch_idx in range(num_batches):
        x_sample = X_pca[day][batch_size * batch_idx:batch_size * (batch_idx+1)]  # (N,1,5)
        interaction_matrix, x_sample_gene_space = predict_gene_interaction(models, x_sample, day, W, device)

        for i in range(batch_size):
            interaction_str = interaction_matrix[i][target_gene_mask][:, source_gene_mask].item()
            source_gene_activated = x_sample_gene_space[i].squeeze()[source_gene_mask].item() > 0.  
            if source_gene_activated:
                num_source_activations += 1
                if interaction == "Activation":
                    num_corr_sign += interaction_str > 0
                elif interaction == "Repression":
                    num_corr_sign += interaction_str < 0
                elif interaction == "Unknown":
                    num_corr_sign += 1
                else:
                    raise ValueError(f"Unknown interaction type: {interaction}")

    return num_corr_sign, num_source_activations


if __name__ == "__main__":
    args = parse_args()
    fix_seed(42)

    DAY_NAMES = ["day0-5", "day6-11", "day12-17", "day18-23", "day24-30"]
    save_path = f"assets/filter_for_interactions"
    os.makedirs(save_path, exist_ok=True)
    num_batches = args.num_batches

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    # Automatically retrieve model names starting with "mnn" from the models folder
    model_names = sorted([
        name for name in os.listdir(args.models_folder)
        if name.startswith("mnn") and os.path.isdir(os.path.join(args.models_folder, name))
    ])
    print(f"Found {len(model_names)} models in {args.models_folder}")
  
    models = []
    for model_name in model_names:
        model = CellMNN.load_from_checkpoint(
            f"{args.models_folder}/{model_name}/best-model.ckpt").to(device)
        model.eval()
        models.append(model)

    pca_dims = models[0].latent_dim
    proposed_interactions = pd.read_csv(
        f'data/tf_targets_trrust/{args.gene}_targets.human.tsv', 
        sep='\t',
        comment='#',     # skip the line starting with #
        header=None,     # no header line left now
        names=['TF', 'Target', 'Type', 'Reference']
    )
    print(proposed_interactions)
    adata = sc.read_h5ad('data/ebdata/ebdata_v3_recomputed_pca.h5ad')
    W = adata.varm['PCs']           # shape = (n_vars, n_pcs)
    n_vars = W.shape[0]
    W = torch.from_numpy(W[:, :pca_dims]).to(device)  # (n_vars, n_pcs)
    X_pca = get_data(ds_name="embryoid_less_preprocessed", val_prop=0.)["X_train"]   # list-like: 5 days
    num_days = len(X_pca)
    gene_names = np.array([name.split()[0] for name in adata.var_names])
    gene_names_set = set(gene_names)

    classification_dict = {
        "TP": 0,
        "TN": 0,
        "FP": 0,
        "FN": 0,
        "Total": 0
    }
    def update_classification_dict(interaction, correct_sign):
        if interaction == "Activation":
            if correct_sign:
                classification_dict["TP"] += 1
            else:
                classification_dict["FP"] += 1
        elif interaction == "Repression":
            if not correct_sign:
                classification_dict["TN"] += 1
            else:
                classification_dict["FN"] += 1
        else:
            raise ValueError(f"Unknown interaction type: {interaction}")
        classification_dict["Total"] += 1


    for row in proposed_interactions.itertuples():
        print(f"\n{'='*60}")
        print(f"genes of interest: {row.TF}, {row.Target} {row.Type}")
        print(f"{'='*60}")
        if row.Target in gene_names_set and row.Type != "Unknown":
            source_gene_mask = np.isin(gene_names, [row.TF])
            target_gene_mask = np.isin(gene_names, [row.Target])

            # Filter for certain interactions and whether they are recovered
            num_corr_per_day, num_inters_per_day = [], []
            for day in range(num_days):
                num_corr_sign, total_interactions = filter_for_interaction(
                    models=models,
                    X_pca=X_pca,
                    day=day,
                    W=W,
                    interaction=row.Type,
                    num_batches=num_batches,
                    batch_size=args.batch_size,
                    source_gene_mask=source_gene_mask,
                    target_gene_mask=target_gene_mask
                )

                if total_interactions != 0:
                    print(f" # of cells with correct sign: {num_corr_sign} out of {total_interactions} ({num_corr_sign / total_interactions * 100:.2f}%)")
                
                num_corr_per_day.append(num_corr_sign)
                num_inters_per_day.append(total_interactions)

            average_corr = np.sum(num_corr_per_day) / np.sum(num_inters_per_day) 
            corr_sign = average_corr > 0.5
            print(f"Average correct sign across all days: {average_corr * 100:.2f}%")

            if row.Type != "Unknown":
                update_classification_dict(interaction=row.Type, correct_sign=corr_sign)
        else:
            print(f"Gene {row.Target} not found in dataset. Skipping interaction check.")

    print(f"\n {classification_dict}")

    # Compute Recall, Precision, F1-Score
    recall = classification_dict["TP"] / (classification_dict["TP"] + classification_dict["FN"]) if (classification_dict["TP"] + classification_dict["FN"]) > 0 else 0
    precision = classification_dict["TP"] / (classification_dict["TP"] + classification_dict["FP"]) if (classification_dict["TP"] + classification_dict["FP"]) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    print(f"\nRecall: {recall:.2f}, Precision: {precision:.2f}, F1-Score: {f1_score:.2f}")

    # Save results to CSV
    results_df = pd.DataFrame({
        "gene": [args.gene],
        "recall": [recall],
        "precision": [precision],
        "f1_score": [f1_score],
        "TP": [classification_dict["TP"]],
        "TN": [classification_dict["TN"]],
        "FP": [classification_dict["FP"]],
        "FN": [classification_dict["FN"]],
        "num_samples": [args.batch_size * num_batches]
    })
    results_csv_path = os.path.join(save_path, f"{args.gene}_results.csv")
    results_df.to_csv(results_csv_path, index=False)
    print(f"Results saved to {results_csv_path}")

    sys.stdout.close()