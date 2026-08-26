import torch
import numpy as np


def predict_gene_interaction(
        models: list[torch.nn.Module], 
        x_sample: np.ndarray, 
        day: int, 
        W: torch.Tensor, 
        device: torch.device,
    ):
    x_sample = torch.from_numpy(x_sample).float().unsqueeze(1).to(device)  # (N,1,5)
    batch_size = x_sample.shape[0]
    t = day * torch.ones((batch_size, 1, 1), device=device)  # (N,1)

    As = []
    for model in models:
        P, eigenvals = model.encode(x_sample, t)  
        P_inv = torch.linalg.inv(P)  
        A = model.construct_A(P_inv, eigenvals, P).squeeze().detach() # (N, n_pcs, n_pcs)
        As.append(A)
    A = torch.stack(As, dim=0).mean(dim=0)  # Average over models (N, n_pcs, n_pcs)

    A_gene_space = torch.einsum('vp, bpq, wq -> bvw', W, A, W)  # (batch, n_vars, n_vars)

    # Map x_sample into gene expression space
    x_sample_squeezed = x_sample.squeeze(1)      # (N, n_pcs)
    x_sample_gene_space = x_sample_squeezed @ W.t() 

    interaction_matrix = A_gene_space * x_sample_gene_space.unsqueeze(1) 
    return interaction_matrix, x_sample_gene_space