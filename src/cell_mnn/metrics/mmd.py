import torch
import torch.nn as nn


def _subsample(x: torch.Tensor, n: int | None) -> torch.Tensor:
    """`n` samples along the sample axis of `x` (..., N, D), without replacement."""
    if n is None or x.shape[-2] <= n:
        return x
    idx = torch.randperm(x.shape[-2], device=x.device)[:n]
    return x[..., idx, :]


class MMDLoss(nn.Module):
    """
    Computes the MMD^2 between two sets of samples x1, x2 
    using the Laplace kernel: k(u, v) = exp(-||u - v||_1 / sigma).

    Shapes:
        x1: (B, N, D) or (N, D)
        x2: (B, M, D) or (M, D)
      where B = batch size (optional), 
            N and M = number of samples per distribution,
            D = feature dimension.
    """
    def __init__(self, sigma=1.0, eps=1e-8):
        super().__init__()
        self.sigma = sigma
        self.eps = eps

    def laplace_kernel(self, x, y):
        """
        Computes Laplace kernel matrix between all pairs of points in x and y.
        x, y: [B, N, D] and [B, M, D] respectively.
        Returns: [B, N, M] kernel matrix.
        """
        # cdist(..., p=1) gives pairwise L1 distances
        dists = torch.cdist(x, y, p=1)
        dists = dists / x.shape[-1]  # Normalize by feature dimension D

        # Threshold small distances to eps:
        dists = torch.clamp_min(dists, self.eps)

        return torch.exp(-dists / self.sigma)   # dividing by D

    def forward(self, x1, x2, n=None):
        """
        `n` samples are drawn from each of x1 and x2 without replacement before the
        kernel matrices are built; None (the default, and what training uses) keeps
        every sample. 
        """
        # Ensure x1, x2 have a batch dimension of size 1 if none is present
        if x1.dim() == 2:
            x1 = x1.unsqueeze(0)  # -> [1, N, D]
        if x2.dim() == 2:
            x2 = x2.unsqueeze(0)  # -> [1, M, D]

        x1 = _subsample(x1, n)
        x2 = _subsample(x2, n)

        Kxx = self.laplace_kernel(x1, x1)  # [B, N, N]
        Kyy = self.laplace_kernel(x2, x2)  # [B, M, M]
        Kxy = self.laplace_kernel(x1, x2)  # [B, N, M]

        # Biased MMD^2 = E_xx + E_yy - 2 E_xy
        mmd_per_batch = Kxx.mean(dim=[1,2]) + Kyy.mean(dim=[1,2]) - 2 * Kxy.mean(dim=[1,2])
        
        # Return the average MMD over the batch to get a scalar
        return mmd_per_batch.mean()
