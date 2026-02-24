import torch

from torchcfm.optimal_transport import wasserstein

if __name__ == "__main__":
    bs = 50000          
    dim = 5           
    x0 = torch.randn(bs, dim, device='cuda') 
    x1 = torch.randn(bs, dim, device='cuda')
    wasserstein(x0, x1, method='sinkhorn')      