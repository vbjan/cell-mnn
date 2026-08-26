import torch
from torch import nn
import pytorch_lightning as pl
from einops import rearrange, repeat
import wandb

from .metrics import MMDLoss, compute_wasserstein


class MLP(torch.nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        width: int = 64,
        depth: int = 3,
        init_scale: float = 0.01
    ):
        super().__init__()

        layers, fan = [], in_dim
        for _ in range(depth):                              
            lin = nn.Linear(fan, width)
            nn.init.kaiming_normal_(
                lin.weight, mode="fan_in", nonlinearity="leaky_relu")
            nn.init.zeros_(lin.bias)
            layers += [lin, nn.LeakyReLU()]
            fan = width

        lin_out = nn.Linear(width, out_dim)
        nn.init.kaiming_normal_(
            lin_out.weight, mode="fan_in", nonlinearity="linear")
        nn.init.zeros_(lin_out.bias)
        layers.append(lin_out)

        self.net = nn.Sequential(*layers)

        with torch.no_grad():
            for p in self.net[-1].parameters():   # last linear layer
                p.mul_(init_scale)

    def forward(self, x):
        return self.net(x)


class CellMNN(pl.LightningModule):
    def __init__(
        self,
        latent_dim: int,
        lr: float,
        lambda_kinetic: float = 0.1,
        gamma: float = 0.7,
        depth: int = 3,
        width: int = 64,
        init_scale: float = 0.01,
        weight_decay: float = 1e-5,
        mmd_sigma: float = 1.0,
    ):
        super(CellMNN, self).__init__()

        # Save all hyperparameters
        self.save_hyperparameters()

        # MLP
        self.latent_dim = latent_dim

        mlp_out_dim = self.latent_dim ** 2

        self.mlp = MLP(in_dim=self.latent_dim + 1,
                       out_dim=mlp_out_dim,
                       width=width,
                       depth=depth,
                       init_scale=init_scale)
        self.loss_fn = MMDLoss(sigma=mmd_sigma)
        self.lr = lr
        self.weight_decay = weight_decay

        self.lambda_kinetic = lambda_kinetic
        self.gamma = gamma

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(),
                                      self.lr,
                                      weight_decay=self.weight_decay)
        return optimizer

    def forward(
            self,
            x_t: torch.Tensor, # (B, 1, D) Convention: (batch, time, feature) dimension
            t: torch.Tensor,  # (B, 1, 1)
            decode_ts: torch.Tensor # (B, T, 1), forward allows broadcasting along the time dimension!
        ):
        # assert x_t, t and decode_ts are 3D tensors
        assert len(x_t.shape) == len(t.shape) == len(
            decode_ts.shape) == 3, "x_t, t and decode_ts must be 3D tensors"
        # assert batch dimensions are equal
        assert x_t.shape[0] == decode_ts.shape[0] == t.shape[0], "Batch dimensions of x_t, decode_ts and t must be equal"
        # assert ones in expected dimensions
        assert x_t.shape[1] == t.shape[1] == t.shape[2] == decode_ts.shape[
            2] == 1, "x_t, t and decode_ts must have ones in the expected dimensions"

        A = self.encode(x_t, t)  # (B, 1, D, D)

        system_traj = self.decode_trajectory(A, x_t, t, decode_ts)  # (B, T, D, 2)
        x_traj = system_traj.select(dim=-1, index=0)  # (B, T, D)
        x_dot_traj = system_traj.select(dim=-1, index=1)  # (B, T, D)

        return x_traj, x_dot_traj, A

    def encode(
        self,
        x_t,  # (B, 1, D)
        t  # (B, 1, 1)
    ):
        # Forward pass MLP - Broadcast x_t to match the time dimension of decode_ts
        obs = torch.cat((x_t, t), dim=-1)  # B 1 D+1
        rep = self.mlp(obs)   # (B, 1, latent_dim ** 2)

        A = rearrange(
            rep, 'b 1 (d1 d2) -> b 1 d1 d2', d1=self.latent_dim, d2=self.latent_dim
        )
        return A

    def decode_trajectory(
        self,
        A,  # (B, 1, D, D)
        x_t,  # (B, 1, D)
        t,  # (B, 1, 1)
        decode_ts  # (B, T, 1)
    ):
        batch_size, num_time_steps, _ = decode_ts.shape
        delta_t = (decode_ts - t).unsqueeze(-1)  # (B, T, 1, 1)
        phi_t = torch.linalg.matrix_exp(A * delta_t)  # (B, T, D, D)
        x_t = repeat(x_t, 'b 1 d -> b t d 1', t=decode_ts.shape[1])
        x_traj = phi_t @ x_t  # (B, T, D)

        # calculate first order derivative
        A_broadcast = repeat(A, 'b 1 d1 d2 -> b t d1 d2',
                             t=num_time_steps)       # (B, T, D, D)
        x_dot_traj = torch.matmul(A_broadcast, x_traj)

        # Stack x_traj and x_dot_traj along a new last dimension for the derivative order
        # (B, T, D, 2) where last dim is [x, dx/dt]
        system_traj = torch.cat([x_traj, x_dot_traj], dim=-1)

        return system_traj

    def training_step(self, batch, batch_idx):
        # TODO: optionally decode on more time points for kinetic energy reg
        # (B, 1, D), (B, 1, 1), (B, T, D), (B, T, 1)
        x_t, t, x_population, t_population = batch
        B, T, D = x_population.shape

        # Decode directly at the timepoints present in the batch (t_population
        # already excludes t_skip unless train_on_skip_day is set)
        x_traj, x_dot_traj, A = self.forward(x_t, t, t_population)

        # Create mask to exclude only the initial condition timepoint
        valid_mask_raw = (t_population.squeeze(-1) > t.squeeze(-1))
        # Create cumulative sum along time dimension to detect first True and beyond
        cumulative_true = torch.cumsum(valid_mask_raw.to(torch.int), dim=1)
        # First True will have cumulative sum of 1, subsequent Trues will have > 1
        valid_mask_i = [(cumulative_true == i) &
                        valid_mask_raw for i in range(1, T)]

        mmd_loss_future = 0.0

        for i_t in range(T):
            for i, valid_mask in enumerate(valid_mask_i):
                # (B,)  which cells can supervise timepoint i_t?
                m = valid_mask[:, i_t]
                if m.any():
                    mmd_loss_future += self.loss_fn(
                        x_traj[m, i_t, :],   # (N_t , D)
                        x_population[m, i_t, :]
                    ) * self.gamma ** i

        kinetic_loss = x_dot_traj.square().mean()
        loss = mmd_loss_future + self.lambda_kinetic * kinetic_loss

        # Calculate trace for each matrix in the batch and then average
        tr_A = torch.diagonal(A, dim1=-2, dim2=-1).sum(-1).mean()
        self.log('Tr(A)', tr_A)
        self.log('||A - A^T||_2',
                 (A - A.transpose(dim0=-2, dim1=-1)).square().mean())
        self.log('train_loss', loss)
        self.log('mmd_loss_future', mmd_loss_future)
        self.log('kinetic_loss', kinetic_loss)
        return loss

    def validation_step(self, batch, batch_idx, num_iter_max=200_000):
        # (I, D)
        t, x_t, x_t_skip, t_skip = batch

        x_t = rearrange(x_t, 'i d -> i 1 d')
        t = rearrange(t, 'i -> i 1 1')
        decode_t = torch.full_like(t, t_skip)

        x_traj, x_dot_traj, A = self.forward(x_t, t, decode_t)
        pred_dist = x_traj[:, 0, :].detach()

        self.log_A_eigenvalues(A)

        mmd = self.loss_fn(pred_dist, x_t_skip)
        self.log(f"val_mmd(t_skip={t_skip})", mmd.cpu().item())

        emd = compute_wasserstein(
            pred_dist.cpu().numpy(),
            x_t_skip.cpu().numpy(),
            num_iter_max=num_iter_max
        )
        self.log(f"val_emd(t_skip={t_skip})", emd)
        return emd
    
    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx, num_iter_max=1_000_000)

    def log_A_eigenvalues(self, A: torch.Tensor, tag: str = "A_eigenvalues"):
        """
        Log the eigenvalues of the predicted matrix **A** to Weights‑and‑Biases.

        Parameters
        ----------
        A   : torch.Tensor
              Shape (B, 1, D, D) or (B, D, D); the matrix (or batch of matrices)
              returned by the MLP inside `forward`.
        tag : str
              Base key under which histograms are stored in wandb.
        """
        with torch.no_grad():
            # drop the singleton time dim if it is present
            A = A.squeeze(1)                     # (B, D, D)
            eigvals = torch.linalg.eigvals(A)    # (B, D) complex tensor

            # wandb can plot complex numbers if we log the parts separately
            self.logger.experiment.log({
                f"{tag}/real": wandb.Histogram(eigvals.real.cpu().numpy().ravel()),
                f"{tag}/imag": wandb.Histogram(eigvals.imag.cpu().numpy().ravel()),
                "epoch": self.current_epoch,
                "global_step": self.global_step,
            })
