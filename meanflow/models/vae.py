from typing import Optional, Tuple

import torch
import torch.nn as nn

from .unet import PositionalEmbedding


class LightVAE(nn.Module):
    def __init__(
        self,
        img_resolution: int,
        in_channels: int,
        latent_dim: int,
        hidden_channels: int = 32,
        augment_dim: int = 0,
    ) -> None:
        super().__init__()
        assert latent_dim > 0, "LightVAE requires a positive latent dimension"

        self.latent_dim = latent_dim
        self.hidden_channels = hidden_channels
        self.img_resolution = img_resolution
        self.input_channels = in_channels * 3
        self.augment_dim = augment_dim

        time_channels = hidden_channels

        self.conv_encoder = nn.Sequential(
            nn.Conv2d(self.input_channels, hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )

        self.pos_embed = PositionalEmbedding(num_channels=time_channels, endpoint=True)
        self.time_linear0 = nn.Linear(time_channels * 2, hidden_channels * 2)
        self.time_linear1 = nn.Linear(hidden_channels * 2, hidden_channels * 2)
        self.activation = nn.SiLU()
        self.map_augment = (
            nn.Linear(augment_dim, hidden_channels * 2, bias=False)
            if augment_dim > 0
            else None
        )

        combined_dim = hidden_channels * 2 + hidden_channels * 2

        self.combiner = nn.Sequential(
            nn.Linear(combined_dim, hidden_channels * 4),
            nn.SiLU(),
            nn.Linear(hidden_channels * 4, hidden_channels * 4),
            nn.SiLU(),
        )

        self.mu_proj = nn.Linear(hidden_channels * 4, latent_dim)
        self.logvar_proj = nn.Linear(hidden_channels * 4, latent_dim)

    def _compute_time_embedding(
        self,
        t: torch.Tensor,
        h: torch.Tensor,
        aug_cond: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if t.dim() > 1:
            t = t.view(t.shape[0])
        if h.dim() > 1:
            h = h.view(h.shape[0])

        t_embed = self.pos_embed(t)
        t_embed = t_embed.reshape(t_embed.shape[0], 2, -1).flip(1).reshape(*t_embed.shape)
        h_embed = self.pos_embed(h)
        h_embed = h_embed.reshape(h_embed.shape[0], 2, -1).flip(1).reshape(*h_embed.shape)

        time_feat = torch.cat([t_embed, h_embed], dim=1)
        if self.map_augment is not None and aug_cond is not None:
            time_feat = time_feat + self.map_augment(aug_cond)

        time_feat = self.time_linear0(time_feat)
        time_feat = self.activation(time_feat)
        time_feat = self.time_linear1(time_feat)
        time_feat = self.activation(time_feat)
        return time_feat

    def encode(
        self,
        e: torch.Tensor,
        x: torch.Tensor,
        z: torch.Tensor,
        time_steps: Tuple[torch.Tensor, torch.Tensor],
        aug_cond: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        t, h = time_steps
        stacked = torch.cat([e, x, z], dim=1)
        features = self.conv_encoder(stacked).flatten(1)

        time_feat = self._compute_time_embedding(t, h, aug_cond)

        combined = torch.cat([features, time_feat], dim=1)
        hidden = self.combiner(combined)

        mu = self.mu_proj(hidden)
        logvar = self.logvar_proj(hidden)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + eps * std

    def kl_divergence(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)

    def forward_latent(
        self,
        e: torch.Tensor,
        x: torch.Tensor,
        z: torch.Tensor,
        time_steps: Tuple[torch.Tensor, torch.Tensor],
        aug_cond: Optional[torch.Tensor],
        eps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        mu, logvar = self.encode(e, x, z, time_steps, aug_cond)
        if eps is None:
            eps = torch.randn_like(mu)
        return self.reparameterize(mu, logvar, eps)

    def kl_loss(
        self,
        e: torch.Tensor,
        x: torch.Tensor,
        z: torch.Tensor,
        time_steps: Tuple[torch.Tensor, torch.Tensor],
        aug_cond: Optional[torch.Tensor],
    ) -> torch.Tensor:
        mu, logvar = self.encode(e, x, z, time_steps, aug_cond)
        kl = self.kl_divergence(mu, logvar)
        return kl.mean()

