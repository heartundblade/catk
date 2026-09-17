import torch
import torch.nn as nn


class FourierEmbedding(nn.Module):
    def __init__(self, input_dim, hidden_dim=192, num_freq_bands=64):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.freqs = nn.Embedding(input_dim, num_freq_bands) if input_dim != 0 else None

        self.mlps = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(num_freq_bands * 2 + 1, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
            ) for _ in range(input_dim)]
        )

        self.to_out = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self._init()

    def _init(self):
        if self.freqs is not None:
            nn.init.normal_(self.freqs.weight, mean=0.0, std=1.0)
        for mlp in self.mlps:
            nn.init.normal_(mlp[-1].weight, std=1e-2)
            nn.init.constant_(mlp[-1].bias, 0)
        nn.init.normal_(self.to_out[-1].weight, std=1e-2)
        nn.init.constant_(self.to_out[-1].bias, 0)

    def forward(self, continuous_inputs):
        x = continuous_inputs.unsqueeze(-1) * self.freqs.weight * 2 * torch.pi
        x = torch.cat([x.cos(), x.sin(), continuous_inputs.unsqueeze(-1)], dim=-1)
        x = torch.stack([self.mlps[i](x[..., i, :]) for i in range(self.input_dim)]).sum(dim=0)

        return self.to_out(x)


class RelationEncoder(nn.Module):
    def __init__(self, hidden_dim=192, num_freq_bands=64):
        super().__init__()
        self.rel_encoder = FourierEmbedding(input_dim=3, hidden_dim=hidden_dim, num_freq_bands=num_freq_bands)

    def forward(self, relations):
        """
        Args:
            relations: [B, N1, N2, 4] — [local_x, local_y, cos_theta_diff, sin_theta_diff]
        Returns:
            embedded_relations: [B, N1, N2, hidden_dim]
        """
        r = torch.sqrt(relations[..., 0] ** 2 + relations[..., 1] ** 2)
        r = torch.where(r < 1e-4, torch.zeros_like(r), r)
        theta = torch.atan2(relations[..., 1], relations[..., 0])
        delta_heading = torch.atan2(relations[..., 3], relations[..., 2])
        rel_input = torch.stack([r, theta, delta_heading], dim=-1)
        return self.rel_encoder(rel_input)