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
            ) for _ in range(input_dim)])

        self.to_out = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, continuous_inputs):
        x = continuous_inputs.unsqueeze(-1) * self.freqs.weight * 2 * torch.pi
        x = torch.cat([x.cos(), x.sin(), continuous_inputs.unsqueeze(-1)], dim=-1)
        x = torch.stack([self.mlps[i](x[:, :, :, i]) for i in range(self.input_dim)]).sum(dim=0)

        return self.to_out(x)


class RelationEncoder(nn.Module):
    def __init__(self, hidden_dim=192, num_freq_bands=64):
        super().__init__()
        self.relation_pos_encoder = FourierEmbedding(input_dim=2, hidden_dim=hidden_dim, num_freq_bands=num_freq_bands)
        self.relation_angle_encoder = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, relations):
        """
        Args:
            relations: [B, N1, N2, 4]
        Returns:
            embedded_relations: [B, N1, N2, hidden_dim]
        """
        encoded_pos = self.relation_pos_encoder(relations[..., :2])
        encoded_angle = self.relation_angle_encoder(relations[..., 2:])
        return encoded_pos + encoded_angle