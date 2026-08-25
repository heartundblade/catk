import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.dp.model.module.timm import Mlp
from src.dp.model.module.local_attention import MultiheadAttentionLocal

def modulate(x, shift, scale, only_first=False):
    if only_first:
        x_first, x_rest = x[:, :1], x[:, 1:]
        x = torch.cat([x_first * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1), x_rest], dim=1)
    else:
        x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    return x


def scale(x, scale, only_first=False):
    if only_first:
        x_first, x_rest = x[:, :1], x[:, 1:]
        x = torch.cat([x_first * (1 + scale.unsqueeze(1)), x_rest], dim=1)
    else:
        x = x * (1 + scale.unsqueeze(1))

    return x


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning for ego and Cross-Attention.
    """
    def __init__(self, dim=192, heads=6, dropout=0.1, mlp_ratio=4.0, attention_mode="full"):
        super().__init__()
        self.num_heads = heads
        self.head_dim = dim // heads
        self.attn_dropout_rate = dropout
        self.attention_mode = attention_mode

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(2 * dim, 6 * dim, bias=True)
        )

        self.norm_sa = nn.LayerNorm(dim)
        self.sa_in_proj = nn.Linear(dim, 3 * dim, bias=True)
        self.sa_out_proj = nn.Linear(dim, dim, bias=True)

        self.norm1 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp1 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)

        # self.norm3 = nn.LayerNorm(dim)
        # self.ca_agent_q_proj = nn.Linear(dim, dim, bias=True)
        # self.ca_agent_kv_proj = nn.Linear(dim, 2 * dim, bias=True)
        # self.ca_agent_out_proj = nn.Linear(dim, dim, bias=True)
        # self.ca_agent_rel_k_proj = nn.Linear(dim, dim)
        # self.ca_agent_rel_v_proj = nn.Linear(dim, dim)

        # self.norm4 = nn.LayerNorm(dim)
        # self.ca_map_q_proj = nn.Linear(dim, dim, bias=True)
        # self.ca_map_kv_proj = nn.Linear(dim, 2 * dim, bias=True)
        # self.ca_map_out_proj = nn.Linear(dim, dim, bias=True)
        # self.ca_map_rel_k_proj = nn.Linear(dim, dim)
        # self.ca_map_rel_v_proj = nn.Linear(dim, dim)

        # self.norm5 = nn.LayerNorm(dim)
        # self.mlp2 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)

    def forward(self, x, agents_feat, maps_feat, y, attn_mask,
                rel_enc_agent=None, rel_enc_map=None,
                encoding_mask_agent=None, encoding_mask_map=None):
        B, C, D = x.shape
        H = self.num_heads
        d = self.head_dim

        # agents_feat: [B_orig, N_agent, D], x: [B_orig*P_orig, C, D]
        # Each row of x corresponds to (batch i//P_orig, agent i%P_orig)
        B_orig = agents_feat.shape[0]
        P_orig = B // B_orig
        batch_idx = torch.arange(B, device=x.device) // P_orig
        agent_idx = torch.arange(B, device=x.device) % P_orig
        agent_feat_per_row = agents_feat[batch_idx, agent_idx, :]  # [B, D]

        shift_sa, scale_sa, gate_sa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(
            torch.cat([y, agent_feat_per_row], dim=-1)
        ).chunk(6, dim=1)

        modulated_x = modulate(self.norm_sa(x), shift_sa, scale_sa)
        qkv = self.sa_in_proj(modulated_x).reshape(B, C, H, 3 * d)
        q, k, v = torch.split(qkv, d, dim=-1)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        sa_out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_dropout_rate if self.training else 0.0,
        )
        sa_out = sa_out.permute(0, 2, 1, 3).reshape(B, C, D)
        sa_out = self.sa_out_proj(sa_out)
        x = x + gate_sa.unsqueeze(1) * sa_out

        modulated_x = modulate(self.norm1(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp1(modulated_x)

        if encoding_mask_agent is not None:
            # encoding_mask_agent: [B_orig, N_agent]
            mask_per_row = encoding_mask_agent[batch_idx, agent_idx]  # [B_sp]
            x = x * (~mask_per_row).float().unsqueeze(-1).unsqueeze(-1)

        # if encoding_mask_agent is not None:
        #     x = x * (~encoding_mask_agent[:, :P]).float().unsqueeze(-1)


        # # === Stage 1: Cross-Attention with Agent ===
        # N_agent = agents_feat.shape[1]

        # modulated_x = modulate(self.norm3(x), shift_ca_agent, scale_ca_agent)
        # ca_q = self.ca_agent_q_proj(modulated_x).reshape(B, P, H, d).permute(0, 2, 1, 3)
        # ca_kv = self.ca_agent_kv_proj(agents_feat).reshape(B, N_agent, H, 2 * d)
        # ca_k, ca_v = torch.split(ca_kv, d, dim=-1)
        
        # ca_k = ca_k.permute(0, 2, 1, 3)
        # ca_v = ca_v.permute(0, 2, 1, 3)
        
        # # rel_k =  self.ca_agent_rel_k_proj(rel_enc_agent).view(B, P, P, H, d)
        # # rel_v = self.ca_agent_rel_v_proj(rel_enc_agent).view(B, P, P, H, d)
        # # ca_k = ca_k.permute(0, 2, 1, 3) + rel_k.permute(0, 3, 1, 2, 4)
        # # ca_v = ca_v.permute(0, 2, 1, 3) + rel_v.permute(0, 3, 1, 2, 4)

        # ca_attn_mask = None
        # rel_cross_v = None
        # if rel_enc_agent is not None:
        #     rel_enc = self.ca_agent_rel_k_proj(rel_enc_agent).reshape(B, P, -1, H, d)
        #     rel_cross_q = rel_enc.permute(0, 3, 1, 4, 2)
        #     ca_attn_mask = torch.matmul(ca_q.unsqueeze(-2), rel_cross_q).squeeze(-2) * (d ** -0.5)
        #     if self.attention_mode == "full":
        #         rel_cross_v = self.ca_agent_rel_v_proj(rel_enc_agent).reshape(B, P, -1, H, d)
        #         rel_cross_v = rel_cross_v.permute(0, 3, 1, 2, 4)

        # if encoding_mask_agent is not None:
        #     if ca_attn_mask is None:
        #         ca_attn_mask = torch.zeros(B, H, P, N_agent, device=x.device, dtype=x.dtype)
        #     ca_attn_mask = ca_attn_mask.masked_fill(encoding_mask_agent[:, None, None, :], float('-inf'))

        # # Mask diagonal to prevent each agent from attending to itself
        # diag_mask = torch.zeros(P, N_agent, device=x.device, dtype=x.dtype)
        # diag_mask.fill_diagonal_(float('-inf'))
        # diag_mask = diag_mask.unsqueeze(0).unsqueeze(0)
        # if ca_attn_mask is None:
        #     ca_attn_mask = diag_mask.expand(B, H, P, N_agent)
        # else:
        #     ca_attn_mask = ca_attn_mask + diag_mask

        # if rel_cross_v is not None:
        #     ca_score = (ca_q @ ca_k.transpose(-2, -1)) * (d ** -0.5)
        #     ca_score = ca_score + ca_attn_mask
        #     ca_attn = F.softmax(ca_score, dim=-1)
        #     ca_attn = ca_attn.nan_to_num(0.0)
        #     ca_attn = F.dropout(ca_attn, p=self.attn_dropout_rate, training=self.training)
        #     ca_out = ca_attn @ ca_v
        #     ca_out = ca_out + torch.matmul(ca_attn.unsqueeze(-2), rel_cross_v).squeeze(-2)
        # else:
        #     ca_out = F.scaled_dot_product_attention(
        #         ca_q, ca_k, ca_v,
        #         attn_mask=ca_attn_mask,
        #         dropout_p=self.attn_dropout_rate if self.training else 0.0,
        #     )
        
        # # ca_out = F.scaled_dot_product_attention(
        # #         ca_q, ca_k, ca_v,
        # #         attn_mask=ca_attn_mask,
        # #         dropout_p=self.attn_dropout_rate if self.training else 0.0,
        # #     )
        # ca_out = ca_out.nan_to_num(0.0)
        # ca_out = ca_out.permute(0, 2, 1, 3).reshape(B, P, D)
        # ca_out = self.ca_agent_out_proj(ca_out)
        # x = x + gate_ca_agent.unsqueeze(1) * ca_out



        # # === Stage 2: Cross-Attention with Map ===
        # N_map = maps_feat.shape[1]

        # modulated_x = modulate(self.norm4(x), shift_ca_map, scale_ca_map)
        # ca_q = self.ca_map_q_proj(modulated_x).reshape(B, P, H, d).permute(0, 2, 1, 3)
        # ca_kv = self.ca_map_kv_proj(maps_feat).reshape(B, N_map, H, 2 * d)
        # ca_k, ca_v = torch.split(ca_kv, d, dim=-1)

        # ca_k = ca_k.permute(0, 2, 1, 3)
        # ca_v = ca_v.permute(0, 2, 1, 3)

        # # rel_k = self.ca_map_rel_k_proj(rel_enc_map).view(B, N_map, N_map, H, d)
        # # rel_v = self.ca_map_rel_v_proj(rel_enc_map).view(B, N_map, N_map, H, d)

        # # ca_k = ca_k.permute(0, 2, 1, 3) + rel_k.permute(0, 3, 1, 2, 4)
        # # ca_v = ca_v.permute(0, 2, 1, 3) + rel_v.permute(0, 3, 1, 2, 4)

        # ca_attn_mask = None
        # rel_cross_v = None
        # if rel_enc_map is not None:
        #     rel_enc = self.ca_map_rel_k_proj(rel_enc_map).reshape(B, P, -1, H, d)
        #     rel_cross_q = rel_enc.permute(0, 3, 1, 4, 2)
        #     ca_attn_mask = torch.matmul(ca_q.unsqueeze(-2), rel_cross_q).squeeze(-2) * (d ** -0.5)
        #     if self.attention_mode == "full":
        #         rel_cross_v = self.ca_map_rel_v_proj(rel_enc_map).reshape(B, P, -1, H, d)
        #         rel_cross_v = rel_cross_v.permute(0, 3, 1, 2, 4)

        # if encoding_mask_map is not None:
        #     if ca_attn_mask is None:
        #         ca_attn_mask = torch.zeros(B, H, P, N_map, device=x.device, dtype=x.dtype)
        #     ca_attn_mask = ca_attn_mask.masked_fill(encoding_mask_map[:, None, None, :], float('-inf'))

        # if rel_cross_v is not None:
        #     ca_score = (ca_q @ ca_k.transpose(-2, -1)) * (d ** -0.5)
        #     ca_score = ca_score + ca_attn_mask
        #     ca_attn = F.softmax(ca_score, dim=-1)
        #     ca_attn = ca_attn.nan_to_num(0.0)
        #     ca_attn = F.dropout(ca_attn, p=self.attn_dropout_rate, training=self.training)
        #     ca_out = ca_attn @ ca_v
        #     ca_out = ca_out + torch.matmul(ca_attn.unsqueeze(-2), rel_cross_v).squeeze(-2)
        # else:
        #     ca_out = F.scaled_dot_product_attention(
        #         ca_q, ca_k, ca_v,
        #         attn_mask=ca_attn_mask,
        #         dropout_p=self.attn_dropout_rate if self.training else 0.0,
        #     )

        # # ca_out = F.scaled_dot_product_attention(
        # #     ca_q, ca_k, ca_v,
        # #     attn_mask=ca_attn_mask,
        # #     dropout_p=self.attn_dropout_rate if self.training else 0.0,
        # # )
        # ca_out = ca_out.nan_to_num(0.0)
        # ca_out = ca_out.permute(0, 2, 1, 3).reshape(B, P, D)
        # ca_out = self.ca_map_out_proj(ca_out)
        # x = x + gate_ca_map.unsqueeze(1) * ca_out

        # x = x + self.mlp2(self.norm5(x))

        return x


class LocalDiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning for ego and Cross-Attention.
    Based on DiTBlock but uses MultiheadAttentionLocal for sparse/local attention via index_pair.
    """
    def __init__(self, dim=192, heads=6, dropout=0.1, mlp_ratio=4.0, attention_mode="full"):
        super().__init__()
        self.num_heads = heads
        self.head_dim = dim // heads
        self.attn_dropout_rate = dropout
        self.attention_mode = attention_mode

        self.norm1 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp1 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 9 * dim, bias=True)
        )
        self.norm3 = nn.LayerNorm(dim)
        self.ca_agent_q_proj = nn.Linear(dim, dim, bias=True)
        self.ca_agent_kv_proj = nn.Linear(dim, 2 * dim, bias=True)
        self.ca_agent_out_proj = nn.Linear(dim, dim, bias=True)
        self.ca_agent_rel_proj = nn.Linear(dim, dim)
        self.ca_agent_rel_v_proj = nn.Linear(dim, dim)

        self.norm4 = nn.LayerNorm(dim)
        self.ca_map_q_proj = nn.Linear(dim, dim, bias=True)
        self.ca_map_kv_proj = nn.Linear(dim, 2 * dim, bias=True)
        self.ca_map_out_proj = nn.Linear(dim, dim, bias=True)
        self.ca_map_rel_proj = nn.Linear(dim, dim)
        self.ca_map_rel_v_proj = nn.Linear(dim, dim)

        self.norm5 = nn.LayerNorm(dim)
        self.mlp2 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)

        self.ca_agent_local_attn = MultiheadAttentionLocal(dim, heads, dropout)
        self.ca_map_local_attn = MultiheadAttentionLocal(dim, heads, dropout)

    def forward(self, x, cross_agent, cross_map, y,
                index_pair_agent=None, index_pair_map=None,
                rel_enc_agent=None, rel_enc_map=None,
                encoding_mask_agent=None, encoding_mask_map=None):
        B, P, D = x.shape
        H = self.num_heads
        d = self.head_dim

        shift_mlp, scale_mlp, gate_mlp, \
        shift_ca_agent, scale_ca_agent, gate_ca_agent, \
        shift_ca_map, scale_ca_map, gate_ca_map = self.adaLN_modulation(y).chunk(9, dim=1)

        modulated_x = modulate(self.norm1(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp1(modulated_x)

        # === Stage 1: Cross-Attention with Agent (Local) ===
        N_agent = cross_agent.shape[1]

        if index_pair_agent is None:
            index_pair_agent = torch.arange(N_agent, device=x.device).unsqueeze(0).expand(B * P, -1)

        modulated_x = modulate(self.norm3(x), shift_ca_agent, scale_ca_agent)
        ca_q = self.ca_agent_q_proj(modulated_x).reshape(B, P, H, d).reshape(B * P, H, d)
        ca_kv = self.ca_agent_kv_proj(cross_agent).reshape(B, N_agent, H, 2 * d)
        ca_k, ca_v = torch.split(ca_kv, d, dim=-1)
        ca_k = ca_k.reshape(B * N_agent, H, d)
        ca_v = ca_v.reshape(B * N_agent, H, d)

        ca_batch_cnt_q = [P] * B
        ca_batch_cnt_kv = [N_agent] * B

        ca_rel = None
        ca_rel_v = None
        if rel_enc_agent is not None and self.attention_mode == "full":
            rel_enc = self.ca_agent_rel_proj(rel_enc_agent).reshape(B, P, -1, H, d)
            rel_enc_flat = rel_enc.reshape(B * P, -1, H, d)
            ca_rel = rel_enc_flat[torch.arange(B * P, device=x.device)[:, None],
                                  index_pair_agent.clamp(min=0)]
            rel_v = self.ca_agent_rel_v_proj(rel_enc_agent).reshape(B, P, -1, H, d)
            rel_v_flat = rel_v.reshape(B * P, -1, H, d)
            ca_rel_v = rel_v_flat[torch.arange(B * P, device=x.device)[:, None],
                                  index_pair_agent.clamp(min=0)]

        ca_attn_mask_local = (index_pair_agent == -1)
        if encoding_mask_agent is not None:
            batch_ids = torch.arange(B, device=x.device).repeat_interleave(P)
            enc_mask_local = encoding_mask_agent[batch_ids[:, None], index_pair_agent.clamp(min=0)]
            ca_attn_mask_local = ca_attn_mask_local | enc_mask_local

        query_indices = torch.arange(P, device=x.device).unsqueeze(0).expand(B, -1).reshape(-1)
        diag_mask = (index_pair_agent == query_indices.unsqueeze(-1))
        ca_attn_mask_local = ca_attn_mask_local | diag_mask

        ca_out, _ = self.ca_agent_local_attn(
            query=x.reshape(B * P, D), key=cross_agent.reshape(B * N_agent, D), value=cross_agent.reshape(B * N_agent, D),
            index_pair=index_pair_agent,
            query_batch_cnt=ca_batch_cnt_q,
            key_batch_cnt=ca_batch_cnt_kv,
            attn_mask=ca_attn_mask_local,
            relation_encodings=ca_rel,
            rel_v=ca_rel_v,
            q_proj=ca_q, k_proj=ca_k, v_proj=ca_v,
            skip_out_proj=True,
        )
        ca_out = ca_out.reshape(B, P, D)
        ca_out = self.ca_agent_out_proj(ca_out)
        x = x + gate_ca_agent.unsqueeze(1) * ca_out

        # === Stage 2: Cross-Attention with Map (Local) ===
        N_map = cross_map.shape[1]

        if index_pair_map is None:
            index_pair_map = torch.arange(N_map, device=x.device).unsqueeze(0).expand(B * P, -1)

        modulated_x = modulate(self.norm4(x), shift_ca_map, scale_ca_map)
        ca_q = self.ca_map_q_proj(modulated_x).reshape(B, P, H, d).reshape(B * P, H, d)
        ca_kv = self.ca_map_kv_proj(cross_map).reshape(B, N_map, H, 2 * d)
        ca_k, ca_v = torch.split(ca_kv, d, dim=-1)
        ca_k = ca_k.reshape(B * N_map, H, d)
        ca_v = ca_v.reshape(B * N_map, H, d)

        ca_batch_cnt_q = [P] * B
        ca_batch_cnt_kv = [N_map] * B

        ca_rel = None
        ca_rel_v = None
        if rel_enc_map is not None and self.attention_mode == "full":
            rel_enc = self.ca_map_rel_proj(rel_enc_map).reshape(B, P, -1, H, d)
            rel_enc_flat = rel_enc.reshape(B * P, -1, H, d)
            ca_rel = rel_enc_flat[torch.arange(B * P, device=x.device)[:, None],
                                  index_pair_map.clamp(min=0)]
            rel_v = self.ca_map_rel_v_proj(rel_enc_map).reshape(B, P, -1, H, d)
            rel_v_flat = rel_v.reshape(B * P, -1, H, d)
            ca_rel_v = rel_v_flat[torch.arange(B * P, device=x.device)[:, None],
                                  index_pair_map.clamp(min=0)]

        ca_attn_mask_local = (index_pair_map == -1)
        if encoding_mask_map is not None:
            batch_ids = torch.arange(B, device=x.device).repeat_interleave(P)
            enc_mask_local = encoding_mask_map[batch_ids[:, None], index_pair_map.clamp(min=0)]
            ca_attn_mask_local = ca_attn_mask_local | enc_mask_local

        ca_out, _ = self.ca_map_local_attn(
            query=x.reshape(B * P, D), key=cross_map.reshape(B * N_map, D), value=cross_map.reshape(B * N_map, D),
            index_pair=index_pair_map,
            query_batch_cnt=ca_batch_cnt_q,
            key_batch_cnt=ca_batch_cnt_kv,
            attn_mask=ca_attn_mask_local,
            relation_encodings=ca_rel,
            rel_v=ca_rel_v,
            q_proj=ca_q, k_proj=ca_k, v_proj=ca_v,
            skip_out_proj=True,
        )
        ca_out = ca_out.reshape(B, P, D)
        ca_out = self.ca_map_out_proj(ca_out)
        x = x + gate_ca_map.unsqueeze(1) * ca_out

        x = x + self.mlp2(self.norm5(x))

        return x


# class RefineBlock(nn.Module):
#     """
#     Lightweight cross-attention refinement block.
#     Performs agent-agent and agent-map cross-attention with residual connections.
#     No adaLN/timestep conditioning - pure cross-attention for refinement before final layer.
#     """
#     def __init__(self, dim=192, heads=6, dropout=0.1):
#         super().__init__()
#         self.num_heads = heads
#         self.head_dim = dim // heads
#         self.attn_dropout_rate = dropout

#         self.norm_agent = nn.LayerNorm(dim)
#         self.ca_agent_q_proj = nn.Linear(dim, dim)
#         self.ca_agent_kv_proj = nn.Linear(dim, 2 * dim)
#         self.ca_agent_out_proj = nn.Linear(dim, dim)

#         self.norm_map = nn.LayerNorm(dim)
#         self.ca_map_q_proj = nn.Linear(dim, dim)
#         self.ca_map_kv_proj = nn.Linear(dim, 2 * dim)
#         self.ca_map_out_proj = nn.Linear(dim, dim)

#         self.ca_agent_rel_k_proj = nn.Linear(dim, dim)
#         self.ca_agent_rel_v_proj = nn.Linear(dim, dim)
#         self.ca_map_rel_k_proj = nn.Linear(dim, dim)
#         self.ca_map_rel_v_proj = nn.Linear(dim, dim)  

#     def forward(self, x, cross_agent, cross_map,
#                 rel_enc_agent=None, rel_enc_map=None,
#                 encoding_mask_agent=None, encoding_mask_map=None):
#         B, P, D = x.shape
#         H = self.num_heads
#         d = self.head_dim

#         # === Agent cross-attention ===
#         N_agent = cross_agent.shape[1]
#         q = self.ca_agent_q_proj(self.norm_agent(x)).reshape(B, P, H, d).permute(0, 2, 1, 3)
#         kv = self.ca_agent_kv_proj(cross_agent).reshape(B, N_agent, H, 2 * d)
#         k, v = torch.split(kv, d, dim=-1)
#         k = k.permute(0, 2, 1, 3)
#         v = v.permute(0, 2, 1, 3)

#         ca_attn_mask = None
#         rel_cross_v = None
#         if rel_enc_agent is not None:
#             rel_enc = self.ca_agent_rel_k_proj(rel_enc_agent).reshape(B, P, -1, H, d)
#             rel_cross_q = rel_enc.permute(0, 3, 1, 4, 2)
#             ca_attn_mask = torch.matmul(q.unsqueeze(-2), rel_cross_q).squeeze(-2) * (d ** -0.5)
#             rel_cross_v = self.ca_agent_rel_v_proj(rel_enc_agent).reshape(B, P, -1, H, d)
#             rel_cross_v = rel_cross_v.permute(0, 3, 1, 2, 4)

#         if encoding_mask_agent is not None:
#             if ca_attn_mask is None:
#                 ca_attn_mask = torch.zeros(B, H, P, N_agent, device=x.device, dtype=x.dtype)
#             ca_attn_mask = ca_attn_mask.masked_fill(encoding_mask_agent[:, None, None, :], float('-inf'))

#         # diag_mask = torch.zeros(P, N_agent, device=x.device, dtype=x.dtype)
#         # diag_mask.fill_diagonal_(float('-inf'))
#         # diag_mask = diag_mask.unsqueeze(0).unsqueeze(0)
#         # ca_attn_mask = diag_mask.expand(B, H, P, N_agent) if ca_attn_mask is None else ca_attn_mask + diag_mask

#         if rel_cross_v is not None:
#             ca_score = (q @ k.transpose(-2, -1)) * (d ** -0.5)
#             ca_score = ca_score + ca_attn_mask
#             ca_attn = F.softmax(ca_score, dim=-1)
#             ca_attn = ca_attn.nan_to_num(0.0)
#             ca_attn = F.dropout(ca_attn, p=self.attn_dropout_rate, training=self.training)
#             out = ca_attn @ v
#             out = out + torch.matmul(ca_attn.unsqueeze(-2), rel_cross_v).squeeze(-2)
#         else:
#             out = F.scaled_dot_product_attention(q, k, v, attn_mask=ca_attn_mask,
#                 dropout_p=self.attn_dropout_rate if self.training else 0.0)
#         out = out.nan_to_num(0.0).permute(0, 2, 1, 3).reshape(B, P, D)
#         x = x + self.ca_agent_out_proj(out)

#         # === Map cross-attention ===
#         N_map = cross_map.shape[1]
#         q = self.ca_map_q_proj(self.norm_map(x)).reshape(B, P, H, d).permute(0, 2, 1, 3)
#         kv = self.ca_map_kv_proj(cross_map).reshape(B, N_map, H, 2 * d)
#         k, v = torch.split(kv, d, dim=-1)
#         k = k.permute(0, 2, 1, 3)
#         v = v.permute(0, 2, 1, 3)

#         ca_attn_mask = None
#         rel_cross_v = None
#         if rel_enc_map is not None:
#             rel_enc = self.ca_map_rel_k_proj(rel_enc_map).reshape(B, P, -1, H, d)
#             rel_cross_q = rel_enc.permute(0, 3, 1, 4, 2)
#             ca_attn_mask = torch.matmul(q.unsqueeze(-2), rel_cross_q).squeeze(-2) * (d ** -0.5)
#             rel_cross_v = self.ca_map_rel_v_proj(rel_enc_map).reshape(B, P, -1, H, d)
#             rel_cross_v = rel_cross_v.permute(0, 3, 1, 2, 4)

#         if encoding_mask_map is not None:
#             if ca_attn_mask is None:
#                 ca_attn_mask = torch.zeros(B, H, P, N_map, device=x.device, dtype=x.dtype)
#             ca_attn_mask = ca_attn_mask.masked_fill(encoding_mask_map[:, None, None, :], float('-inf'))

#         if rel_cross_v is not None:
#             ca_score = (q @ k.transpose(-2, -1)) * (d ** -0.5)
#             ca_score = ca_score + ca_attn_mask
#             ca_attn = F.softmax(ca_score, dim=-1)
#             ca_attn = ca_attn.nan_to_num(0.0)
#             ca_attn = F.dropout(ca_attn, p=self.attn_dropout_rate, training=self.training)
#             out = ca_attn @ v
#             out = out + torch.matmul(ca_attn.unsqueeze(-2), rel_cross_v).squeeze(-2)
#         else:
#             out = F.scaled_dot_product_attention(q, k, v, attn_mask=ca_attn_mask,
#                 dropout_p=self.attn_dropout_rate if self.training else 0.0)
#         out = out.nan_to_num(0.0).permute(0, 2, 1, 3).reshape(B, P, D)
#         x = x + self.ca_map_out_proj(out)

#         return x

class ChunkAggregator(nn.Module):
    """VectorMapBackbone-style chunk aggregator for action sequences.
    
    point_mlp: per-point feature extraction (O(S) complexity)
    attn_pool: learnable query token + MHA -> single chunk vector
    
    Replaces bidirectional GRU with a fully parallel, lightweight design
    adapted from the VectorMapBackbone in encoder.py.
    """
    def __init__(self, hidden_dim=192, num_heads=4, dropout=0.1):
        super().__init__()
        self.point_mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            Mlp(hidden_dim, hidden_dim * 4, hidden_dim,
                act_layer=nn.GELU, drop=dropout),
        )
        self.query_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.attn_pool = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.attn_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        """
        Args:
            x: [N, S, D]  action features within a chunk
        Returns:
            [N, D]  aggregated chunk feature
        """
        x = self.point_mlp(x)                # [N, S, D]
        N = x.shape[0]
        q = self.query_token.expand(N, -1, -1)  # [N, 1, D]
        x, _ = self.attn_pool(q, x, x)       # [N, 1, D]
        x = self.attn_norm(x.squeeze(1))      # [N, D]
        return x

class RefineBlock(nn.Module):
    """
    Intra-chunk self-attention refinement block with agent adaLN modulation.
    """
    def __init__(self, dim, heads, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = Mlp(dim, dim * 4, dim, act_layer=nn.GELU, drop=0.)
        self.agent_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 2 * dim, bias=True)
        )

    def forward(self, x, agent_ctx):
        """
        Args:
            x: [B, S, D]  action features within a chunk
            agent_ctx: [B, D]  agent context for per-agent modulation
        Returns:
            x: [B, S, D]
        """
        shift, scale = self.agent_modulation(agent_ctx).chunk(2, dim=-1)  # [B, D] each
        x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

        h = self.norm1(x)
        h = self.attn(h, h, h)[0]
        x = x + h
        h = self.norm2(x)
        x = x + self.ffn(h)
        return x
    
class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    Each chunk has its own output head to prevent gradient concentration
    from all temporal segments onto a single weight matrix.
    """
    def __init__(self, hidden_size, output_size, num_chunks=1):
        super().__init__()
        self.num_chunks = num_chunks
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, hidden_size // 2, bias=True),
                nn.GELU(approximate="tanh"),
                nn.Dropout(0.1),
                nn.Linear(hidden_size // 2, output_size, bias=True)
            )
            for _ in range(num_chunks)
        ])

    def forward(self, x):
        # x: [N, S, D] where N = B*P*C
        N, S, D = x.shape
        C = self.num_chunks
        x = x.reshape(N // C, C, S, D)
        chunk_outputs = []
        for i in range(C):
            chunk_out = self.heads[i](x[:, i])  # [N/C, S, O]
            chunk_outputs.append(chunk_out)
        x = torch.stack(chunk_outputs, dim=1)   # [N/C, C, S, O]
        x = x.reshape(N, S, -1)                 # [N, S, O]
        return x