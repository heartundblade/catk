import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiheadAttentionLocal(nn.Module):

    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        self.dropout = dropout

        self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        self.to_k_proj = nn.Linear(embed_dim, embed_dim)
        self.to_v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.constant_(self.in_proj_bias, 0.0)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)
        nn.init.xavier_uniform_(self.to_k_proj.weight)
        nn.init.constant_(self.to_k_proj.bias, 0.0)
        nn.init.xavier_uniform_(self.to_v_proj.weight)
        nn.init.constant_(self.to_v_proj.bias, 0.0)

    def forward(self, query, key, value, index_pair, 
                query_batch_cnt=None, key_batch_cnt=None, 
                index_pair_batch=None, attn_mask=None,
                relation_encodings=None,  # [N, M, H, D] or pre-gathered [N, L, H, D]
                rel_v=None,               # [N, M, H, D] or pre-gathered [N, L, H, D], separate value projection
                key_padding_mask=None,    # [M] bool, True=invalid key
                q_proj=None, k_proj=None, v_proj=None,  # pre-projected [*, H, D]
                skip_out_proj=False):
        if q_proj is not None:
            q = q_proj * self.scaling
            k = k_proj
            v = v_proj
            N = q.shape[0]
        else:
            N, C = query.shape
            q = F.linear(query, self.in_proj_weight[:C], self.in_proj_bias[:C]) * self.scaling
            k = F.linear(key, self.in_proj_weight[C:2*C], self.in_proj_bias[C:2*C])
            v = F.linear(value, self.in_proj_weight[2*C:], self.in_proj_bias[2*C:])
            q = q.view(N, self.num_heads, self.head_dim)
            k = k.view(-1, self.num_heads, self.head_dim)
            v = v.view(-1, self.num_heads, self.head_dim)

        L = index_pair.shape[1]

        valid_mask = (index_pair != -1)  # [N, L]
        safe_idx = index_pair.clamp(min=0)

        if query_batch_cnt is not None and key_batch_cnt is not None:
            key_batch_tensor = torch.as_tensor(key_batch_cnt).to(q.device)
            query_batch_tensor = torch.as_tensor(query_batch_cnt).to(q.device)
            key_offsets = F.pad(torch.cumsum(key_batch_tensor, dim=0)[:-1], (1, 0), value=0)
            query_batch_ids = torch.repeat_interleave(torch.arange(len(query_batch_cnt), device=q.device), query_batch_tensor)
            safe_idx = safe_idx + key_offsets[query_batch_ids][:, None]

        if relation_encodings is not None:
            if relation_encodings.shape[1] == L:
                rel_local = relation_encodings
            else:
                rel_local = relation_encodings[torch.arange(N, device=q.device)[:, None], safe_idx]
        
        # rel_k = self.to_k_proj(rel_local).view(N, L, self.num_heads, self.head_dim)
        # rel_v = self.to_v_proj(rel_local).view(N, L, self.num_heads, self.head_dim)
        # k_local = k[safe_idx] + rel_k  # [N, L, H, D]
        # v_local = v[safe_idx] + rel_v  # [N, L, H, D]
        k_local = k[safe_idx]  # [N, L, H, D]
        v_local = v[safe_idx]  # [N, L, H, D]

        rel_attn_bias = None
        rel_v_local = None
        if relation_encodings is not None:
            if relation_encodings.shape[1] == L:
                rel_local = relation_encodings
            else:
                rel_local = relation_encodings[torch.arange(N, device=q.device)[:, None], safe_idx]
            rel_attn_bias = torch.einsum('nhd,nlhd->nlh', q, rel_local)
            if rel_v is not None:
                if rel_v.shape[1] == L:
                    rel_v_local = rel_v
                else:
                    rel_v_local = rel_v[torch.arange(N, device=q.device)[:, None], safe_idx]
            else:
                rel_v_local = None

        attn = torch.einsum('nhd,nlhd->nlh', q, k_local)
        attn = attn.masked_fill(~valid_mask.unsqueeze(-1), float('-inf'))
        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask.unsqueeze(-1), float('-inf'))
        if key_padding_mask is not None:
            key_mask_local = key_padding_mask[safe_idx]
            attn = attn.masked_fill(key_mask_local.unsqueeze(-1), float('-inf'))
        if rel_attn_bias is not None:
            attn = attn + rel_attn_bias
        attn = F.softmax(attn, dim=1)
        attn = attn.nan_to_num(0.0)
        attn = F.dropout(attn, p=self.dropout, training=self.training)

        output = torch.einsum('nlh,nlhd->nhd', attn, v_local)
        if rel_v_local is not None:
            output = output + torch.einsum('nlh,nlhd->nhd', attn, rel_v_local)
        output = output.reshape(N, self.embed_dim)
        if not skip_out_proj:
            output = self.out_proj(output)

        return output, attn.sum(dim=-1) / self.num_heads