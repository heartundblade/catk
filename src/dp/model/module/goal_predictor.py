import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossTransformer(nn.Module):
    def __init__(self, hidden_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout, batch_first=True)
        self.norm_1 = nn.LayerNorm(hidden_dim)
        self.norm_2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4), 
            nn.GELU(), 
            nn.Dropout(dropout), 
            nn.Linear(hidden_dim * 4, hidden_dim), 
            nn.Dropout(dropout)
        )

    def forward(self, query, key, attn_mask=None, key_mask=None):
        value = key
        
        if key_mask is not None:
            attention_output, _ = self.cross_attention(query, key, value, key_padding_mask=key_mask)
        elif attn_mask is not None:
            attention_output, _ = self.cross_attention(query, key, value, attn_mask=attn_mask)
        else:
            attention_output, _ = self.cross_attention(query, key, value)
        
        attention_output = self.norm_1(attention_output)
        output = self.norm_2(self.ffn(attention_output) + attention_output)
        
        return output


class GoalPredictor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self._agents_len = config.agent_num
        self._future_len = config.future_len
        self._action_len = config.action_len
        
        # Anchor encoder - project 2D anchor points to hidden dimension
        self.anchor_encoder = nn.Sequential(
            nn.Linear(2, 128), 
            nn.ReLU(), 
            nn.Linear(128, 256)
        )
        
        # Cross attention layers
        self.attention_layers = nn.ModuleList([
            CrossTransformer(256, 8) for _ in range(4)
        ])
        
        # Action decoder
        self.act_decoder = nn.Sequential(
            nn.Linear(256, 256), 
            nn.ELU(), 
            nn.Dropout(0.1),
            nn.Linear(256, (self._future_len // self._action_len) * 2)
        )
        
        # Score decoder
        self.score_decoder = nn.Sequential(
            nn.Linear(256, 128), 
            nn.ELU(), 
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )
        
    def forward(self, encoder_outputs, anchors):
        """
        Args:
            encoder_outputs: dict containing 'encoding' from the encoder
            anchors: [B, P, Q, 2] anchor points
        Returns:
            goal_actions: [B, P, Q, num_actions, 2]
            goal_scores: [B, P, Q]
        """
        encodings = encoder_outputs['encoding']  # [B, total_tokens, hidden_dim]
        
        # Get agent encodings (first P tokens are agents)
        agent_encodings = encodings[:, :self._agents_len]  # [B, P, hidden_dim]
        
        # Encode anchors
        anchors_encoded = self.anchor_encoder(anchors)  # [B, P, Q, hidden_dim]
        
        # Create query: agent encoding + anchor encoding
        query = agent_encodings[:, :, None, :] + anchors_encoded  # [B, P, Q, hidden_dim]
        
        num_batch, num_agents, num_queries, _ = query.shape
        
        
        # for i in range(self._agents_len):
        #     query_content = self.attention_layers[0](query[:, i], encodings)
        #     query_content = self.attention_layers[1](query_content, encodings)
        #     query_content = query_content + query[:, i]
        #     query_content = self.attention_layers[2](query_content, encodings)
        #     query_content = self.attention_layers[3](query_content, encodings)
            
        #     actions.append(self.act_decoder(query_content).reshape(
        #         num_batch, num_queries, self._future_len // self._action_len, 2
        #     ))
        #     scores.append(self.score_decoder(query_content).squeeze(-1))
        
        # actions = torch.stack(actions, dim=1)  # [B, P, Q, num_actions, 2]
        # scores = torch.stack(scores, dim=1)     # [B, P, Q]


        # Process all agents in parallel using matrix operations
        # Reshape query from [B, P, Q, hidden_dim] to [B*P, Q, hidden_dim]
        query_flat = query.view(num_batch * num_agents, num_queries, -1)
        # Repeat encodings for each agent: [B, total_tokens, hidden_dim] -> [B*P, total_tokens, hidden_dim]
        encodings_repeated = encodings.unsqueeze(1).repeat(1, num_agents, 1, 1)
        encodings_flat = encodings_repeated.view(num_batch * num_agents, -1, query.size(-1))
        
        # Apply attention layers in parallel
        query_content = self.attention_layers[0](query_flat, encodings_flat)
        query_content = self.attention_layers[1](query_content, encodings_flat)
        query_content = query_content + query_flat  # Residual connection
        query_content = self.attention_layers[2](query_content, encodings_flat)
        query_content = self.attention_layers[3](query_content, encodings_flat)
        
        # Reshape back to [B, P, Q, hidden_dim]
        query_content = query_content.view(num_batch, num_agents, num_queries, -1)
        
        # Decode actions and scores
        actions = self.act_decoder(query_content).view(
            num_batch, num_agents, num_queries, self._future_len // self._action_len, 2
        )  # [B, P, Q, num_actions, 2]
        scores = self.score_decoder(query_content).squeeze(-1)  # [B, P, Q]
        
        return actions, scores