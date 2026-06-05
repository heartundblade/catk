import torch
import torch.nn as nn
# from timm.models.layers import Mlp
# from timm.layers import DropPath

from src.dp.model.module.mixer import MixerBlock
from src.dp.model.module.timm import Mlp, DropPath

class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.hidden_dim = config.hidden_dim

        self.token_num = config.agent_num + config.static_map_num + config.lane_num + config.roadline_num

        self.agents_encoder = AgentFusionEncoder(config.time_len, drop_path_rate=config.encoder_drop_path_rate, hidden_dim=config.hidden_dim, depth=config.encoder_depth)
        self.static_map_encoder = StaticMapFusionEncoder(config.static_map_len, drop_path_rate=config.encoder_drop_path_rate, hidden_dim=config.hidden_dim)
        self.lane_encoder = LaneFusionEncoder(config.lane_len, drop_path_rate=config.encoder_drop_path_rate, hidden_dim=config.hidden_dim, depth=config.encoder_depth)
        self.roadline_encoder = RoadlineFusionEncoder(config.roadline_len, drop_path_rate=config.encoder_drop_path_rate, hidden_dim=config.hidden_dim, depth=config.encoder_depth)

        self.fusion = FusionEncoder(
            hidden_dim=config.hidden_dim, 
            num_heads=config.num_heads, 
            drop_path_rate=config.encoder_drop_path_rate, 
            depth=config.encoder_depth, 
            device=config.device
        )

        # position embedding encode x, y, cos, sin, type
        self.pos_emb = nn.Linear(8, config.hidden_dim)

    def forward(self, inputs):

        encoder_outputs = {}

        # agents
        agents = inputs['agents_history']
        agents_type = inputs['agents_type']

        # vector maps
        lanes = inputs['lanes']
        lanes_speed_limit = inputs['lanes_speed_limit']
        lanes_has_speed_limit = inputs['lanes_has_speed_limit']
        roadlines = inputs['roadlines']
        static_maps = inputs['static_maps']

        B = agents.shape[0]

        encoding_agents, agents_mask, agents_pos = self.agents_encoder(agents, agents_type)
        encoding_static, static_mask, static_pos = self.static_map_encoder(static_maps)
        encoding_lanes, lanes_mask, lane_pos = self.lane_encoder(lanes, lanes_speed_limit, lanes_has_speed_limit)
        encoding_roadlines, roadlines_mask, roadline_pos = self.roadline_encoder(roadlines)

        encoding_input = torch.cat([encoding_agents, encoding_static, encoding_lanes, encoding_roadlines], dim=1)

        encoding_pos = torch.cat([agents_pos, static_pos, lane_pos, roadline_pos], dim=1).view(B * self.token_num, -1)
        encoding_mask = torch.cat([agents_mask, static_mask, lanes_mask, roadlines_mask], dim=1).view(-1)
        encoding_pos = self.pos_emb(encoding_pos[~encoding_mask])
        encoding_pos_result = torch.zeros((B * self.token_num, self.hidden_dim), device=encoding_pos.device)
        encoding_pos_result[~encoding_mask] = encoding_pos  # Fill in valid parts

        encoding_input = encoding_input + encoding_pos_result.view(B, self.token_num, -1)

        encoder_outputs['encoding'] = self.fusion(encoding_input, encoding_mask.view(B, self.token_num))

        return encoder_outputs


class SelfAttentionBlock(nn.Module):
    def __init__(self, dim=192, heads=6, dropout=0.1, mlp_ratio=4.0):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)

        self.drop_path = DropPath(dropout) if dropout > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=dropout)

    def forward(self, x, mask):
        x = x + self.drop_path(self.attn(self.norm1(x), x, x, key_padding_mask=mask)[0])
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class AgentFusionEncoder(nn.Module):
    def __init__(self, time_len, drop_path_rate=0.3, hidden_dim=192, depth=3, tokens_mlp_dim=64, channels_mlp_dim=128):
        super().__init__()

        self._hidden_dim = hidden_dim
        self._channel = channels_mlp_dim

        self.type_emb = nn.Embedding(4, channels_mlp_dim)

        self.channel_pre_project = Mlp(in_features=8+1, hidden_features=channels_mlp_dim, out_features=channels_mlp_dim, act_layer=nn.GELU, drop=0.)
        self.token_pre_project = Mlp(in_features=time_len, hidden_features=tokens_mlp_dim, out_features=tokens_mlp_dim, act_layer=nn.GELU, drop=0.)

        self.blocks = nn.ModuleList([MixerBlock(tokens_mlp_dim, channels_mlp_dim, drop_path_rate) for i in range(depth)])

        self.norm = nn.LayerNorm(channels_mlp_dim)
        self.emb_project = Mlp(in_features=channels_mlp_dim, hidden_features=hidden_dim, out_features=hidden_dim, act_layer=nn.GELU, drop=drop_path_rate)


    def forward(self, x, agent_type):
        '''
        x: B, P, V, D (x, y, cos, sin, vx, vy, w, l, z)
        agent_type: B, P, 1 (type)
        '''
        x = x[..., :8]

        pos = x[:, :, -1, :8].clone() # x, y, cos, sin
        # agent: [1,0,0,0]
        pos[..., -4:] = 0.0
        pos[..., -4] = 1.0
        
        B, P, V, _ = x.shape
        mask_v = torch.sum(torch.ne(x[..., :8], 0), dim=-1).to(x.device) == 0
        mask_p = torch.sum(~mask_v, dim=-1) == 0
        x = torch.cat([x, (~mask_v).float().unsqueeze(-1)], dim=-1)
        x = x.view(B * P, V, -1)

        valid_indices = ~mask_p.view(-1) 
        x = x[valid_indices] 

        x = self.channel_pre_project(x)
        x = x.permute(0, 2, 1)
        x = self.token_pre_project(x)
        x = x.permute(0, 2, 1)
        for block in self.blocks:
            x = block(x)

        # pooling
        x = torch.mean(x, dim=1)

        agent_type = agent_type.long().view(B * P)
        agent_type = agent_type[valid_indices]
        type_embedding = self.type_emb(agent_type)  # Type embedding for valid data
        x = x + type_embedding

        x = self.emb_project(self.norm(x))

        x_result = torch.zeros((B * P, x.shape[-1]), device=x.device)
        x_result[valid_indices] = x  # Fill in valid parts
        
        return x_result.view(B, P, -1) , mask_p.reshape(B, -1), pos.view(B, P, -1)


class StaticMapFusionEncoder(nn.Module):
    def __init__(self, static_map_len, drop_path_rate=0.3, depth=3, tokens_mlp_dim=64, hidden_dim=192, channels_mlp_dim=128, device='cuda'):
        super().__init__()

        self._hidden_dim = hidden_dim
        self.type_emb = nn.Embedding(3, hidden_dim)

        # self.projection = Mlp(in_features=dim, hidden_features=hidden_dim, out_features=hidden_dim, act_layer=nn.GELU, drop=drop_path_rate)

        self._map_len = static_map_len
        self._channel = channels_mlp_dim

        self.type_emb = nn.Embedding(3, channels_mlp_dim)

        self.channel_pre_project = Mlp(in_features=4, hidden_features=channels_mlp_dim, out_features=channels_mlp_dim, act_layer=nn.GELU, drop=0.)
        self.token_pre_project = Mlp(in_features=static_map_len, hidden_features=tokens_mlp_dim, out_features=tokens_mlp_dim, act_layer=nn.GELU, drop=0.)

        self.blocks = nn.ModuleList([MixerBlock(tokens_mlp_dim, channels_mlp_dim, drop_path_rate) for i in range(depth)])

        self.norm = nn.LayerNorm(channels_mlp_dim)
        self.emb_project = Mlp(in_features=channels_mlp_dim, hidden_features=hidden_dim, out_features=hidden_dim, act_layer=nn.GELU, drop=drop_path_rate)

    def forward(self, x):
        '''
        x: B, P, V, D (x, y, cos, sin, type)
        '''
        B, P, V, _ = x.shape
        static_map_type = x[:, :, 0, 4]
        x = x[..., :4]

        spatial_pos = x[:, :, int(self._map_len / 2), :4].clone()
        
        pos = torch.zeros(B, P, 8, device=x.device)
        pos[..., :4] = spatial_pos
        # static: [0,1,0,0]
        pos[..., 5] = 1.0

        mask_v = torch.sum(torch.ne(x[..., :4], 0), dim=-1).to(x.device) == 0
        mask_p = torch.sum(~mask_v, dim=-1) == 0
        x = x.view(B * P, V, -1)

        valid_indices = ~mask_p.view(-1) 
        x = x[valid_indices] 

        x = self.channel_pre_project(x)
        x = x.permute(0, 2, 1)
        x = self.token_pre_project(x)
        x = x.permute(0, 2, 1)
        for block in self.blocks:
            x = block(x)

        x = torch.mean(x, dim=1)

        static_map_type = static_map_type.long().view(B * P)
        static_map_type = static_map_type[valid_indices]
        type_embedding = self.type_emb(static_map_type)
        
        x = x + type_embedding
        x = self.emb_project(self.norm(x))

        x_result = torch.zeros((B * P, x.shape[-1]), device=x.device)
        x_result[valid_indices] = x  # Fill in valid parts
        
        return x_result.view(B, P, -1) , mask_p.reshape(B, -1), pos.view(B, P, -1)


class LaneFusionEncoder(nn.Module):
    def __init__(self, lane_len, drop_path_rate=0.3, hidden_dim=192, depth=3, tokens_mlp_dim=64, channels_mlp_dim=128):
        super().__init__()

        self._lane_len = lane_len
        self._channel = channels_mlp_dim

        self.speed_limit_emb = nn.Linear(1, channels_mlp_dim)
        self.unknown_speed_emb = nn.Embedding(1, channels_mlp_dim)
        self.traffic_emb = nn.Embedding(9, channels_mlp_dim)
        self.type_emb = nn.Embedding(4, channels_mlp_dim)

        self.channel_pre_project = Mlp(in_features=4, hidden_features=channels_mlp_dim, out_features=channels_mlp_dim, act_layer=nn.GELU, drop=0.)
        self.token_pre_project = Mlp(in_features=lane_len, hidden_features=tokens_mlp_dim, out_features=tokens_mlp_dim, act_layer=nn.GELU, drop=0.)

        self.blocks = nn.ModuleList([MixerBlock(tokens_mlp_dim, channels_mlp_dim, drop_path_rate) for i in range(depth)])

        self.norm = nn.LayerNorm(channels_mlp_dim)
        self.emb_project = Mlp(in_features=channels_mlp_dim, hidden_features=hidden_dim, out_features=hidden_dim, act_layer=nn.GELU, drop=drop_path_rate)

    def forward(self, x, speed_limit, has_speed_limit):
        '''
        x: B, P, V, D (x, y, cos, sin, traffic, type)
        speed_limit: B, P, 1
        has_speed_limit: B, P, 1
        '''
        B, P, V, _ = x.shape

        traffic = x[:, :, 0, 4]
        lane_type = x[:, :, 0, 5]
        x = x[..., :4]
        
        spatial_pos = x[:, :, int(self._lane_len / 2), :4].clone()
        
        pos = torch.zeros(B, P, 8, device=x.device)
        pos[..., :4] = spatial_pos
        # lane: [0,0,1,0]
        pos[..., 6] = 1.0

        mask_v = torch.sum(torch.ne(x[..., :4], 0), dim=-1).to(x.device) == 0
        mask_p = torch.sum(~mask_v, dim=-1) == 0
        x = x.view(B * P, V, -1)

        valid_indices = ~mask_p.view(-1) 
        x = x[valid_indices] 

        x = self.channel_pre_project(x)
        x = x.permute(0, 2, 1)
        x = self.token_pre_project(x)
        x = x.permute(0, 2, 1)
        for block in self.blocks:
            x = block(x)  

        x = torch.mean(x, dim=1)

        # Reshape speed_limit and traffic to match flattened dimensions
        speed_limit = speed_limit.view(B * P, 1)
        has_speed_limit = has_speed_limit.view(B * P, 1)
        traffic = traffic.long().view(B * P)

        # Apply embedding directly to valid speed limit data
        has_speed_limit = has_speed_limit[valid_indices].squeeze(-1)
        speed_limit = speed_limit[valid_indices].squeeze(-1)
        # speed_limit_embedding = torch.zeros((speed_limit.shape[0], self._channel), device=x.device)

        # if has_speed_limit.sum() > 0:
        #     speed_limit_with_limit = self.speed_limit_emb(speed_limit[has_speed_limit].unsqueeze(-1))
        #     speed_limit_embedding[has_speed_limit] = speed_limit_with_limit

        # if (~has_speed_limit).sum() > 0:
        #     speed_limit_no_limit = self.unknown_speed_emb.weight.expand(
        #         (~has_speed_limit).sum().item(), -1
        #     )
        #     speed_limit_embedding[~has_speed_limit] = speed_limit_no_limit
        
        unknown_indices = torch.zeros(speed_limit.shape[0], dtype=torch.long, device=x.device)
        unknown_emb_all = self.unknown_speed_emb(unknown_indices)  # [B*P, C]

        speed_limit_emb_all = self.speed_limit_emb(speed_limit.unsqueeze(-1))  # [V, C]

        speed_limit_embedding = torch.where(
            has_speed_limit.unsqueeze(-1).bool(),   # [V, 1] -> broadcast to [V, C]
            speed_limit_emb_all,
            unknown_emb_all
        )

        # Process traffic lights directly for valid positions
        traffic = traffic[valid_indices]
        traffic_light_embedding = self.traffic_emb(traffic)  # Traffic light embedding for valid data

        lane_type = lane_type.long().view(B * P)
        lane_type = lane_type[valid_indices]
        type_embedding = self.type_emb(lane_type)

        x = x + speed_limit_embedding + traffic_light_embedding + type_embedding
        x = self.emb_project(self.norm(x))

        x_result = torch.zeros((B * P, x.shape[-1]), device=x.device)
        x_result[valid_indices] = x  # Fill in valid parts
        
        return x_result.view(B, P, -1) , mask_p.reshape(B, -1), pos.view(B, P, -1)

class RoadlineFusionEncoder(nn.Module):
    def __init__(self, roadline_len, drop_path_rate=0.3, hidden_dim=192, depth=3, tokens_mlp_dim=64, channels_mlp_dim=128):
        super().__init__()

        self._roadline_len = roadline_len
        self._channel = channels_mlp_dim

        self.type_emb = nn.Embedding(12, channels_mlp_dim)

        self.channel_pre_project = Mlp(in_features=4, hidden_features=channels_mlp_dim, out_features=channels_mlp_dim, act_layer=nn.GELU, drop=0.)
        self.token_pre_project = Mlp(in_features=roadline_len, hidden_features=tokens_mlp_dim, out_features=tokens_mlp_dim, act_layer=nn.GELU, drop=0.)

        self.blocks = nn.ModuleList([MixerBlock(tokens_mlp_dim, channels_mlp_dim, drop_path_rate) for i in range(depth)])

        self.norm = nn.LayerNorm(channels_mlp_dim)
        self.emb_project = Mlp(in_features=channels_mlp_dim, hidden_features=hidden_dim, out_features=hidden_dim, act_layer=nn.GELU, drop=drop_path_rate)

    def forward(self, x):
        '''
        x: B, P, V, D (x, y, cos, sin, type)
        '''
        B, P, V, _ = x.shape
        roadline_type = x[:, :, 0, 4:]
        x = x[..., :4]

        spatial_pos = x[:, :, int(self._roadline_len / 2), :4].clone()
        
        pos = torch.zeros(B, P, 8, device=x.device)
        pos[..., :4] = spatial_pos
        # roadline: [0,0,0,1]
        pos[..., 7] = 1.0

        mask_v = torch.sum(torch.ne(x[..., :4], 0), dim=-1).to(x.device) == 0
        mask_p = torch.sum(~mask_v, dim=-1) == 0
        x = x.view(B * P, V, -1)

        valid_indices = ~mask_p.view(-1) 
        x = x[valid_indices] 

        x = self.channel_pre_project(x)
        x = x.permute(0, 2, 1)
        x = self.token_pre_project(x)
        x = x.permute(0, 2, 1)
        for block in self.blocks:
            x = block(x)

        x = torch.mean(x, dim=1)

        roadline_type = roadline_type.long().view(B * P)
        roadline_type = roadline_type[valid_indices]
        type_embedding = self.type_emb(roadline_type)
        
        x = x + type_embedding
        x = self.emb_project(self.norm(x))

        x_result = torch.zeros((B * P, x.shape[-1]), device=x.device)
        x_result[valid_indices] = x  # Fill in valid parts
        
        return x_result.view(B, P, -1) , mask_p.reshape(B, -1), pos.view(B, P, -1)

class FusionEncoder(nn.Module):
    def __init__(self, hidden_dim=192, num_heads=6, drop_path_rate=0.3, depth=3, device='cuda'):
        super().__init__()

        dpr = drop_path_rate

        self.blocks = nn.ModuleList(
            [SelfAttentionBlock(hidden_dim, num_heads, dropout=dpr) for i in range(depth)]
        )

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, mask):

        mask[:, 0] = False

        for b in self.blocks:
            x = b(x, mask)

        return self.norm(x)