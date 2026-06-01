from typing import Any, Callable, Dict, List, Tuple

import torch
import torch.nn as nn

import lightning.pytorch as pl

from src.dp.model.module.encoder import Encoder
from src.dp.model.module.decoder import Decoder
from src.dp.utils.normalizer import StateNormalizer, ObservationNormalizer
from src.dp.utils.lr_schedule import CosineAnnealingWarmUpRestarts
from src.dp.utils.train_utils import transform_coords_to_sdc_frame

class Diffusion_Planner(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.cfg = config

        self.state_normalizer = StateNormalizer.from_json(self.cfg)
        self.observation_normalizer = ObservationNormalizer.from_json(self.cfg.normalization_file_path)

        self.encoder = Diffusion_Planner_Encoder(self.cfg)
        self.decoder = Diffusion_Planner_Decoder(self.cfg)

    @property
    def sde(self):
        return self.decoder.decoder.sde
    
    def forward(self, inputs):

        encoder_outputs = self.encoder(inputs)
        decoder_outputs = self.decoder(encoder_outputs, inputs)

        return encoder_outputs, decoder_outputs
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.learning_rate)
        scheduler = CosineAnnealingWarmUpRestarts(optimizer, self.cfg.train_epochs, self.cfg.warm_up_epoch)
        return [optimizer], [scheduler]
    
    def training_step(self, batch, batch_idx):
        # Create a deep copy of batch to avoid modifying the original
        inputs = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                inputs[key] = value.clone()
            else:
                inputs[key] = value

        inputs["agents_history"][..., :6] = transform_coords_to_sdc_frame(inputs["agents_history"][..., :6], inputs['sdc_coord'])
        inputs["agents_future"][..., :6] = transform_coords_to_sdc_frame(inputs["agents_future"][..., :6], inputs['sdc_coord'])
        inputs["lanes"][..., :4] = transform_coords_to_sdc_frame(inputs["lanes"][..., :4], inputs['sdc_coord'])
        inputs["roadlines"][..., :4] = transform_coords_to_sdc_frame(inputs["roadlines"][..., :4], inputs['sdc_coord'])
        inputs["static_maps"][..., :4] = transform_coords_to_sdc_frame(inputs["static_maps"][..., :4], inputs['sdc_coord'])
        
        inputs = self.observation_normalizer(inputs)

        loss = {}
        dpm_loss, loss, _ = self.diffusion_loss_func(
            inputs=inputs,
            marginal_prob=self.sde.marginal_prob,
            # futures=batch["agents_future"],
            norm=self.state_normalizer,
            loss=loss,
            model_type="x_start",
        )
        
        # Log training losses for monitoring
        self.log("train/dpm_loss", dpm_loss, prog_bar=True)
        self.log("train/ego_planning_loss", loss["ego_planning_loss"], prog_bar=True)
        self.log("train/neighbor_prediction_loss", loss["neighbor_prediction_loss"])
        self.log("train/total_loss", loss["ego_planning_loss"] + loss["neighbor_prediction_loss"])
        
        return dpm_loss
    
    def validation_step(self, batch, batch_idx):
        inputs = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                inputs[key] = value.clone()
            else:
                inputs[key] = value

        inputs["agents_history"][..., :6] = transform_coords_to_sdc_frame(inputs["agents_history"][..., :6], inputs['sdc_coord'])
        inputs["agents_future"][..., :6] = transform_coords_to_sdc_frame(inputs["agents_future"][..., :6], inputs['sdc_coord'])
        inputs["lanes"][..., :4] = transform_coords_to_sdc_frame(inputs["lanes"][..., :4], inputs['sdc_coord'])
        inputs["roadlines"][..., :4] = transform_coords_to_sdc_frame(inputs["roadlines"][..., :4], inputs['sdc_coord'])
        inputs["static_maps"][..., :4] = transform_coords_to_sdc_frame(inputs["static_maps"][..., :4], inputs['sdc_coord'])
        
        inputs = self.observation_normalizer(inputs)

        loss = {}
        dpm_loss, loss, _ = self.diffusion_loss_func(
            inputs=inputs,
            marginal_prob=self.sde.marginal_prob,
            norm=self.state_normalizer,
            loss=loss,
            model_type="x_start",
        )
        
        # Log validation losses for monitoring
        self.log("val/dpm_loss", dpm_loss, prog_bar=True, batch_size=inputs["agents_history"].shape[0])
        self.log("val/ego_planning_loss", loss["ego_planning_loss"], prog_bar=True)
        self.log("val/neighbor_prediction_loss", loss["neighbor_prediction_loss"])
        self.log("val/total_loss", loss["ego_planning_loss"] + loss["neighbor_prediction_loss"])
        
        return loss
    
    def diffusion_loss_func(
        self,
        inputs: Dict[str, torch.Tensor],
        marginal_prob: Callable[[torch.Tensor], torch.Tensor],

        # futures: Tuple[torch.Tensor, torch.Tensor],
        
        norm: StateNormalizer,
        loss: Dict[str, Any],

        model_type: str,
        eps: float = 1e-3,
    ):
        agents_future = inputs["agents_future"]  # [B, P, T, 9]
        agents_future_valid = inputs["agents_future_valid"]  # [B, P, T]

        B, P, T, _ = agents_future.shape
        # ego_current, neighbors_current = inputs["ego_current_state"][:, :4], inputs["neighbor_agents_past"][:, :Pn, -1, :4]
        # neighbor_current_mask = torch.sum(torch.ne(neighbors_current[..., :4], 0), dim=-1) == 0
        # neighbor_mask = torch.concat((neighbor_current_mask.unsqueeze(-1), neighbor_future_mask), dim=-1)

        # gt_future = torch.cat([ego_future[:, None, :, :], neighbors_future[..., :]], dim=1) # [B, P = 1 + 1 + neighbor, T, 4]
        current_states = inputs["agents_history"][:, :, -1:, :4] # [B, P, 1, 4]
        gt_future = torch.cat([current_states, norm(agents_future[..., 1:, :4])], dim=2) # [B, P, T+1, 4]

        if self.training:
            t = torch.rand(B, device=agents_future.device) * (1 - eps) + eps # [B,]
            z = torch.randn_like(agents_future[..., 1:, :4], device=agents_future.device) # [B, P, T, 4]
            
            mean, std = marginal_prob(gt_future[:, :, 1:, :], t)
            std = std.view(-1, *([1] * (len(gt_future[:, :, 1:, :].shape)-1)))

            xT = mean + std * z
            xT = torch.cat([gt_future[:, :, :1, :], xT], dim=2)
            
            merged_inputs = {
                **inputs,
                "sampled_trajectories": xT,
                "diffusion_time": t,
            }

            _, decoder_output = self.forward(merged_inputs)
            # Training mode returns "score"
            pred = decoder_output["score"][:, :, 1:, :]  # [B, P, T, 4]

            # Compute dpm_loss based on model_type
            if model_type == "score":
                dpm_loss = torch.sum((pred * std + z)**2, dim=-1)
            elif model_type == "x_start":
                dpm_loss = torch.sum((pred - gt_future[:, :, 1:, :])**2, dim=-1)
        
        else:
            _, decoder_output = self.forward(inputs)
            pred = decoder_output["prediction"]  # [B, P, T, 4]

            agents_future = inputs["agents_future"][:, :, 1:, :4]  # [B, P, T, 4]
            dpm_loss = torch.sum((pred - agents_future)**2, dim=-1)
        
        neighbor_future_valid = agents_future_valid.clone()
        neighbor_future_valid[:, 0, :] = False
        masked_prediction_loss = dpm_loss[neighbor_future_valid[:, :, 1:]]

        if masked_prediction_loss.numel() > 0:
            loss["neighbor_prediction_loss"] = masked_prediction_loss.mean()
        else:
            loss["neighbor_prediction_loss"] = torch.tensor(0.0, device=masked_prediction_loss.device)

        loss["ego_planning_loss"] = dpm_loss[:, 0, :].mean()

        assert not torch.isnan(dpm_loss).sum(), f"loss cannot be nan, z={z}"

        return dpm_loss[agents_future_valid[:, :, 1:]].mean(), loss, decoder_output

class Diffusion_Planner_Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.encoder = Encoder(config)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
        self.apply(_basic_init)

        # Initialize embedding MLP:
        nn.init.normal_(self.encoder.pos_emb.weight, std=0.02)
        nn.init.normal_(self.encoder.agents_encoder.type_emb.weight, std=0.02)
        nn.init.normal_(self.encoder.lane_encoder.speed_limit_emb.weight, std=0.02)
        nn.init.normal_(self.encoder.lane_encoder.traffic_emb.weight, std=0.02)

    def forward(self, inputs):

        encoder_outputs = self.encoder(inputs)

        return encoder_outputs
    

class Diffusion_Planner_Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.decoder = Decoder(config)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
        self.apply(_basic_init)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.decoder.dit.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.decoder.dit.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.decoder.dit.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.decoder.dit.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.decoder.dit.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.decoder.dit.final_layer.proj[-1].weight, 0)
        nn.init.constant_(self.decoder.dit.final_layer.proj[-1].bias, 0)

    def forward(self, encoder_outputs, inputs):

        decoder_outputs = self.decoder(encoder_outputs, inputs)
        
        return decoder_outputs