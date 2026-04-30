import torch
import lightning.pytorch as pl
from .modules import Encoder, Denoiser, GoalPredictor
from .utils import DDPM_Sampler
from .model_utils import (
    inverse_kinematics, 
    roll_out, 
    batch_transform_trajs_to_global_frame,
    batch_transform_polylines_to_local_frame,
    batch_transform_trajs_to_local_frame,
    batch_calculate_relations,
)
from src.vbd.sim_agent.utils import *
from src.vbd.data_preprocess.data_preprocess import calculate_relations

from torch.nn.functional import smooth_l1_loss, cross_entropy
from src.smart.metrics import (
    WOSACMetric,
    WOSACMetrics
)
from src.utils.wosac_utils import get_scenario_rollouts, get_scenario_id_int_tensor

class VBD(pl.LightningModule):
    """
    Versertile Behavior Diffusion model.
    """

    def __init__(
        self,
        cfg: dict,
    ):
        """
        Initialize the VBD model.

        Args:
            cfg (dict): Configuration parameters for the model.
        """
        super().__init__()
        self.save_hyperparameters()
        
        self.cfg = cfg
        self._future_len = cfg['future_len']
        self._agents_len = cfg['agents_len']
        self._action_len = cfg['action_len']
        self._diffusion_steps = cfg['diffusion_steps']
        self._encoder_layers = cfg['encoder_layers']
        self._encoder_version = cfg.get('encoder_version', 'v1')
        self._action_mean = cfg['action_mean']
        self._action_std = cfg['action_std']
        self._step_len = cfg['step_len']
        
        self._train_encoder = cfg.get('train_encoder', True)
        self._train_denoiser = cfg.get('train_denoiser', True)
        self._train_predictor = cfg.get('train_predictor', True)
        self._with_predictor = cfg.get('with_predictor', True)
        self._prediction_type = cfg.get('prediction_type', 'sample')
        self._schedule_type = cfg.get('schedule_type', 'cosine')
        self._replay_buffer = cfg.get('replay_buffer', False)
        self._embeding_dim = cfg.get('embeding_dim', 5) # By default, the embed is the noised trajectory so the dimension is 5
        
        self._val_open_loop = cfg.get('val_open_loop', True)
        self._val_closed_loop = cfg.get('val_closed_loop', False)
        self._n_rollout_closed_val = cfg.get('n_rollout_closed_val', 32)
        self.log_epoch = cfg.get('log_epoch', -1)

        self.encoder = Encoder(self._encoder_layers, version=self._encoder_version)
        
        self.denoiser = Denoiser(
            future_len=self._future_len,
            action_len=self._action_len,
            agents_len=self._agents_len,
            steps=self._diffusion_steps,
            input_dim = self._embeding_dim,
        )
        if self._with_predictor:
            self.predictor = GoalPredictor(
                future_len=self._future_len,
                agents_len=self._agents_len,
                action_len=self._action_len,
            )
        else:
            self.predictor = None
            self._train_predictor = False

        self.noise_scheduler = DDPM_Sampler(
            steps=self._diffusion_steps,
            schedule=self._schedule_type,
            s = cfg.get('schedule_s', 0.0),
            e = cfg.get('schedule_e', 1.0),
            tau = cfg.get('schedule_tau', 1.0),
            scale = cfg.get('schedule_scale', 1.0),
        )
        if cfg.get('fast_wosac_metric', False):
            self.wosac_metrics = WOSACMetric('2024')
        else:
            self.wosac_metrics = WOSACMetrics("val_closed")

        self.register_buffer('action_mean', torch.tensor(self._action_mean))  
        self.register_buffer('action_std', torch.tensor(self._action_std))

    ################### Training Setup ###################
    def configure_optimizers(self):
        '''
        This function is called by Lightning to create the optimizer and learning rate scheduler.
        '''
        if not self._train_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        if not self._train_denoiser: 
            for param in self.denoiser.parameters():
                param.requires_grad = False
        if self._with_predictor and (not self._train_predictor):
            for param in self.predictor.parameters():
                param.requires_grad = False

        params_to_update = []
        for param in self.parameters():
            if param.requires_grad == True:
                params_to_update.append(param)              
        
        assert len(params_to_update) > 0, 'No parameters to update'
        
        optimizer = torch.optim.AdamW(
            params_to_update, 
            lr=self.cfg['lr'],
            weight_decay=self.cfg['weight_decay']
        )
        
        lr_warmpup_step = self.cfg['lr_warmup_step']
        lr_step_freq = self.cfg['lr_step_freq']
        lr_step_gamma = self.cfg['lr_step_gamma']

        def lr_update(step, warmup_step, step_size, gamma):
            if step < warmup_step:
                # warm up lr
                lr_scale = 1 - (warmup_step - step) / warmup_step * 0.95
            else:
                n = (step - warmup_step) // step_size
                lr_scale = gamma ** n
        
            if lr_scale < 1e-2:
                lr_scale = 1e-2
            elif lr_scale > 1:
                lr_scale = 1
        
            return lr_scale
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: lr_update(
                step, 
                lr_warmpup_step, 
                lr_step_freq,
                lr_step_gamma,
            )
        )
        
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def forward(self, inputs, noised_actions_normalized, diffusion_step):
        """
        Forward pass of the VBD model.

        Args:
            inputs: Input data.
            noised_actions: noised actions.
            diffusion_step: Diffusion step.

        Returns:
            output_dict: Dictionary containing the model outputs.
        """
        # Encode scene
        output_dict = {}
        encoder_outputs = self.encoder(inputs)
        
        if self._train_denoiser:
            denoiser_outputs = self.forward_denoiser(encoder_outputs, noised_actions_normalized, diffusion_step)
            output_dict.update(denoiser_outputs)
            
        if self._train_predictor:
            predictor_outputs = self.forward_predictor(encoder_outputs)
            output_dict.update(predictor_outputs)
            
        return output_dict

    def forward_denoiser(self, encoder_outputs, noised_actions_normalized, diffusion_step):
        """
        Forward pass of the denoiser module.

        Args:
            encoder_outputs: Outputs from the encoder module.
            noised_actions: noised actions.
            diffusion_step: Diffusion step.

        Returns:
            denoiser_outputs: Dictionary containing the denoiser outputs.
        """
        noised_actions = self.unnormalize_actions(noised_actions_normalized)
        denoiser_output = self.denoiser(encoder_outputs, noised_actions, diffusion_step)
        denoised_actions_normalized = self.noise_scheduler.q_x0(
            denoiser_output, 
            diffusion_step, 
            noised_actions_normalized,
            prediction_type=self._prediction_type
        )
        current_states = encoder_outputs['agents'][:, :self._agents_len, -1]
        assert encoder_outputs['agents'].shape[1] >= self._agents_len, 'Too many agents to consider'
        
        # Roll out
        denoised_actions = self.unnormalize_actions(denoised_actions_normalized)
        denoised_trajs = roll_out(current_states, denoised_actions,
                    action_len=self.denoiser._action_len, global_frame=True)
        
        return {
            'denoiser_output': denoiser_output,
            'denoised_actions_normalized': denoised_actions_normalized,
            'denoised_actions': denoised_actions,
            'denoised_trajs': denoised_trajs,
        }

    def forward_predictor(self, encoder_outputs):
        """
        Forward pass of the predictor module.

        Args:
            encoder_outputs: Outputs from the encoder module.

        Returns:
            predictor_outputs: Dictionary containing the predictor outputs.
        """
        # Predict goal
        goal_actions_normalized, goal_scores = self.predictor(encoder_outputs)
        
        current_states = encoder_outputs['agents'][:, :self._agents_len, -1]
        assert encoder_outputs['agents'].shape[1] >= self._agents_len, 'Too many agents to consider'

        # Roll out
        goal_actions = self.unnormalize_actions(goal_actions_normalized)
        goal_trajs = roll_out(current_states[:, :, None, :], goal_actions,
                    action_len=self.predictor._action_len, global_frame=True)
        
        return {
            'goal_actions_normalized': goal_actions_normalized,
            'goal_actions': goal_actions,
            'goal_scores': goal_scores,
            'goal_trajs': goal_trajs,
        }

    def forward_and_get_loss(self, batch, prefix = '', debug = False):
        """
        Forward pass of the model and compute the loss.

        Args:
            batch: Input batch.
            prefix: Prefix for the loss keys.
            debug: Flag to enable debug mode.

        Returns:
            total_loss: Total loss.
            log_dict: Dictionary containing the loss values.
            debug_outputs: Dictionary containing debug outputs.
        """
        # data inputs
        self._agents_len = 64  # TODO: delete this after re-training
        agents_future = batch['agents_future'][:, :self._agents_len]  # [B, self._agents_len, self._future_len+1, 9]

        agents_future_valid = batch['agents_future_valid'][:, :self._agents_len]
        # agents_future_valid = torch.ne(agents_future.sum(-1), 0)
        
        agents_interested = batch['agents_interested'][:, :self._agents_len]
        anchors = batch['anchors'][:, :self._agents_len]

        # get actions from trajectory
        gt_actions, gt_actions_valid = inverse_kinematics(
            agents_future,
            agents_future_valid,
            dt=0.1,
            action_len=self._action_len,
        )
        
        gt_actions_normalized = self.normalize_actions(gt_actions)
        B, A, T, D = gt_actions_normalized.shape
        
        log_dict = {}
        debug_outputs = {}
        total_loss = 0
        
        ############## Run Encoder ##############
        encoder_outputs = self.encoder(batch)
        history = encoder_outputs['agents']
        
        # are_agents_zero1 = (history[:, :self._agents_len, :, :] == 0).all(dim=(2, 3)).squeeze(0)
        # are_agents_zero2 = (history[:, self._agents_len:, :, :] == 0).all(dim=(2, 3)).squeeze(0)
        # print(are_agents_zero1, are_agents_zero2)  # [True, False, ...]        

        ############### Denoise #################
        if self._train_denoiser:
            diffusion_steps = torch.randint(
                0, self.noise_scheduler.num_steps, (B,),
                device=agents_future.device
            ).long().unsqueeze(-1).repeat(1, A).view(B, A, 1, 1)  # [B, A, 1, 1]
            
            # sample noise 
            # noise = torch.randn(B*A, T, D).type_as(agents_future)
            noise = torch.randn(B, A, T, D).type_as(agents_future)
            
            # noise the input
            noised_action_normalized = self.noise_scheduler.add_noise(
                gt_actions_normalized, #.reshape(B*A, T, D),
                noise,
                diffusion_steps#, .reshape(B*A),
            )#.reshape(B, A, T, D)
            # noise = noise.reshape(B, A, T, D)

            if self._replay_buffer:
                with torch.no_grad():
                    # Forward for one step
                    denoise_outputs = self.forward_denoiser(
                        encoder_outputs, 
                        gt_actions_normalized, 
                        diffusion_steps.view(B,A)
                    )

                    x_0 = denoise_outputs['denoised_actions_normalized']

                    # Step to sample from P(x_t-1 | x_t, x_0)
                    x_t_prev = self.noise_scheduler.step(
                        model_output = x_0,
                        timesteps = diffusion_steps,
                        sample = noised_action_normalized,
                        prediction_type=self._prediction_type if hasattr(self, '_prediction_type') else 'sample',
                    )
                    noised_action_normalized = x_t_prev.detach()
            
            denoise_outputs = self.forward_denoiser(
                encoder_outputs, 
                noised_action_normalized, 
                diffusion_steps.view(B,A)
            )
            
            debug_outputs.update(denoise_outputs)
            debug_outputs['noise'] = noise
            debug_outputs['diffusion_steps'] = diffusion_steps

            # Get Loss
            denoised_trajs = denoise_outputs['denoised_trajs']
            if self._prediction_type == 'sample':
                state_loss_mean, yaw_loss_mean = self.denoise_loss(
                    denoised_trajs,
                    agents_future, agents_future_valid,
                    agents_interested,
                )
                denoise_loss = state_loss_mean + yaw_loss_mean 
                total_loss += denoise_loss
                
                # Predict the noise
                _, diffusion_loss = self.noise_scheduler.get_noise(
                    x_0 = denoise_outputs['denoised_actions_normalized'],
                    x_t = noised_action_normalized,
                    timesteps=diffusion_steps,
                    gt_noise=noise,
                )

                log_dict.update({
                    prefix+'state_loss': state_loss_mean.item(),
                    prefix+'yaw_loss': yaw_loss_mean.item(),
                    prefix+'diffusion_loss': diffusion_loss.item()
                })

            elif self._prediction_type == 'error':
                denoiser_output = denoise_outputs['denoiser_output']
                denoise_loss = torch.nn.functional.mse_loss(
                    denoiser_output, noise, reduction='mean'
                )
                total_loss += denoise_loss
                log_dict.update({
                    prefix+'diffusion_loss': denoise_loss.item(),
                })

            elif self._prediction_type == 'mean':
                pred_action_normalized = denoise_outputs['denoised_actions_normalized']
                denoise_loss = self.action_loss(
                    pred_action_normalized, gt_actions_normalized, gt_actions_valid, agents_interested
                )
                total_loss += denoise_loss
                log_dict.update({
                    prefix+'action_loss': denoise_loss.item(),
                })
            else:
                raise ValueError('Invalid prediction type')
                
            denoise_ade, denoise_fde = self.calculate_metrics_denoise(
                denoised_trajs, agents_future, agents_future_valid, agents_interested, 8
            )
            
            log_dict.update({
                prefix+'denoise_ADE': denoise_ade,
                prefix+'denoise_FDE': denoise_fde,
            })
        
        ############### Behavior Prior Prediction #################
        if self._train_predictor:
            goal_outputs = self.forward_predictor(encoder_outputs)
            debug_outputs.update(goal_outputs)

            # get loss
            goal_scores = goal_outputs['goal_scores']
            goal_trajs = goal_outputs['goal_trajs']
            
            goal_loss_mean, score_loss_mean = self.goal_loss(
                goal_trajs, goal_scores, agents_future,
                agents_future_valid, anchors,
                agents_interested,
            )

            pred_loss = goal_loss_mean + 0.05 * score_loss_mean
            total_loss += 1.0 * pred_loss
            
            pred_ade, pred_fde = self.calculate_metrics_predict(
                goal_trajs, agents_future, agents_future_valid, agents_interested, 8
            )
            
            log_dict.update({
                prefix+'goal_loss': goal_loss_mean.item(),
                prefix+'score_loss': score_loss_mean.item(),
                prefix+'pred_ADE': pred_ade,
                prefix+'pred_FDE': pred_fde,
            })
        
        log_dict[prefix+'loss'] = total_loss.item()
        
        if debug:
            return total_loss, log_dict, debug_outputs
        else:
            return total_loss, log_dict

    def step_denoiser(self, x_t: torch.Tensor, c: dict, t: int):
        """
        Perform a denoising step to sample x_{t-1} ~ P[x_{t-1} | x_t, D(x_t, c, t)].
        
        Args:
            x_t (torch.Tensor): The input tensor representing the current state. Shape: (num_batch, num_agent, num_action, action_dim)
            c (dict): The conditional variable dictionary.
            t (int): The number of diffusion steps.
            
        Returns:
            denoiser_output (dict): The denoiser outputs.
            x_t_prev (torch.Tensor): The tensor representing the previous noised action. Shape: (num_batch, num_agent, num_action, action_dim)
        """
        
        if self.denoiser is None:
            raise RuntimeError("Denoiser is not defined")
        
        # Denoise to reconstruct x_0 ~ D(x_t, c, t)
        denoiser_output = self.forward_denoiser(
            encoder_outputs=c,
            noised_actions_normalized=x_t,
            diffusion_step=t,
        )
        
        x_0 = denoiser_output['denoised_actions_normalized']
        
        # Step to sample from P(x_t-1 | x_t, x_0)
        x_t_prev = self.noise_scheduler.step(
            model_output = x_0,
            timesteps = t,
            sample = x_t,
            prediction_type=self._prediction_type if hasattr(self, '_prediction_type') else 'sample',
        )

        return denoiser_output, x_t_prev

    def sample_denoiser(
            self, 
            batch, 
            num_samples=1, 
            x_t = None, 
            fix_t: int = -1, 
            calc_loss: bool = False, 
            **kwargs
        ):
        """
        Perform denoising inference on the given batch of data.

        Args:
            batch (dict): The input batch of data.
            guidance_func (callable, optional): A callable function that provides guidance for denoising. Defaults to None.
            early_stop (int, optional): The index of the step at which denoising should stop. Defaults to 0.
            skip (int, optional): The number of steps to skip between denoising iterations. Defaults to 1.
            **kwargs: Additional keyword arguments for guidance.
        Returns:
            dict: The denoising outputs, including the history of noised action normalization. States are global frame.

        """
        # Encode the scene
        batch = self.batch_to_device(batch, self.device)
        
        encoder_outputs = self.encoder(batch)
        
        if num_samples > 1:
            encoder_outputs = duplicate_batch(encoder_outputs, num_samples)
        
        agents_history = encoder_outputs['agents']
        num_batch, num_agent = agents_history.shape[:2]
        num_step = self._future_len//self._action_len
        action_dim = 2
        
        diffusion_steps = list(
            reversed(
                range(
                    # self.early_stop, 
                    0,
                    self.noise_scheduler.num_steps, 
                    # self.skip
                    1,
                )
            )
        )
        
        # History
        x_t_history = []
        denoiser_output_history = []
        # guide_history = []
        
        # Inital X_T
        if x_t is None:
            x_t = torch.randn(num_batch, num_agent, num_step, action_dim, device=self.device)
        else:
            x_t = x_t.to(self.device)

        for t in diffusion_steps:
            x_t_history.append(x_t.detach().cpu().numpy())

            denoiser_outputs, x_t = self.step_denoiser(
                x_t = x_t, 
                c = encoder_outputs, 
                t = fix_t if fix_t >= 0 else t,
            )
            
            denoiser_output_history.append(torch_dict_to_numpy(denoiser_outputs))
        
        denoiser_outputs['history'] = {
            'x_t_history': np.stack(x_t_history, axis=0),
            'denoiser_output_history': stack_dict(denoiser_output_history),
            # 'guide_history': stack_dict(guide_history),
        }
        
        return denoiser_outputs

    def training_step(self, batch, batch_idx):
        """
        Training step of the model.

        Args:
            batch: Input batch.
            batch_idx: Batch index.

        Returns:
            loss: Loss value.
        """
        loss, log_dict = self.forward_and_get_loss(batch, prefix='train/')
        self.log_dict(
            log_dict, 
            on_step=True, on_epoch=False, sync_dist=True,
            prog_bar=True
        )
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """
        Validation step of the model.

        Args:
            batch: Input batch.
            batch_idx: Batch index.
        """
        if self._val_open_loop:
            loss, log_dict = self.forward_and_get_loss(batch, prefix='val/')
            self.log_dict(log_dict, 
                        on_step=False, on_epoch=True, sync_dist=True,
                        prog_bar=True)

        if self._val_closed_loop:
            step_len = self._step_len
            future_len = self._future_len
            pred_traj, pred_z, pred_head = [], [], []
            for _ in range(self._n_rollout_closed_val):
                traj = []
                # z = []
                head = []
                for t in range((future_len+step_len-1) // step_len):
                    print('closed-loop step', t)
                    if t == 0:
                        denoiser_outputs = self.sample_denoiser(batch)
                        batch_ = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                    else:
                        denoiser_outputs = self.sample_denoiser(batch_)
                    
                    # global frame
                    pred_trajs = denoiser_outputs['denoised_trajs'][:, :, :step_len, :]  # [B, A, step_len, 5]
                    
                    if t == 1:
                        pred_trajs_mid_step = denoiser_outputs['denoised_trajs']
                        input_history = batch_['agents_history']

                    batch_['agents_history'] = torch.cat(
                        [batch_['agents_history'][:,:,-1:,:5], pred_trajs], dim=2
                    )  # global frame
                    batch_['agents_history'] = torch.cat(
                        [batch_['agents_history'], batch['agents_history'][:, : , :, -4:]], dim=-1
                    )

                    batch_['relations'] = batch_calculate_relations(
                        batch_['agents_history'], # global frame
                        batch_['polylines'], # global frame
                        batch_['traffic_light_points'], # global frame
                        device=self.device
                    )

                    # batch_ = self.batch_to_device(batch_, self.device)

                    traj.append(pred_trajs[:, :, :, :2])
                    head.append(pred_trajs[:, :, :, 2])
                    # z.append(batch['agents_history'][:, :, -1, 8])
                pred_traj.append(torch.cat(traj, dim=-2))
                # pred_z.append(z)
                pred_head.append(torch.cat(head, dim=-1))

            pred_traj = torch.stack(pred_traj, dim=1)
            # pred_z = torch.stack(pred_z, dim=0)
            pred_z = batch['agents_history'][:, :, -1, 8]
            pred_z = pred_z.unsqueeze(-1).repeat(1, 1, future_len)
            pred_z = pred_z.unsqueeze(1).repeat(1, self._n_rollout_closed_val, 1, 1)
            pred_head = torch.stack(pred_head, dim=1)

            simulated_states = torch.cat(
                [pred_traj, pred_z.unsqueeze(-1), pred_head.unsqueeze(-1)], dim=-1
            )  # [B, num_scenarios, A, future_len, 4]
            print(simulated_states.shape)
            
            import pickle
            import os
            log_data = {
                'agents_interested': batch_['agents_interested'].cpu().detach().numpy(),
                'simulated_states': simulated_states.cpu().detach().numpy(),
                'agents_future': batch_['agents_future'].cpu().detach().numpy() if hasattr(batch['agents_future'], 'cpu') else batch['agents_future'],
                'polylines': batch_['polylines'].cpu().detach().numpy() if hasattr(batch['polylines'], 'cpu') else batch['polylines'],
                'pred_trajs_mid_step': pred_trajs_mid_step.cpu().detach().numpy(),
                'input_history': input_history.cpu().detach().numpy(),
            }
            
            # 确保日志目录存在
            log_dir = '/home/zhanghailiang/Repos/catk/logs'
            os.makedirs(log_dir, exist_ok=True)
            
            log_file_path = os.path.join(log_dir, 'test.pkl')
            with open(log_file_path, 'wb') as f:
                pickle.dump(log_data, f)
            
            if isinstance(self.wosac_metrics, WOSACMetric):
                self.wosac_metrics.update(
                scenario_id=batch["scenario_id"],
                gt_scenarios=batch["gt_scenario"],
                agent_id=batch["agent"]["id"],
                agent_batch=batch["agent"]["batch"],
                simulated_states=simulated_states,
            )
            else:
                scenario_rollouts = get_scenario_rollouts(
                    scenario_id=get_scenario_id_int_tensor(
                        batch["scenario_id"], self.device
                    ),
                    agent_id=batch["agents_id"],
                    simulated_states=simulated_states,
                )
                self.wosac_metrics.update(batch["tfrecord_path"], scenario_rollouts)
        
        return loss

    def test_step(self, batch, batch_idx):
        pass

    def on_validation_epoch_end(self):
        if self._val_closed_loop:
            epoch_wosac_metrics = self.wosac_metrics.compute()
            # epoch_wosac_metrics["val_closed/ADE"] = self.minADE.compute()
            if self.global_rank == 0:
                epoch_wosac_metrics["epoch"] = (
                    self.log_epoch if self.log_epoch >= 0 else self.current_epoch
                )
                self.logger.log_metrics(epoch_wosac_metrics)

            self.wosac_metrics.reset()
            # self.minADE.reset()

            if self.global_rank == 0:
                if self.wosac_submission.is_active:
                    self.wosac_submission.save_sub_file()

    def on_test_epoch_end(self):
        pass

    ################### Loss function ###################
    def denoise_loss(
            self, 
            denoised_trajs,
            agents_future, 
            agents_future_valid,
            agents_interested
        ):
            """
            Calculates the denoise loss for the denoised actions and trajectories.

            Args:
                denoised_actions_normalized (torch.Tensor): Normalized denoised actions tensor of shape [B, A, T, C].
                denoised_trajs (torch.Tensor): Denoised trajectories tensor of shape [B, A, T, C].
                agents_future (torch.Tensor): Future agent positions tensor of shape [B, A, T, 3].
                agents_future_valid (torch.Tensor): Future agent validity tensor of shape [B, A, T].
                gt_actions_normalized (torch.Tensor): Normalized ground truth actions tensor of shape [B, A, T, C].
                gt_actions_valid (torch.Tensor): Ground truth actions validity tensor of shape [B, A, T].
                agents_interested (torch.Tensor): Interested agents tensor of shape [B, A].

            Returns:
                state_loss_mean (torch.Tensor): Mean state loss.
                yaw_loss_mean (torch.Tensor): Mean yaw loss.
                action_loss_mean (torch.Tensor): Mean action loss.
            """
            
            agents_future = agents_future[..., 1:, :3]
            future_mask = agents_future_valid[..., 1:] * (agents_interested[..., None] > 0)

            # Calculate State Loss
            # [B, A, T]
            state_loss = smooth_l1_loss(denoised_trajs[..., :2], agents_future[..., :2], reduction='none').sum(-1)
            yaw_error = (denoised_trajs[..., 2] - agents_future[..., 2])
            yaw_error = torch.atan2(torch.sin(yaw_error), torch.cos(yaw_error))
            yaw_loss = torch.abs(yaw_error)
            
            # Filter out the invalid state
            state_loss = state_loss * future_mask
            yaw_loss = yaw_loss * future_mask
            
            # Calculate the mean loss
            state_loss_mean = state_loss.sum() / future_mask.sum()
            yaw_loss_mean = yaw_loss.sum() / future_mask.sum()
            
            return state_loss_mean, yaw_loss_mean

    def action_loss(
        self, actions, actions_gt, actions_valid, agents_interested
    ):
        """
        Calculates the loss for action prediction.

        Args:
            actions (torch.Tensor): Tensor of shape [B, A, T, 2] representing predicted actions.
            actions_gt (torch.Tensor): Tensor of shape [B, A, T, 2] representing ground truth actions.
            actions_valid (torch.Tensor): Tensor of shape [B, A, T] representing validity of actions.
            agents_interested (torch.Tensor): Tensor of shape [B, A] representing interest in agents.

        Returns:
            action_loss_mean (torch.Tensor): Mean action loss.
        """
        # Get Mask
        action_mask = actions_valid * (agents_interested[..., None] > 0)
        
        # Calculate the action loss
        action_loss = smooth_l1_loss(actions, actions_gt, reduction='none').sum(-1)
        action_loss = action_loss * action_mask
        
        # Calculate the mean loss
        action_loss_mean = action_loss.sum() / action_mask.sum()
        
        return action_loss_mean

    def goal_loss(
        self, trajs, scores, agents_future,
        agents_future_valid, anchors,
        agents_interested
    ):
        """
        Calculates the loss for trajectory prediction.

        Args:
            trajs (torch.Tensor): Tensor of shape [B*A, Q, T, 3] representing predicted trajectories.
            scores (torch.Tensor): Tensor of shape [B*A, Q] representing predicted scores.
            agents_future (torch.Tensor): Tensor of shape [B, A, T, 3] representing future agent states.
            agents_future_valid (torch.Tensor): Tensor of shape [B, A, T] representing validity of future agent states.
            anchors (torch.Tensor): Tensor of shape [B, A, Q, 2] representing anchor points.
            agents_interested (torch.Tensor): Tensor of shape [B, A] representing interest in agents.

        Returns:
            traj_loss_mean (torch.Tensor): Mean trajectory loss.
            score_loss_mean (torch.Tensor): Mean score loss.
        """
        # Convert Anchor to Global Frame
        current_states = agents_future[:, :, 0, :3] 
        anchors_global = batch_transform_trajs_to_global_frame(anchors, current_states)
        num_batch, num_agents, num_query, _ = anchors_global.shape
        
        # Get Mask
        traj_mask = agents_future_valid[..., 1:] * (agents_interested[..., None] > 0) # [B, A, T]
        
        # Flatten batch and agents
        goal_gt = agents_future[:, :, -1:, :2].flatten(0, 1) # [B*A, 1, 2]
        trajs_gt = agents_future[:, :, 1:, :3].flatten(0, 1) # [B*A, T, 3]
        trajs = trajs.flatten(0, 1)[..., :3] # [B*A, Q, T, 3]
        anchors_global = anchors_global.flatten(0, 1) # [B*A, Q, 2]
        
        # Find the closest anchor
        idx_anchor = torch.argmin(torch.norm(anchors_global - goal_gt, dim=-1), dim=-1) # [B*A,]
        
        # For agents that do not have valid end point, use the minADE
        dist = torch.norm(trajs[:, :, :, :2] - trajs_gt[:, None, :, :2], dim=-1) # [B*A, Q, T]
        dist = dist * traj_mask.flatten(0, 1)[:, None, :] # [B*A, Q, T]
        idx = torch.argmin(dist.mean(-1), dim=-1) # [B*A,]

        # Select trajectory
        idx = torch.where(agents_future_valid[..., -1].flatten(0, 1), idx_anchor, idx)
        trajs_select = trajs[torch.arange(num_batch*num_agents), idx] # [B*A, T, 3]
        
        # Calculate the trajectory loss
        traj_loss = smooth_l1_loss(trajs_select, trajs_gt, reduction='none').sum(-1) # [B*A, T]
        traj_loss = traj_loss * traj_mask.flatten(0, 1) # [B*A, T]
        
        # Calculate the score loss
        scores = scores.flatten(0, 1) # [B*A, Q]
        score_loss = cross_entropy(scores, idx, reduction='none') # [B*A]
        score_loss = score_loss * (agents_interested.flatten(0, 1) > 0) # [B*A]
        
        # Calculate the mean loss
        traj_loss_mean = traj_loss.sum() / traj_mask.sum()
        score_loss_mean = score_loss.sum() / (agents_interested > 0).sum()

        return traj_loss_mean, score_loss_mean

    @torch.no_grad()
    def calculate_metrics_denoise(self, 
            denoised_trajs, agents_future, agents_future_valid,
            agents_interested, top_k = None
        ):
            """
            Calculates the denoising metrics for the predicted trajectories.

            Args:
                denoised_trajs (torch.Tensor): Denoised trajectories of shape [B, A, T, 2].
                agents_future (torch.Tensor): Ground truth future trajectories of agents of shape [B, A, T, 2].
                agents_future_valid (torch.Tensor): Validity mask for future trajectories of agents of shape [B, A, T].
                agents_interested (torch.Tensor): Interest mask for agents of shape [B, A].
                top_k (int, optional): Number of top agents to consider. Defaults to None.

            Returns:
                Tuple[float, float]: A tuple containing the denoising ADE (Average Displacement Error) and FDE (Final Displacement Error).
            """
            
            if not top_k:
                top_k = self._agents_len  
            
            pred_traj = denoised_trajs[:, :top_k, :, :2] # [B, A, T, 2]
            gt = agents_future[:, :top_k, 1:, :2] # [B, A, T, 2]
            gt_mask = (agents_future_valid[:, :top_k, 1:] \
                & (agents_interested[:, :top_k, None] > 0)).bool() # [B, A, T] 

            denoise_mse = torch.norm(pred_traj - gt, dim = -1)
            denoise_ADE = denoise_mse[gt_mask].mean()
            denoise_FDE = denoise_mse[...,-1][gt_mask[...,-1]].mean()
            
            return denoise_ADE.item(), denoise_FDE.item()

    @torch.no_grad()
    def calculate_metrics_predict(self,
            goal_trajs, agents_future, agents_future_valid,
            agents_interested, top_k = None
        ):
            """
            Calculates the metrics for predicting goal trajectories.

            Args:
                goal_trajs (torch.Tensor): Tensor of shape [B, A, Q, T, 2] representing the goal trajectories.
                agents_future (torch.Tensor): Tensor of shape [B, A, T, 2] representing the future trajectories of agents.
                agents_future_valid (torch.Tensor): Tensor of shape [B, A, T] representing the validity of future trajectories.
                agents_interested (torch.Tensor): Tensor of shape [B, A] representing the interest level of agents.
                top_k (int, optional): The number of top agents to consider. Defaults to None.

            Returns:
                tuple: A tuple containing the goal Average Displacement Error (ADE) and goal Final Displacement Error (FDE).
            """
            
            if not top_k:
                top_k = self._agents_len
            goal_trajs = goal_trajs[:, :top_k, :, :, :2] # [B, A, Q, T, 2]
            gt = agents_future[:, :top_k, 1:, :2] # [B, A, T, 2]
            gt_mask = (agents_future_valid[:, :top_k, 1:] \
                & (agents_interested[:, :top_k, None] > 0)).bool() # [B, A, T] 
                   
            goal_mse = torch.norm(goal_trajs - gt[:, :, None, :, :], dim = -1) # [B, A, Q, T]
            goal_mse = goal_mse * gt_mask[..., None, :] # [B, A, Q, T]
            best_idx = torch.argmin(goal_mse.sum(-1), dim = -1) 
            
            best_goal_mse = goal_mse[torch.arange(goal_mse.shape[0])[:, None],
                                     torch.arange(goal_mse.shape[1])[None, :],
                                     best_idx]
            
            goal_ADE = best_goal_mse.sum() / gt_mask.sum()
            goal_FDE = best_goal_mse[..., -1].sum()/gt_mask[..., -1].sum()
            
            return goal_ADE.item(), goal_FDE.item()

    ################### Helper Functions ##############
    def batch_to_device(self, input_dict: dict, device: torch.device = 'cuda'):
        """
        Move the tensors in the input dictionary to the specified device.

        Args:
            input_dict (dict): A dictionary containing tensors to be moved.
            device (torch.device): The target device to move the tensors to.

        Returns:
            dict: The input dictionary with tensors moved to the specified device.
        """
        for key, value in input_dict.items():
            if isinstance(value, torch.Tensor):
                input_dict[key] = value.to(device)

        return input_dict

    def normalize_actions(self, actions: torch.Tensor):
        """
        Normalize the given actions using the mean and standard deviation.

        Args:
            actions : The actions to be normalized.

        Returns:
            The normalized actions.
        """
        return (actions - self.action_mean) / self.action_std

    def unnormalize_actions(self, actions: torch.Tensor):
        """
        Unnormalize the given actions using the stored action standard deviation and mean.

        Args:
            actions: The normalized actions to be unnormalized.

        Returns:
             The unnormalized actions.
        """
        return actions * self.action_std + self.action_mean
