#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim

from ..modules import ASEV1
from ..storage import ASERolloutStorage
from ..storage.replay_buffer import ReplayBuffer
from rl_lab.assets.loder_for_algs import AmpMotion
from ..utils.amp_utils import Normalizer

class ASEPPOV1:
    actor_critic: ASEV1

    def __init__(
        self,
        actor_critic,
        num_learning_epochs=1,
        num_mini_batches=1,
        clip_param=0.2,
        gamma=0.998,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.0,
        learning_rate=1e-3,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="fixed",
        desired_kl=0.01,
        device="cpu",
        amp_data:AmpMotion = None,
        min_std=None,
        amp_replay_buffer_size = 100000,
        *args, **kwargs
    ):
        self.device = device

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate

        self.min_std = min_std
        # load amp data
        
        self.amp_storage = ReplayBuffer(amp_data.amp_obs_num, amp_replay_buffer_size, device)
        self.amp_data = amp_data
        self.amp_normalizer = Normalizer(amp_data.amp_obs_num)

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None  # initialized later
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        self.transition = ASERolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        # Hyper params
        self.normalize_value = True

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape):
        
        latent_shape = self.actor_critic.ase_latent_shape
        self.storage = ASERolloutStorage(
            num_envs, num_transitions_per_env, actor_obs_shape, 
            critic_obs_shape, action_shape, latent_shape,self.device
            
        )

    def test_mode(self):
        self.actor_critic.test()

    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, critic_obs,amp_obs,cur_episode_length):
        # Compute the actions and values
        self.actor_critic.update_latents(cur_episode_length)
        
        self.transition.actions = self.actor_critic.act(obs).detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        self.transition.amp_observations = amp_obs
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos,next_amp_obs_with_term):
        """
        处理环境的步进结果。

        参数:
            rewards: 从环境中获得的奖励。
            dones: 标记表示剧集是否结束。
            infos: 来自环境的额外信息，例如超时。
        """
        #amp数据归一化
        norm_amp_obs = self.amp_normalizer.normalize_torch(self.transition.amp_observations,self.device)
        norm_amp_obs_with_term = self.amp_normalizer.normalize_torch(next_amp_obs_with_term,self.device)
        
        amp_obs_trans = torch.cat([norm_amp_obs,norm_amp_obs_with_term],dim = -1)

        disc_r,enc_r = self.actor_critic.calc_amp_rewards(amp_obs_trans)
        #把amp reward加到reward上，此处reward会进入到advantage的计算中，从而影响ppo算法的损失
        self.transition.rewards = self.actor_critic.task_reward_w * rewards \
                                + self.actor_critic.disc_reward_w * disc_r \
                                + self.actor_critic.enc_reward_w * enc_r            
        self.transition.dones = dones
        #ase latent
        self.transition.ase_latent = self.actor_critic.ase_latents.detach()
        #将amp obs存入amp replay buffer
        self.amp_storage.insert(self.transition.amp_observations, next_amp_obs_with_term)
        
        # 超时引导
        if "time_outs" in infos:
            # 如果有超时信息，使用它来调整奖励
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * infos["time_outs"].unsqueeze(1).to(self.device), 1
            )
        
        # 记录转换
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        
        # 重置已完成剧集的演员-评论家网络
        self.actor_critic.reset(dones)

    def compute_returns(self, last_critic_obs):
        last_values = self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self, env):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy_loss = 0
        mean_bound_loss = 0
        mean_disc_loss = 0
        mean_enc_loss = 0
        mean_diversity_loss = 0
        
        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        
        amp_rl_trans_generator = self.amp_storage.feed_forward_generator(
            self.num_learning_epochs * self.num_mini_batches,
            self.storage.num_envs * self.storage.num_transitions_per_env // self.num_mini_batches,
        )
        amp_motion_data_trans_generator = self.amp_data.feed_forward_generator(
            self.num_learning_epochs * self.num_mini_batches,
            self.storage.num_envs * self.storage.num_transitions_per_env // self.num_mini_batches,
        )

        for ((
            obs_batch,
            critic_obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            # ase latent
            ase_latent_batch,
            
            hid_states_batch,
            masks_batch),
            rl_state_trans,
            data_state_trans,
        ) in zip(generator,amp_rl_trans_generator,amp_motion_data_trans_generator):
            self.actor_critic.train_mod = True
            #使用ase_forward
            
            #预处理amp obs
            policy_state, policy_next_state = rl_state_trans
            expert_state, expert_next_state = data_state_trans 
            
            policy_state_unnorm = torch.clone(policy_state)
            expert_state_unnorm = torch.clone(expert_state)

            if self.amp_normalizer is not None:
                with torch.no_grad():
                    policy_state = self.amp_normalizer.normalize_torch(policy_state, self.device)
                    policy_next_state = self.amp_normalizer.normalize_torch(policy_next_state, self.device)
                    expert_state = self.amp_normalizer.normalize_torch(expert_state, self.device)
                    expert_next_state = self.amp_normalizer.normalize_torch(expert_next_state, self.device)
            
            rl_state_trans = torch.cat([policy_state, policy_next_state], dim=-1)
            data_state_trans = torch.cat([expert_state, expert_next_state], dim=-1)     
            data_state_trans.requires_grad_(True)
            
            self.actor_critic.act(obs_batch,ase_latent_batch)
            
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            value_batch = self.actor_critic.evaluate(
                critic_obs_batch, masks=masks_batch, ase_latents = ase_latent_batch,hidden_states=hid_states_batch[1]
            )
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            # KL
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)

                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            bound_loss = self.actor_critic.bound_loss(mu_batch)
            
            disc_agent_logit = self.actor_critic.eval_disc(rl_state_trans)
            disc_demo_logit = self.actor_critic.eval_disc(data_state_trans)            
            # 计算判别器损失
            disc_loss = self.actor_critic.disc_loss(disc_agent_logit,
                                                     disc_demo_logit, 
                                                     data_state_trans)
            
            enc_pred = self.actor_critic.eval_enc(rl_state_trans)
            # 计算编码器损失
            enc_latents = ase_latent_batch
            enc_loss = self.actor_critic.enc_loss(enc_pred, enc_latents, rl_state_trans)
            
            diversity_loss = self.actor_critic.diversity_loss(obs_batch, mu_batch, ase_latent_batch)
            
            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean() \
                + self.actor_critic.bounds_loss_coef * bound_loss.mean() \
                + self.actor_critic.disc_coef * disc_loss + self.actor_critic.enc_coef * enc_loss \
                + self.actor_critic.amp_diversity_bonus * diversity_loss.mean()
 
            # Gradient step
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            self.actor_critic.std.data = self.actor_critic.std.data.clamp(min=self.min_std)
            if self.amp_normalizer is not None:
                self.amp_normalizer.update(policy_state_unnorm.cpu().numpy())
                self.amp_normalizer.update(expert_state_unnorm.cpu().numpy())

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy_loss += entropy_batch.mean().item()
            mean_bound_loss += bound_loss.mean().item()
            mean_disc_loss += disc_loss.item()
            mean_enc_loss += enc_loss.item()
            if self.actor_critic._enable_amp_diversity_bonus():
                mean_diversity_loss += diversity_loss.mean().item()
            else:
                mean_diversity_loss = 0.0

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy_loss /= num_updates
        mean_bound_loss /= num_updates
        mean_disc_loss /= num_updates
        mean_enc_loss /= num_updates
        mean_diversity_loss /= num_updates

        self.storage.clear()
        return (
            mean_value_loss,
            mean_surrogate_loss,
            mean_entropy_loss,
            mean_bound_loss,
            mean_disc_loss,
            mean_enc_loss,
            mean_diversity_loss
        )
