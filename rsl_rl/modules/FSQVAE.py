#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.nn import RMSNorm
from vector_quantize_pytorch import FSQ
from rsl_rl.utils.wrappers import (
    Z_settings,
)


class VQVAEEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dims,
        activation,
    ):
        super().__init__()
        self.activation = activation
        
        encoder_layers = []
        encoder_layers.append(nn.Linear(input_dim, hidden_dims[0]))
        encoder_layers.append(self.activation)
        for layer_index in range(len(hidden_dims) - 1):
            encoder_layers.append(nn.Linear(hidden_dims[layer_index], hidden_dims[layer_index + 1]))
            encoder_layers.append(self.activation)
        encoder_layers.append(nn.Linear(hidden_dims[-1], output_dim))
        
        self.encoder = nn.Sequential(*encoder_layers)
        
    def forward(self, x):
        z_e = self.encoder(x)
        return z_e


class VQVAEDecoder(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dims,
        activation,
        state_dimensions,
        bot_neck_prop_embed_size,
        bot_neck_z_embed_size,
        
    ):
        super().__init__()
        self.activation = activation
        
        observation_embd_layers = []
        observation_embd_layers.append(nn.Linear(state_dimensions, bot_neck_prop_embed_size))
        observation_embd_layers.append(self.activation)
        self.observation_embd = nn.Sequential(*observation_embd_layers)
        
        z_embd_layers = []
        z_embd_layers.append(nn.Linear(input_dim, bot_neck_z_embed_size))
        z_embd_layers.append(self.activation)
        self.z_embd = nn.Sequential(*z_embd_layers)
        
        decoder_layers = []
        decoder_layers.append(nn.Linear(bot_neck_prop_embed_size + bot_neck_z_embed_size, hidden_dims[0]))
        decoder_layers.append(self.activation)
        for layer_index in range(len(hidden_dims) - 1):
            decoder_layers.append(nn.Linear(hidden_dims[layer_index], hidden_dims[layer_index + 1]))
            decoder_layers.append(self.activation)
        decoder_layers.append(nn.Linear(hidden_dims[-1], output_dim))
        
        self.decoder = nn.Sequential(*decoder_layers)
    def forward(self, z_embedded, observations):
        observation_embd = self.observation_embd(observations)
        decoder_input = torch.cat((z_embedded, observation_embd), dim=1)
        mean = self.decoder(decoder_input)
        return mean


class FSQVAE(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        num_dataset,
        encoder_hidden_dims=[512, 256],
        decoder_hidden_dims=[256, 256],
        critic_hidden_dims=[256, 256],
        fsqlevels = [8,6,5],
        z_length = 32,
        activation="elu",
        value_activation="tanh",
        init_noise_std=1,
        One_step_obs=45,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        self.activation = get_activation(activation)
        self.value_activation = get_activation(value_activation)
        
        num_actor_obs = num_actor_obs * 3 + num_dataset
        
        self.input_init = False
        self.One_step_obs = One_step_obs
        self.z_length = z_length

        # VQVAE Encoder
        self.encoder = VQVAEEncoder(
            input_dim=num_actor_obs,
            output_dim=self.z_length,
            hidden_dims=encoder_hidden_dims,
            activation=self.activation,
        )
        self.quantize_t = FSQ(levels=fsqlevels)
        # VQVAE Decoder
        self.decoder = VQVAEDecoder(
            input_dim=self.z_length,
            output_dim=num_actions,
            hidden_dims=decoder_hidden_dims,
            activation=self.activation,
            state_dimensions=self.One_step_obs,
        )
        
        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(num_critic_obs, critic_hidden_dims[0]))
        critic_layers.append(self.value_activation)
        for layer_index in range(len(critic_hidden_dims) - 1):
            critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], critic_hidden_dims[layer_index + 1]))
            critic_layers.append(self.value_activation)
        critic_layers.append(nn.Linear(critic_hidden_dims[-1], 1))
        self.critic = nn.Sequential(*critic_layers)

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        self.z_e = None
        self.z_q = None
        Normal.set_default_validate_args = False
        
        print(f"RMS: {self.rms}")
        print(f"Encoder: {self.encoder}")
        print(f"Decoder: {self.decoder}")
        print(f"Critic: {self.critic}")

    def encode(self,input):
        logits = self.encoder(input)
        quant_t, id_t = self.quantize_t(logits)
        return quant_t, id_t        

    def decode(self,z_embedded, observations):
        mean = self.decoder.forward(self, z_embedded, observations)
        return mean


    @staticmethod
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))
        ]

    def reset(self, dones=None):
        pass

    def forward(self,input,return_id=True):
        quant_t, diff, id_t, = self.encode(input)
        dec = self.decoder(quant_t,input[:,self.One_step_obs])
        if return_id:
            return dec, diff, id_t
        return dec, diff        
        

    @property
    def vector_z_e(self):
        return self.z_e

    @property
    def vector_z_q(self):
        return self.z_q
    
    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        quant_t, id_t, = self.encode(observations)
        
        mean = self.decoder(quant_t,observations[:,self.One_step_obs])        
        self.distribution = Normal(mean, self.std)
        return mean

    def act(self, observations, **kwargs):
        mean = self.update_distribution(observations)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        actions_mean = self.update_distribution(observations)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value


def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.CReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None