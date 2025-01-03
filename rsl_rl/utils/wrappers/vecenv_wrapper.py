# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import torch

from rsl_rl.env.vec_env import VecEnv
class RslRlVecEnvWrapper(VecEnv):
    def __init__(self, env):
        """Initializes the wrapper.

        Note:
            The wrapper calls :meth:`reset` at the start since the RSL-RL runner does not call reset.

        Args:
            env: The environment to wrap around.

        Raises:
            ValueError: When the environment is not an instance of :class:`ManagerBasedRLEnv` or :class:`DirectRLEnv`.
        """
        # initialize the wrapper
        self.unwrapped = env
        # store information required by wrapper
        self.num_envs = self.unwrapped.num_envs
        self.device = self.unwrapped.device
        self.max_episode_length = self.unwrapped.max_episode_length
        self.num_actions = self.unwrapped.num_actions
        self.num_obs = self.unwrapped.num_obs
        # -- privileged observations
        if self.unwrapped.num_privileged_obs is not None:
            self.num_privileged_obs = self.unwrapped.num_privileged_obs
        # reset at the start since the RSL-RL runner does not call reset
        self.unwrapped.reset()

    def __str__(self):
        """Returns the wrapper name and the :attr:`env` representation string."""
        return f"<{type(self).__name__}{self.env}>"

    def __repr__(self):
        """Returns the string representation of the wrapper."""
        return str(self)

    """
    Properties -- Wrapper
    """
    @classmethod
    def class_name(cls) -> str:
        """Returns the class name of the wrapper."""
        return cls.__name__


    """
    Properties
    """

    def get_observations(self) -> tuple[torch.Tensor, dict]:
        """Returns the current observations of the environment."""
        obs_dict = self.unwrapped.get_observations()
        return obs_dict["policy"], {"observations": obs_dict}

    @property
    def episode_length_buf(self) -> torch.Tensor:
        """The episode length buffer."""
        return self.unwrapped.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor):
        """Set the episode length buffer.

        Note:
            This is needed to perform random initialization of episode lengths in RSL-RL.
        """
        self.unwrapped.episode_length_buf = value

    """
    Operations - MDP
    """

    def reset(self) -> tuple[torch.Tensor, dict]:  # noqa: D102
        # reset the environment
        obs_dict, _ = self.unwrapped.reset()
        # return observations
        return obs_dict["policy"], {"observations": obs_dict}

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        # record step information
        obs_dict, rew, dones, extras = self.unwrapped.step(actions)
        # move extra observations to the extras dict
        obs = obs_dict["policy"]
        extras["observations"] = obs_dict

        # return the step information
        return obs, rew, dones, extras

