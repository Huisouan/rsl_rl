#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

"""Implementation of different RL agents."""

from .pmcppo import PMCPPO
from .epmcppo import EPMCPPO
from .aseppo import ASEPPO
from .aseppov1 import ASEV1
from ..modules.amp_discriminator import AMPDiscriminator
from ..modules.pairwise_amp_discriminator import PAMPDiscriminator
from .amp_ppo import AMPPPO
from .cvqvaeppo import CVQVAEPPO
from .ppo import PPO
from .him_ppo import HIMPPO
from .pairwise_amp_ppo import PAMPPPO
__all__ = [ "PPO",
            "PMCPPO", 
            "EPMCPPO",
            "ASEPPO",
            "ASEV1",
            "AMPDiscriminator",
            "PAMPDiscriminator",
            "AMPPPO",
            "PAMPPPO",
            "CVQVAEPPO",
            "HIMPPO",
           ]
