import numpy as np
from torch import nn
import torch
from torch.distributions import Normal
from cares_reinforcement_learning.networks.common import MLP, BasePolicy
from cares_reinforcement_learning.util.configurations import PPO2Config

class DefaultActor(BasePolicy):

    # pylint: disable=super-init-not-called
    def __init__(
        self,
        observation_size: int,
        num_actions: int,
        hidden_sizes: list[int] | None = None,
        log_std_bounds: list[float] | None = None,
    ):
        if hidden_sizes is None:
            hidden_sizes = [1024, 1024]

        if log_std_bounds is None:
            log_std_bounds = [-20.0, 2.0]
        self.log_std_bounds = log_std_bounds

        # pylint: disable-next=non-parent-init-called
        BasePolicy.__init__(self, observation_size, num_actions)

        self.act_net = nn.Sequential(
            nn.Linear(observation_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], num_actions),
            nn.Tanh(),
        )

        # init log std param as -0.5. This is NOT dependent on state like log_std in sac's actor
        log_std = -0.5 * np.ones(num_actions, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))

    def get_distribution(self, state:torch.Tensor):
        mu = self.act_net(state)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(self.log_std)
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)
        std = log_std.exp()

        dist = Normal(mu,std)
        return dist

    def forward(
        self, state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        dist = self.get_distribution(state)
        sample = dist.rsample()
        log_pi = dist.log_prob(sample).sum(-1, keepdim=True)

        return sample, dist.loc, log_pi


class Actor(BasePolicy):
    '''Output: (sampled value, log prob of sample, distribution mean)'''

    def __init__(self, observation_size: int, num_actions: int, config: PPO2Config):

        super().__init__(
            input_size=observation_size,
            num_actions=num_actions,
        )

        # network for calculating mean
        self.act_net : MLP | nn.Sequential = MLP(
            input_size=observation_size,
            output_size=num_actions,
            config=config.actor_config,
        )
  
        # init log std param as -0.35. (about 0.7 as std)
        # This is NOT dependent on state like log_std in sac's actor
        log_std = -0.35 * np.ones(num_actions, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))

        # bound log std for numerical stability
        self.log_std_bounds = config.log_std_bounds

    def get_distribution(self, state:torch.Tensor):
        mu = self.act_net(state)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std_min, log_std_max = self.log_std_bounds
        log_std = torch.clamp(self.log_std, log_std_min, log_std_max)
        std = log_std.exp()

        dist = Normal(mu,std)
        
        return dist

    def forward(
        self, state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        dist = self.get_distribution(state)
        sample = dist.rsample()
        log_pi = dist.log_prob(sample).sum(-1, keepdim=True)

        return sample, log_pi, dist.loc

