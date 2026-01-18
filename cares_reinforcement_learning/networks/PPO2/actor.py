from torch import nn
import torch
from cares_reinforcement_learning.networks.common import BasePolicy, TanhGaussianPolicy
from cares_reinforcement_learning.util.configurations import PPO2Config


# class BaseActor(nn.Module):
#     def __init__(self, act_net: nn.Module, num_actions: int):
#         super().__init__()

#         self.num_actions = num_actions
#         self.act_net = act_net

#     def forward(self, state: torch.Tensor) -> torch.Tensor:
#         output = self.act_net(state)
#         return output

class DefaultActor(TanhGaussianPolicy):
    # DiagGaussianActor
    """torch.distributions implementation of an diagonal Gaussian policy."""

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

        # pylint: disable-next=non-parent-init-called
        BasePolicy.__init__(self, observation_size, num_actions)

        self.act_net = nn.Sequential(
            nn.Linear(observation_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
        )

        self.mean_linear = nn.Linear(hidden_sizes[-1], num_actions)
        self.log_std_linear = nn.Linear(hidden_sizes[-1], num_actions)

# class DefaultActor(BaseActor):
#     def __init__(self, observation_size: int, num_actions: int):
#         hidden_sizes = [1024, 1024]

#         act_net = nn.Sequential(
#             nn.Linear(observation_size, hidden_sizes[0]),
#             nn.ReLU(),
#             nn.Linear(hidden_sizes[0], hidden_sizes[1]),
#             nn.ReLU(),
#             nn.Linear(hidden_sizes[1], num_actions),
#             nn.Tanh(),
#         )

#         super().__init__(act_net=act_net, num_actions=num_actions)


class Actor(TanhGaussianPolicy):
    # DiagGaussianActor
    """torch.distributions implementation of an diagonal Gaussian policy."""

    def __init__(self, observation_size: int, num_actions: int, config: PPO2Config):

        log_std_bounds = config.log_std_bounds

        super().__init__(
            input_size=observation_size,
            num_actions=num_actions,
            log_std_bounds=log_std_bounds,
            config=config.actor_config,
        )


# class Actor(BaseActor):
#     def __init__(self, observation_size: int, num_actions: int, config: PPO2Config):

#         act_net = MLP(
#             input_size=observation_size,
#             output_size=num_actions,
#             config=config.actor_config,
#         )

#         super().__init__(act_net=act_net, num_actions=num_actions)
