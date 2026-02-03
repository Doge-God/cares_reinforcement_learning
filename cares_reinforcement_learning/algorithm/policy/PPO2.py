"""
Original Paper:
                https://arxiv.org/abs/1707.06347
Good Explanation:
                https://www.youtube.com/watch?v=5P7I-xPq8u8
Code based on:
                https://github.com/ericyangyu/PPO-for-Beginners
                https://github.com/nikhilbarhate99/PPO-PyTorch
                https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/rl/ppo/gae.py
                https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/ppo/ppo.py

Comparing to previous implementation in this lib:
                - Uses GAE for advantage estimation instead of monte carlo
                - Standard deviation of action treated as learnable param instead of fixed
                - Added entropy bonus to ensure exploration 
"""

import logging
import os
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

import cares_reinforcement_learning.util.training_utils as tu
from cares_reinforcement_learning.algorithm.algorithm import VectorAlgorithm
from cares_reinforcement_learning.networks.PPO2 import Actor, Critic
from cares_reinforcement_learning.util.configurations import PPO2Config
from cares_reinforcement_learning.util.training_context import (
    TrainingContext,
    ActionContext,
)


class PPO2(VectorAlgorithm):
    def __init__(
        self,
        actor_network: Actor,
        critic_network: Critic,
        config: PPO2Config,
        device: torch.device,
    ):
        super().__init__(policy_type="policy", config=config, device=device)

        self.actor_net = actor_network.to(device)
        self.critic_net = critic_network.to(device)

        self.gamma = config.gamma
        self.action_num = self.actor_net.num_actions
        self.device = device

        # lambda for gae calc
        self.lambda_gae = config.lambda_gae

        # bounds for std
        self.log_std_bound = config.log_std_bounds

        self.actor_net_optimiser = torch.optim.Adam(
            self.actor_net.parameters(), lr=config.actor_lr
        )
        self.critic_net_optimiser = torch.optim.Adam(
            self.critic_net.parameters(), lr=config.critic_lr
        )

        self.updates_per_iteration = config.updates_per_iteration
        self.eps_clip = config.eps_clip

        # coef for c1 and c2 in the original paper
        self.value_loss_coef = config.value_loss_coef
        # entropy bonus to ensure exploration
        self.entropy_bonus_coef = config.entropy_bonus_coef

    def select_action_from_policy(self, action_context: ActionContext) -> np.ndarray:
        self.actor_net.eval()
        state = action_context.state
        evaluation = action_context.evaluation

        assert isinstance(state, np.ndarray)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(self.device)
            state_tensor = state_tensor.unsqueeze(0)
            if evaluation:
                (_, _, action) = self.actor_net(state_tensor) # distribution mean
            else:
                (action, log_prob, mean) = self.actor_net(state_tensor) # sampled
            action = action.cpu().data.numpy().flatten()

        self.actor_net.train()
        return action


    def _calculate_value(self, state: np.ndarray, action: np.ndarray) -> float:  # type: ignore[override]
        #NOTE: abs func in Algorithm super class, not used here for some reason
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)
        state_tensor = state_tensor.unsqueeze(0)

        with torch.no_grad():
            value = self.critic_net(state_tensor)

        return value[0].item()

    def _evaluate_policy(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        '''
        Calculate value of state and log prob of action with current policy
        Output: (value tensor (shape: numStepPerTrain,1), log prob of actions tensor(shape: numStepPerTrain,1))'''

        v = self.critic_net(state)#.squeeze()  # shape: numStepPerTrain, 1 
        #NOTE:5000 is default number_steps_per_train_policy in config

        dist = self.actor_net.get_distribution(state)

        log_prob = dist.log_prob(action).sum(-1, keepdim=True) # shape: numStepPerTrain,1
        # use to check how the entropy bonus is affecting training:
        #print(f"std max: {dist.scale.max()} std min: {dist.scale.min()}")
        # TODO: record entropy in training log

        entropy = dist.entropy().sum(-1,keepdim=True)

        return v, log_prob, entropy
    
    
    def _process_batch(self,
        batch_rewards:torch.Tensor, batch_dones: torch.Tensor, batch_values: torch.Tensor):
        '''Loop through experience batch, calculate advantage (with GAE, for actor), one step td error (for logging) 
        & true reward to go (for critic)

        Expect shape for all INPUT: (num steps per train)

        Output: (batch_advantages, batch_td_errors, batch_reward_to_goes) each shape: (num steps per train, 1)
        '''
        advantages:list[float] = []
        td_errors:list[float] = []
        reward_to_goes: list[float] = []

        last_advantage = 0
        last_value = batch_values[-1]
        discounted_reward = 0

        for reward, done, value in zip(reversed(batch_rewards), reversed(batch_dones), reversed(batch_values)):
            # resetting rolling variables when encountering a done
            is_done_mask = 1-done # 0 if done
            last_value = last_value * is_done_mask
            last_advantage = last_advantage * is_done_mask

            # calc reward to go
            discounted_reward = reward + self.gamma * is_done_mask * discounted_reward
            reward_to_goes.insert(0, discounted_reward)

            # calculate current step td error
            td_error = reward + self.gamma*last_value - value # loop inversed, in actual order: (state,value,reward) -> (last state, last value)

            # GAE advantage def:
            # A^{GAE(\gamma,\lambda)}_{t}=\sum^{\infty}_{l=0} (\gamma \lambda)^l \delta_{t+l}
            # e.g. at step 4: 
            #   last_advantage = td_4 + gam*lam*td_5  
            # so at next iteration at step 3:
            #   last_advantage = td_3 + gam*lam*(td_4 + gam*lam*td_5) <-- last_advantage from previous iteration
            # thus follows prev def
            last_advantage = td_error + self.gamma*self.lambda_gae*last_advantage

            advantages.insert(0,last_advantage)
            td_errors.insert(0, td_error)
            last_value = value


        batch_advantages = torch.tensor(
            advantages, dtype=torch.float32, device=self.device
        ).unsqueeze(-1)

        batch_td_errors = torch.tensor(
            td_errors, dtype=torch.float32, device=self.device
        ).unsqueeze(-1)

        batch_reward_to_goes = torch.tensor(
            reward_to_goes, dtype=torch.float32, device=self.device
        ).unsqueeze(-1)

        return batch_advantages, batch_td_errors, batch_reward_to_goes
        

    def train_policy(self, training_context: TrainingContext) -> dict[str, Any]:

        memory = training_context.memory
        # pylint: disable-next=unused-argument
        batch_size = training_context.batch_size

        experiences = memory.flush()
        states, actions, rewards, next_states, dones = experiences

        # Convert to tensors using helper method (no next_states needed for PPO, so pass dummy data)
        (
            states_tensor,
            actions_tensor,
            rewards_tensor,
            _,  # next_states not used in PPO
            dones_tensor,
            _,  # weights not needed
        ) = tu.batch_to_tensors(
            np.asarray(states),
            np.asarray(actions),
            np.asarray(rewards),
            np.asarray(next_states),
            np.asarray(dones),
            self.device,
        )


        # both shape numStepPerTrain,1
        value_tensor,old_policy_log_prob_tensor,_ = self._evaluate_policy(states_tensor, actions_tensor)

        # this is updating ACTOR: these values should be treated as constant
        advantages_tensor, td_errors_tensor, rtgs_tensor = self._process_batch(rewards_tensor.detach(), dones_tensor.detach(), value_tensor.detach().squeeze())
      
        # Normalize advantages
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-10)
    
        td_errors = torch.abs(td_errors_tensor).data.cpu().numpy()

        for _ in range(self.updates_per_iteration):
            current_values, curr_log_probs, entropy = self._evaluate_policy(states_tensor, actions_tensor)

            # Calculate ratios. 
            ratios = torch.exp(curr_log_probs - old_policy_log_prob_tensor.detach())

            # Finding Surrogate Loss
            surrogate_loss_one = ratios * advantages_tensor
            surrogate_loss_two = (
                torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages_tensor
            )
            min_surrogate_loss = torch.minimum(surrogate_loss_one, surrogate_loss_two)

            # find entropy loss
            entropy_bonus_loss = self.entropy_bonus_coef * entropy

            # final loss of clipped objective PPO
            actor_loss = -(min_surrogate_loss + entropy_bonus_loss).mean()
            critic_loss = self.value_loss_coef * F.mse_loss(current_values, rtgs_tensor)

            # run backward passes
            self.actor_net_optimiser.zero_grad()
            actor_loss.backward(retain_graph=True)
            self.actor_net_optimiser.step()

            self.critic_net_optimiser.zero_grad()
            critic_loss.backward()
            self.critic_net_optimiser.step()

        info: dict[str, Any] = {}
        info["td_errors"] = td_errors
        info["critic_loss"] = critic_loss.item()
        info["actor_loss"] = actor_loss.item()

        return info

    def save_models(self, filepath: str, filename: str) -> None:
        if not os.path.exists(filepath):
            os.makedirs(filepath)

        checkpoint = {
            "actor": self.actor_net.state_dict(),
            "critic": self.critic_net.state_dict(),
            "actor_optimizer": self.actor_net_optimiser.state_dict(),
            "critic_optimizer": self.critic_net_optimiser.state_dict(),
        }
        torch.save(checkpoint, f"{filepath}/{filename}_checkpoint.pth")
        logging.info("models and optimisers have been saved...")

    def load_models(self, filepath: str, filename: str) -> None:
        checkpoint = torch.load(f"{filepath}/{filename}_checkpoint.pth")
        self.actor_net.load_state_dict(checkpoint["actor"])
        self.critic_net.load_state_dict(checkpoint["critic"])

        self.actor_net_optimiser.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_net_optimiser.load_state_dict(checkpoint["critic_optimizer"])
        logging.info("models and optimisers have been loaded...")
