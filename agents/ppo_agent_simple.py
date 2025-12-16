"""
Simplified PPO Agent for educational comparison.
Implements Proximal Policy Optimization with clipped objective.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple
import numpy as np


class SimplePPOAgent:
    """
    PPO agent with actor-critic architecture.
    Uses GAE for advantage estimation and clipped surrogate objective.
    """
    
    def __init__(
        self,
        feature_dim: int,
        action_sizes: Dict[str, int],
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        ppo_epochs: int = 4,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        device: str = "cpu"
    ):
        self.feature_dim = feature_dim
        self.action_sizes = action_sizes
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.ppo_epochs = ppo_epochs
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.device = torch.device(device)
        
        # Actor-Critic network
        self.actor_critic = self._build_actor_critic().to(self.device)
        self.optimizer = torch.optim.Adam(self.actor_critic.parameters(), lr=lr)
        
        # Episode buffer
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        
    def _build_actor_critic(self) -> nn.Module:
        """Build actor-critic network."""
        class ActorCritic(nn.Module):
            def __init__(self, feature_dim, action_sizes):
                super().__init__()
                # Shared feature extractor
                self.shared = nn.Sequential(
                    nn.Linear(feature_dim, 256),
                    nn.ReLU(),
                    nn.Linear(256, 256),
                    nn.ReLU()
                )
                # Actor heads (policy)
                self.actor_heads = nn.ModuleDict({
                    key: nn.Linear(256, size)
                    for key, size in action_sizes.items()
                })
                # Critic head (value function)
                self.critic = nn.Linear(256, 1)
                
            def forward(self, x):
                shared_features = self.shared(x)
                # Actor outputs (logits for each action head)
                actor_logits = {
                    key: head(shared_features)
                    for key, head in self.actor_heads.items()
                }
                # Critic output (state value)
                value = self.critic(shared_features)
                return actor_logits, value
        
        return ActorCritic(self.feature_dim, self.action_sizes)
    
    def select_action(self, state: torch.Tensor, mask: Dict[str, torch.Tensor] = None) -> Tuple[Dict[str, int], torch.Tensor, torch.Tensor]:
        """
        Sample action from policy and compute value.
        
        Returns:
            action: Dict of selected action indices
            log_prob: Log probability of the action
            value: State value estimate
        """
        with torch.no_grad():
            logits_dict, value = self.actor_critic(state)
        
        action = {}
        log_prob_total = 0.0
        
        for key, logits in logits_dict.items():
            # Apply mask if provided
            if mask and key in mask:
                logits = logits.clone()
                logits[~mask[key]] = -1e8
            
            # Sample from categorical distribution
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action_idx = dist.sample()
            log_prob = dist.log_prob(action_idx)
            
            action[key] = action_idx.item()
            log_prob_total += log_prob
        
        return action, log_prob_total, value.squeeze()
    
    def store_transition(self, state: torch.Tensor, action: Dict[str, int], 
                        log_prob: torch.Tensor, reward: float, 
                        value: torch.Tensor, done: bool):
        """Store transition for episode."""
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
    
    def compute_gae(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation (GAE).
        
        Returns:
            advantages: GAE advantages
            returns: Discounted returns
        """
        advantages = []
        gae = 0
        
        values = torch.stack(self.values).to(self.device)
        rewards = torch.tensor(self.rewards, device=self.device)
        dones = torch.tensor(self.dones, device=self.device, dtype=torch.float32)
        
        # Compute GAE backwards
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0.0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        
        advantages = torch.tensor(advantages, device=self.device)
        returns = advantages + values
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
    
    def update(self) -> Tuple[float, float, float]:
        """
        Update policy using PPO algorithm.
        
        Returns:
            policy_loss: Policy loss
            value_loss: Value function loss
            entropy: Policy entropy
        """
        if len(self.rewards) == 0:
            return 0.0, 0.0, 0.0
        
        # Compute advantages and returns
        advantages, returns = self.compute_gae()
        
        # Prepare data
        states = torch.stack(self.states).to(self.device)
        old_log_probs = torch.stack(self.log_probs).to(self.device)
        
        # PPO update for multiple epochs
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        
        for epoch in range(self.ppo_epochs):
            # Forward pass
            logits_dict, values = self.actor_critic(states)
            values = values.squeeze()
            
            # Recompute log probs for current policy
            new_log_probs = []
            entropies = []
            
            for i, action_dict in enumerate(self.actions):
                log_prob_total = 0.0
                entropy_total = 0.0
                
                for key, action_idx in action_dict.items():
                    logits = logits_dict[key][i]
                    probs = F.softmax(logits, dim=-1)
                    dist = torch.distributions.Categorical(probs)
                    log_prob = dist.log_prob(torch.tensor(action_idx, device=self.device))
                    entropy = dist.entropy()
                    
                    log_prob_total += log_prob
                    entropy_total += entropy
                
                new_log_probs.append(log_prob_total)
                entropies.append(entropy_total)
            
            new_log_probs = torch.stack(new_log_probs)
            entropies = torch.stack(entropies)
            
            # Compute policy loss (PPO clipped objective)
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Compute value loss
            value_loss = F.mse_loss(values, returns)
            
            # Compute entropy bonus
            entropy_loss = -entropies.mean()
            
            # Total loss
            loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
            
            # Backprop
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 0.5)
            self.optimizer.step()
            
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += -entropy_loss.item()
        
        # Clear buffer
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        
        return (total_policy_loss / self.ppo_epochs, 
                total_value_loss / self.ppo_epochs,
                total_entropy / self.ppo_epochs)
    
    def save(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'actor_critic': self.actor_critic.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, path)
    
    def load(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor_critic.load_state_dict(checkpoint['actor_critic'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
    
    def get_num_params(self) -> int:
        """Return total number of parameters."""
        return sum(p.numel() for p in self.actor_critic.parameters())
