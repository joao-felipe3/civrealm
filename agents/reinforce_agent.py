"""
REINFORCE (Monte Carlo Policy Gradient) Agent for CivRealm.
Simplified implementation for educational comparison with DQN.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple
import numpy as np


class REINFORCEAgent:
    """
    REINFORCE agent using Monte Carlo policy gradient.
    """
    
    def __init__(
        self,
        feature_dim: int,
        action_sizes: Dict[str, int],
        lr: float = 3e-4,
        gamma: float = 0.99,
        device: str = "cpu"
    ):
        self.feature_dim = feature_dim
        self.action_sizes = action_sizes
        self.gamma = gamma
        self.device = torch.device(device)
        
        # Policy network (stochastic)
        self.policy_net = self._build_policy_network().to(self.device)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        
        # Episode memory
        self.log_probs = []
        self.rewards = []
        
    def _build_policy_network(self) -> nn.Module:
        """Build policy network with action heads for each action type."""
        class PolicyNetwork(nn.Module):
            def __init__(self, feature_dim, action_sizes):
                super().__init__()
                self.shared = nn.Sequential(
                    nn.Linear(feature_dim, 256),
                    nn.ReLU(),
                    nn.Linear(256, 256),
                    nn.ReLU()
                )
                # Separate heads for each action type
                self.heads = nn.ModuleDict({
                    key: nn.Linear(256, size)
                    for key, size in action_sizes.items()
                })
                
            def forward(self, x):
                shared_features = self.shared(x)
                return {
                    key: head(shared_features)
                    for key, head in self.heads.items()
                }
        
        return PolicyNetwork(self.feature_dim, self.action_sizes)
    
    def select_action(self, state: torch.Tensor, mask: Dict[str, torch.Tensor] = None) -> Tuple[Dict[str, int], torch.Tensor]:
        """
        Sample action from policy.
        
        Returns:
            action: Dict of selected action indices
            log_prob: Log probability of the action
        """
        self.policy_net.eval()
        with torch.no_grad():
            logits_dict = self.policy_net(state)
        
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
        
        return action, log_prob_total
    
    def store_transition(self, log_prob: torch.Tensor, reward: float):
        """Store transition for episode."""
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
    
    def update(self) -> float:
        """
        Update policy using Monte Carlo returns (REINFORCE update).
        
        Returns:
            policy_loss: The computed policy gradient loss
        """
        if len(self.rewards) == 0:
            return 0.0
        
        # Compute discounted returns
        returns = []
        G = 0
        for r in reversed(self.rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        
        returns = torch.tensor(returns, device=self.device)
        
        # Normalize returns (helps with stability)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Compute policy gradient loss
        policy_loss = []
        for log_prob, G in zip(self.log_probs, returns):
            policy_loss.append(-log_prob * G)
        
        policy_loss = torch.stack(policy_loss).sum()
        
        # Backprop and update
        self.optimizer.zero_grad()
        policy_loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        # Clear episode memory
        self.log_probs = []
        self.rewards = []
        
        return policy_loss.item()
    
    def save(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, path)
    
    def load(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
    
    def get_num_params(self) -> int:
        """Return total number of parameters."""
        return sum(p.numel() for p in self.policy_net.parameters())
