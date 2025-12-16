"""DQN agent with epsilon-greedy action selection and mask handling."""
from __future__ import annotations

from typing import Dict, Optional, Tuple
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from networks import CivDQNNetwork


class DQNAgent:
    """DQN agent for structured action spaces with masking."""

    def __init__(
        self,
        network: CivDQNNetwork,
        action_dims: Dict[str, int],
        lr: float = 1e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 10000,
        target_update_freq: int = 1000,
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.update_counter = 0
        self.action_dims = action_dims

        # Networks
        self.q_network = network.to(self.device)
        self.target_network = copy.deepcopy(self.q_network).to(self.device)
        self.target_network.eval()

        # Optimizer
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)

        # Epsilon schedule
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.steps = 0

    def select_action(
        self,
        obs: Dict[str, np.ndarray],
        masks: Dict[str, np.ndarray],
        eval_mode: bool = False,
    ) -> Dict[str, int]:
        """Select action with epsilon-greedy policy and mask enforcement."""
        epsilon = self._get_epsilon() if not eval_mode else 0.0

        if np.random.rand() < epsilon:
            # Random action (respecting masks)
            return self._sample_masked_action(masks)
        else:
            # Greedy action
            return self._select_greedy_action(obs, masks)

    def _select_greedy_action(
        self,
        obs: Dict[str, np.ndarray],
        masks: Dict[str, np.ndarray],
    ) -> Dict[str, int]:
        """Select action with highest Q-value (applying masks)."""
        self.q_network.eval()
        with torch.no_grad():
            # Convert obs to tensors
            obs_tensor = {k: torch.from_numpy(v).to(self.device) for k, v in obs.items()}
            
            # Get Q-values for all heads
            q_values = self.q_network(obs_tensor)

            action = {}
            for head_name, q_vals in q_values.items():
                # Apply mask if available, otherwise assume all valid
                mask_key = f"{head_name}_mask"
                if mask_key in masks:
                    mask = torch.from_numpy(masks[mask_key]).to(self.device)
                    mask = self._align_mask_for_q(mask, q_vals)
                    q_vals = q_vals + (mask - 1.0) * 1e9

                action[head_name] = q_vals.squeeze(0).argmax().item()

        self.q_network.train()
        return action

    def _sample_masked_action(self, masks: Dict[str, np.ndarray]) -> Dict[str, int]:
        """Sample random action respecting masks (or full range if mask missing)."""
        action = {}
        for head_name, dim in self.action_dims.items():
            mask_key = f"{head_name}_mask"
            if mask_key in masks:
                mask = self._align_mask_numpy(masks[mask_key], dim)
                valid_indices = np.where(mask > 0)[0]
                if len(valid_indices) > 0:
                    action[head_name] = int(np.random.choice(valid_indices))
                    continue
            # Fallback: uniform over all actions for this head
            action[head_name] = int(np.random.randint(0, dim))
        return action

    def train_step(
        self,
        obs_batch: Dict[str, torch.Tensor],
        action_batch: Dict[str, torch.Tensor],
        reward_batch: torch.Tensor,
        next_obs_batch: Dict[str, torch.Tensor],
        done_batch: torch.Tensor,
        masks_batch: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, float]:
        """Perform one training step."""
        # Move to device
        obs_batch = {k: v.to(self.device) for k, v in obs_batch.items()}
        action_batch = {k: v.to(self.device) for k, v in action_batch.items()}
        reward_batch = reward_batch.to(self.device)
        next_obs_batch = {k: v.to(self.device) for k, v in next_obs_batch.items()}
        done_batch = done_batch.to(self.device)

        # Get current Q-values
        current_q_values = self.q_network(obs_batch)

        # Get target Q-values
        with torch.no_grad():
            next_q_values = self.target_network(next_obs_batch)
            
            # Apply masks to next Q-values if available
            if masks_batch is not None:
                for head_name, next_q in next_q_values.items():
                    mask_key = f"{head_name}_mask"
                    if mask_key in masks_batch:
                        mask = masks_batch[mask_key].to(self.device)
                        mask = self._align_mask_for_q(mask, next_q)
                        next_q = next_q + (mask - 1.0) * 1e9
                        next_q_values[head_name] = next_q

            # Compute targets for each head (max over valid actions)
            target_values = []
            for head_name in current_q_values.keys():
                next_q_max = next_q_values[head_name].max(dim=1)[0]

                # If everything was masked, max will be a large negative (~-1e9). Treat as 0 to avoid exploding targets.
                next_q_max = torch.where(next_q_max < -1e8, torch.zeros_like(next_q_max), next_q_max)

                # Sanitize non-finite
                if not torch.isfinite(next_q_max).all():
                    next_q_max = torch.nan_to_num(next_q_max, nan=0.0, posinf=0.0, neginf=0.0)

                target = reward_batch + self.gamma * next_q_max * (1.0 - done_batch)
                # Clamp targets to a reasonable range to stabilize TD updates
                target = torch.clamp(target, -50.0, 50.0)
                target_values.append(target)

        # Compute loss for each head
        losses = []
        for idx, (head_name, current_q) in enumerate(current_q_values.items()):
            action_indices = action_batch[head_name].unsqueeze(1)
            q_selected = current_q.gather(1, action_indices).squeeze(1)
            target = target_values[idx]

            # Sanitize non-finite targets and predictions
            if not torch.isfinite(target).all():
                target = torch.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)
            if not torch.isfinite(q_selected).all():
                q_selected = torch.nan_to_num(q_selected, nan=0.0, posinf=0.0, neginf=0.0)

            loss = F.smooth_l1_loss(q_selected, target)
            losses.append(loss)

        # Total loss
        total_loss = sum(losses)

        # Optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Update target network
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        # Return metrics
        return {
            "loss": total_loss.item(),
            "q_mean": sum(q.mean().item() for q in current_q_values.values()) / len(current_q_values),
        }

    def _get_epsilon(self) -> float:
        """Get current epsilon value."""
        self.steps += 1
        progress = min(self.steps / self.epsilon_decay_steps, 1.0)
        return self.epsilon_start + (self.epsilon_end - self.epsilon_start) * progress

    @property
    def epsilon(self) -> float:
        """Current epsilon value without incrementing step counter."""
        progress = min(self.steps / self.epsilon_decay_steps, 1.0)
        return self.epsilon_start + (self.epsilon_end - self.epsilon_start) * progress

    @staticmethod
    def _align_mask_numpy(mask: np.ndarray, dim: int) -> np.ndarray:
        """Align a 1D numpy mask to expected action dimension via pad/truncate."""
        m = np.asarray(mask)
        if m.ndim != 1:
            m = m.reshape(-1)
        if m.shape[0] == dim:
            return m
        if m.shape[0] < dim:
            pad = np.zeros((dim - m.shape[0],), dtype=m.dtype)
            return np.concatenate([m, pad], axis=0)
        return m[:dim]

    @staticmethod
    def _align_mask_for_q(mask: torch.Tensor, q_values: torch.Tensor) -> torch.Tensor:
        """Align mask tensor to match Q-values shape [batch, dim].

        CivRealm sometimes returns masks whose length does not match the declared
        Discrete(n) in the action space. To be robust, we pad with zeros (invalid)
        or truncate.
        """
        # Ensure both are 2D [batch, dim]
        if q_values.dim() == 1:
            q_values = q_values.unsqueeze(0)
        if mask.dim() == 1:
            mask = mask.unsqueeze(0)
        
        if mask.dim() != 2 or q_values.dim() != 2:
            # Fallback: return all-valid mask matching q_values
            return torch.ones_like(q_values, dtype=torch.float32)
        
        batch, dim = q_values.shape
        mask_batch, mask_dim = mask.shape
        
        # Handle transposed masks (dim, batch) -> transpose to (batch, dim)
        if mask_batch == dim and mask_dim == batch:
            mask = mask.t()
            mask_batch, mask_dim = mask.shape
        
        # Fix batch dimension mismatch
        if mask_batch == 1 and batch > 1:
            # Broadcast singleton batch
            mask = mask.expand(batch, -1)
            mask_batch = batch
        elif mask_batch != batch:
            # Create new mask with correct batch size
            mask = torch.ones((batch, mask_dim), dtype=mask.dtype, device=mask.device)
            mask_batch = batch
        
        # Fix action dimension mismatch
        if mask_dim < dim:
            # Pad with zeros (invalid actions)
            pad = torch.zeros((mask_batch, dim - mask_dim), dtype=mask.dtype, device=mask.device)
            mask = torch.cat([mask, pad], dim=1)
        elif mask_dim > dim:
            # Truncate extra actions
            mask = mask[:, :dim]
        
        # Ensure dtype matches q_values for arithmetic compatibility
        if mask.dtype != q_values.dtype:
            mask = mask.to(dtype=q_values.dtype)
        
        return mask

    def save(self, path: str) -> None:
        """Save agent state."""
        torch.save({
            "q_network": self.q_network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "steps": self.steps,
            "update_counter": self.update_counter,
        }, path)

    def load(self, path: str, strict: bool = True) -> None:
        """Load agent state.
        
        Args:
            path: Path to checkpoint
            strict: If False, ignore incompatible layers (useful for architecture changes)
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        if strict:
            self.q_network.load_state_dict(checkpoint["q_network"])
            self.target_network.load_state_dict(checkpoint["target_network"])
        else:
            # Load with partial matching - ignore incompatible layers
            q_dict = self.q_network.state_dict()
            q_checkpoint = checkpoint["q_network"]
            
            # Filter out layers with size mismatch
            matched = {}
            skipped = []
            for k, v in q_checkpoint.items():
                if k in q_dict and q_dict[k].shape == v.shape:
                    matched[k] = v
                else:
                    skipped.append(k)
            
            q_dict.update(matched)
            self.q_network.load_state_dict(q_dict)
            
            # Same for target network
            t_dict = self.target_network.state_dict()
            t_checkpoint = checkpoint["target_network"]
            matched = {}
            for k, v in t_checkpoint.items():
                if k in t_dict and t_dict[k].shape == v.shape:
                    matched[k] = v
            t_dict.update(matched)
            self.target_network.load_state_dict(t_dict)
            
            if skipped:
                print(f"    ⚠ Skipped {len(skipped)} incompatible layers")
        
        try:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        except:
            print("    ⚠ Optimizer state not loaded (incompatible)")
        
        self.steps = checkpoint.get("steps", 0)
        self.update_counter = checkpoint.get("update_counter", 0)
