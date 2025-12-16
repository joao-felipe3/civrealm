"""Experience replay buffer for DQN agent."""
from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Tuple
import random

import numpy as np
import torch


class ReplayBuffer:
    """Stores transitions with dict observations and actions."""

    def __init__(self, capacity: int = 10000) -> None:
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        obs: Dict[str, np.ndarray],
        action: Dict[str, int],
        reward: float,
        next_obs: Dict[str, np.ndarray],
        done: bool,
        masks: Optional[Dict[str, np.ndarray]] = None,
    ) -> None:
        """Add a transition to the buffer."""
        self.buffer.append((obs, action, reward, next_obs, done, masks))

    def sample(self, batch_size: int) -> Tuple[
        Dict[str, torch.Tensor],  # obs
        Dict[str, torch.Tensor],  # actions
        torch.Tensor,              # rewards
        Dict[str, torch.Tensor],  # next_obs
        torch.Tensor,              # dones
        Optional[Dict[str, torch.Tensor]],  # masks
    ]:
        """Sample a batch of transitions."""
        if len(self.buffer) < batch_size:
            raise ValueError(f"Not enough samples: {len(self.buffer)} < {batch_size}")

        batch = random.sample(self.buffer, batch_size)

        # Unpack batch
        obs_list, action_list, reward_list, next_obs_list, done_list, mask_list = zip(*batch)

        # Stack observations
        obs_batch = self._stack_obs(obs_list)
        next_obs_batch = self._stack_obs(next_obs_list)

        # Stack actions
        action_batch = self._stack_actions(action_list)

        # Convert rewards and dones
        rewards = torch.tensor(reward_list, dtype=torch.float32)
        dones = torch.tensor(done_list, dtype=torch.float32)

        # Stack masks if present
        masks_batch = None
        if mask_list[0] is not None:
            masks_batch = self._stack_masks(mask_list)

        return obs_batch, action_batch, rewards, next_obs_batch, dones, masks_batch

    def _stack_obs(self, obs_list: List[Dict[str, np.ndarray]]) -> Dict[str, torch.Tensor]:
        """Stack dict observations into batched tensors."""
        keys = obs_list[0].keys()
        batched = {}
        for key in keys:
            arrays = [obs[key] for obs in obs_list]
            batched[key] = torch.from_numpy(np.stack(arrays, axis=0))
        return batched

    def _stack_actions(self, action_list: List[Dict[str, int]]) -> Dict[str, torch.Tensor]:
        """Stack dict actions into batched tensors."""
        keys = action_list[0].keys()
        batched = {}
        for key in keys:
            values = [action[key] for action in action_list]
            batched[key] = torch.tensor(values, dtype=torch.long)
        return batched

    def _stack_masks(self, mask_list: List[Dict[str, np.ndarray]]) -> Dict[str, torch.Tensor]:
        """Stack dict masks into batched tensors."""
        keys = mask_list[0].keys()
        batched = {}
        for key in keys:
            arrays = [mask[key] for mask in mask_list]
            batched[key] = torch.from_numpy(np.stack(arrays, axis=0))
        return batched

    def __len__(self) -> int:
        return len(self.buffer)


if __name__ == "__main__":
    print("Testing ReplayBuffer...")

    buffer = ReplayBuffer(capacity=100)

    # Add dummy transitions
    for i in range(50):
        obs = {
            "map": np.random.randn(16, 16, 112).astype(np.float32),
            "unit": np.random.randn(128, 125).astype(np.float32),
            "city": np.random.randn(32, 248).astype(np.float32),
            "player": np.random.randn(32).astype(np.float32),
        }
        action = {
            "actor_type": np.random.randint(0, 3),
            "unit_id": np.random.randint(0, 128),
            "unit_action_type": np.random.randint(0, 20),
        }
        reward = np.random.randn()
        next_obs = {
            "map": np.random.randn(16, 16, 112).astype(np.float32),
            "unit": np.random.randn(128, 125).astype(np.float32),
            "city": np.random.randn(32, 248).astype(np.float32),
            "player": np.random.randn(32).astype(np.float32),
        }
        done = i % 10 == 0
        masks = {
            "actor_type_mask": np.random.randint(0, 2, size=3).astype(np.float32),
            "unit_id_mask": np.random.randint(0, 2, size=128).astype(np.float32),
            "unit_action_type_mask": np.random.randint(0, 2, size=20).astype(np.float32),
        }

        buffer.push(obs, action, reward, next_obs, done, masks)

    print(f"Buffer size: {len(buffer)}")

    # Sample a batch
    batch_size = 8
    obs_b, action_b, rewards_b, next_obs_b, dones_b, masks_b = buffer.sample(batch_size)

    print(f"\nSampled batch of size {batch_size}:")
    print(f"  Observations:")
    for k, v in obs_b.items():
        print(f"    {k}: {v.shape}")
    print(f"  Actions:")
    for k, v in action_b.items():
        print(f"    {k}: {v.shape}")
    print(f"  Rewards: {rewards_b.shape}")
    print(f"  Next observations:")
    for k, v in next_obs_b.items():
        print(f"    {k}: {v.shape}")
    print(f"  Dones: {dones_b.shape}")
    print(f"  Masks:")
    for k, v in masks_b.items():
        print(f"    {k}: {v.shape}")

    print("\n✓ ReplayBuffer test passed!")
