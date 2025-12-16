"""DQN network with multi-head outputs for structured CivRealm actions."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn

# Allow direct execution
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from networks.feature_extractor import CivFeatureExtractor, TensorDict
else:
    from .feature_extractor import CivFeatureExtractor, TensorDict


class CivDQNNetwork(nn.Module):
    """Maps CivRealm observations to per-head Q-values for all discrete action parts."""

    def __init__(
        self,
        feature_extractor: CivFeatureExtractor,
        action_dims: Dict[str, int],
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.feature_extractor = feature_extractor
        self.action_dims = action_dims

        self.heads = nn.ModuleDict()
        for head_name, dim in action_dims.items():
            self.heads[head_name] = nn.Sequential(
                nn.Linear(feature_extractor.output_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, dim),
            )

    def forward(self, obs: TensorDict) -> Dict[str, torch.Tensor]:
        features = self.feature_extractor(obs)
        return {name: head(features) for name, head in self.heads.items()}


if __name__ == "__main__":
    print("Testing CivDQNNetwork...")
    
    action_dims = {
        "actor_type": 3,
        "unit_id": 128,
        "unit_action_type": 20,
        "city_id": 32,
        "city_action_type": 50,
    }

    # Create feature extractor and DQN network
    extractor = CivFeatureExtractor()
    dqn = CivDQNNetwork(
        feature_extractor=extractor,
        action_dims=action_dims,
    )
    
    print(f"Network created with {sum(p.numel() for p in dqn.parameters())} parameters")
    
    # Test with dummy observation
    dummy_obs = {
        "map": torch.randn(16, 16, 112),
        "unit": torch.randn(128, 125),
        "city": torch.randn(32, 248),
        "player": torch.randn(32),
    }
    
    q_values = dqn(dummy_obs)
    print(f"\nOutput heads:")
    for head_name, q_vals in q_values.items():
        print(f"  {head_name}: {q_vals.shape}")
    
    # Test with batch
    batch_obs = {
        "map": torch.randn(4, 16, 16, 112),
        "unit": torch.randn(4, 128, 125),
        "city": torch.randn(4, 32, 248),
        "player": torch.randn(4, 32),
    }
    
    batch_q_values = dqn(batch_obs)
    print(f"\nBatch output shapes:")
    for head_name, q_vals in batch_q_values.items():
        print(f"  {head_name}: {q_vals.shape}")
    
    print("\n✓ CivDQNNetwork test passed!")
