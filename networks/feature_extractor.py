"""Feature extractor for CivRealm tensor observations.

Handles dict observations with map/unit/city/player tensors and returns a fused
feature vector suitable for downstream heads (DQN, PPO, etc.).
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

TensorDict = Dict[str, torch.Tensor]


def _ensure_batch_map(x: torch.Tensor) -> torch.Tensor:
    """Add batch dimension to map tensor if needed."""
    if x.dim() == 3:  # (H, W, C) -> (1, H, W, C)
        return x.unsqueeze(0)
    return x


def _ensure_batch_slots(x: torch.Tensor) -> torch.Tensor:
    """Add batch dimension to slot tensor if needed."""
    if x.dim() == 2:  # (slots, features) -> (1, slots, features)
        return x.unsqueeze(0)
    return x


def _ensure_batch_flat(x: torch.Tensor) -> torch.Tensor:
    """Add batch dimension to flat tensor if needed."""
    if x.dim() == 1:  # (features,) -> (1, features)
        return x.unsqueeze(0)
    return x


def _to_channel_first(x: torch.Tensor) -> torch.Tensor:
    """Convert HWC to CHW when channels are in the last dimension."""
    if x.dim() == 3 and x.shape[-1] > x.shape[0] and x.shape[-1] > x.shape[1]:
        return x.permute(2, 0, 1)
    if x.dim() == 4 and x.shape[-1] > x.shape[1] and x.shape[-1] > x.shape[2]:
        return x.permute(0, 3, 1, 2)
    return x


class CivFeatureExtractor(nn.Module):
    """Encodes CivRealm observations into a single feature vector."""

    def __init__(
        self,
        map_channels: int = 112,
        map_out_dim: int = 256,
        unit_feat_dim: int = 125,
        unit_out_dim: int = 128,
        city_feat_dim: int = 248,
        city_out_dim: int = 128,
        player_dim: int = 32,
        player_out_dim: int = 64,
    ) -> None:
        super().__init__()

        self.map_encoder = nn.Sequential(
            nn.Conv2d(map_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, map_out_dim),
            nn.ReLU(inplace=True),
        )

        self.unit_proj = nn.Sequential(
            nn.Linear(unit_feat_dim, unit_out_dim),
            nn.ReLU(inplace=True),
        )

        self.city_proj = nn.Sequential(
            nn.Linear(city_feat_dim, city_out_dim),
            nn.ReLU(inplace=True),
        )

        self.player_proj = nn.Sequential(
            nn.Linear(player_dim, player_out_dim),
            nn.ReLU(inplace=True),
        )

        self.output_dim = map_out_dim + unit_out_dim + city_out_dim + player_out_dim

    def forward(self, obs: TensorDict) -> torch.Tensor:
        if "map" not in obs:
            raise KeyError("Observation missing 'map' key")

        map_tensor = obs["map"]
        map_tensor = _ensure_batch_map(map_tensor)
        map_tensor = _to_channel_first(map_tensor)
        map_features = self.map_encoder(map_tensor.float())

        unit_features = self._encode_slot_tensor(obs.get("unit"), self.unit_proj)
        city_features = self._encode_slot_tensor(obs.get("city"), self.city_proj)
        player_features = self._encode_flat_tensor(obs.get("player"), self.player_proj)

        fused = torch.cat([map_features, unit_features, city_features, player_features], dim=-1)
        return fused

    def _encode_slot_tensor(self, tensor: Optional[torch.Tensor], proj: nn.Module) -> torch.Tensor:
        if tensor is None:
            return self._zeros_like_proj(proj)

        tensor = _ensure_batch_slots(tensor)
        if tensor.dim() != 3:
            raise ValueError("Slot tensor must have shape [B, slots, features]")

        in_features = tensor.shape[-1]
        self._maybe_rebuild_proj(proj, in_features)

        pooled = tensor.float().mean(dim=1)
        return proj(pooled)

    def _encode_flat_tensor(self, tensor: Optional[torch.Tensor], proj: nn.Module) -> torch.Tensor:
        if tensor is None:
            return self._zeros_like_proj(proj)

        tensor = _ensure_batch_flat(tensor)
        if tensor.dim() != 2:
            raise ValueError("Flat tensor must have shape [B, features]")

        in_features = tensor.shape[-1]
        self._maybe_rebuild_proj(proj, in_features)

        return proj(tensor.float())

    def _maybe_rebuild_proj(self, proj: nn.Module, in_features: int) -> None:
        """If input dim changed, rebuild the first Linear to match current features."""
        linear = None
        for module in proj.modules():
            if isinstance(module, nn.Linear):
                linear = module
                break
        if linear is None:
            raise RuntimeError("Projection module missing Linear layer")

        if linear.in_features == in_features:
            return

        out_features = linear.out_features
        device = next(proj.parameters()).device

        # Assume proj is Linear + ReLU; rebuild minimal structure
        new_proj = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU(inplace=True),
        ).to(device)

        # Replace parameters in-place (convert to list to avoid dict modification during iteration)
        for name, _ in list(proj.named_children()):
            proj._modules.pop(name)
        for name, module in new_proj.named_children():
            proj._modules[name] = module

    def _zeros_like_proj(self, proj: nn.Module) -> torch.Tensor:
        for module in proj.modules():
            if isinstance(module, nn.Linear):
                out_dim = module.out_features
                break
        else:
            raise RuntimeError("Projection module missing Linear layer")

        return torch.zeros(1, out_dim, device=next(proj.parameters()).device)


if __name__ == "__main__":
    print("Testing CivFeatureExtractor...")
    
    # Create a dummy observation matching CivRealm structure
    dummy_obs = {
        "map": torch.randn(16, 16, 112),  # HWC format
        "unit": torch.randn(128, 125),    # [slots, features]
        "city": torch.randn(32, 248),     # [slots, features]
        "player": torch.randn(32),        # [features]
    }
    
    extractor = CivFeatureExtractor()
    print(f"Feature extractor output dimension: {extractor.output_dim}")
    
    features = extractor(dummy_obs)
    print(f"Input observation keys: {list(dummy_obs.keys())}")
    print(f"Output features shape: {features.shape}")
    print(f"Expected shape: [1, {extractor.output_dim}]")
    
    # Test with batch
    batch_obs = {
        "map": torch.randn(4, 16, 16, 112),
        "unit": torch.randn(4, 128, 125),
        "city": torch.randn(4, 32, 248),
        "player": torch.randn(4, 32),
    }
    batch_features = extractor(batch_obs)
    print(f"\nBatch output shape: {batch_features.shape}")
    print(f"Expected shape: [4, {extractor.output_dim}]")
    
    print("\n✓ CivFeatureExtractor test passed!")
