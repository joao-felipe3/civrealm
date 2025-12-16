"""Utility functions for action handling in CivRealm."""
from __future__ import annotations

from typing import Dict, Any
import numpy as np
import gymnasium as gym


def get_action_dims(action_space: gym.Space) -> Dict[str, int]:
    """Extract action dimensions from CivRealm action space.
    
    Args:
        action_space: The environment's action space
        
    Returns:
        Dictionary mapping head names to their dimensions
    """
    if isinstance(action_space, gym.spaces.Dict):
        dims = {}
        for key, space in action_space.spaces.items():
            if isinstance(space, gym.spaces.Discrete):
                dims[key] = space.n
            elif isinstance(space, gym.spaces.MultiDiscrete):
                # Use the first dimension or max
                dims[key] = int(space.nvec[0]) if len(space.nvec) > 0 else 1
        return dims
    else:
        raise ValueError(f"Unsupported action space type: {type(action_space)}")


def extract_masks(obs: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Extract action masks from observation dict.
    
    Args:
        obs: Observation dictionary
        
    Returns:
        Dictionary of masks (only keys ending with '_mask')
    """
    masks = {}
    for key, value in obs.items():
        if key.endswith("_mask"):
            if isinstance(value, np.ndarray):
                masks[key] = value
            else:
                masks[key] = np.array(value)
    return masks


def extract_obs_features(obs: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Extract feature tensors from observation dict (excluding masks).
    
    Args:
        obs: Observation dictionary
        
    Returns:
        Dictionary of feature tensors (no mask keys)
    """
    features = {}
    for key, value in obs.items():
        if not key.endswith("_mask"):
            if isinstance(value, np.ndarray):
                features[key] = value
            else:
                features[key] = np.array(value)
    return features


if __name__ == "__main__":
    print("Testing action utilities...")
    
    # Test get_action_dims with mock action space
    from gymnasium.spaces import Dict as DictSpace, Discrete
    
    mock_action_space = DictSpace({
        "actor_type": Discrete(3),
        "unit_id": Discrete(128),
        "unit_action_type": Discrete(20),
    })
    
    dims = get_action_dims(mock_action_space)
    print(f"Action dimensions: {dims}")
    assert dims == {"actor_type": 3, "unit_id": 128, "unit_action_type": 20}
    
    # Test mask extraction
    mock_obs = {
        "map": np.random.randn(16, 16, 112),
        "unit": np.random.randn(128, 125),
        "actor_type_mask": np.array([1, 0, 1]),
        "unit_id_mask": np.random.randint(0, 2, 128),
        "unit_action_type_mask": np.random.randint(0, 2, 20),
    }
    
    masks = extract_masks(mock_obs)
    print(f"\nExtracted masks: {list(masks.keys())}")
    assert "actor_type_mask" in masks
    assert "map" not in masks
    
    features = extract_obs_features(mock_obs)
    print(f"Extracted features: {list(features.keys())}")
    assert "map" in features
    assert "actor_type_mask" not in features
    
    print("\n✓ Action utilities test passed!")
