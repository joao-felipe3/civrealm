"""Simple DQN training using CivRealm's TensorBaselineEnv (safer port handling)."""
import os
os.environ["RAY_DEDUP_LOGS"] = "0"

import sys
import time
import numpy as np
import torch
from pathlib import Path

# Import CivRealm and our modules
import civrealm

from agents import DQNAgent, ReplayBuffer
from networks import CivFeatureExtractor, CivDQNNetwork
from utils.action_utils import get_action_dims, extract_masks, extract_obs_features
from civtensor.envs.freeciv_tensor_env.freeciv_tensor_env import TensorBaselineEnv


def train_with_tensor_env():
    """Training using TensorBaselineEnv wrapper (avoids direct gym.make port issues)."""
    print("=" * 70)
    print("DQN TRAINING - Using TensorBaselineEnv")
    print("=" * 70)
    
    try:
        print("\n[1] Creating TensorBaselineEnv...")
        env = TensorBaselineEnv(parallel_number=1, task="development_build_city easy")
        print("    ✓ Environment created")
        
        # Get one observation to understand structure
        print("\n[2] Getting sample observation...")
        obs, reward, done, info, available_actions = env.reset()
        
        # Convert to single sample format
        if isinstance(obs, list):
            obs = obs[0]
        
        print(f"    ✓ Got observation with keys: {list(obs.keys())}")
        print(f"    ✓ Available actions: {type(available_actions)}")
        
        print("\n[3] Testing action selection...")
        actions = {}
        for key in [
            "actor_type", "city_id", "city_action_type", 
            "unit_id", "unit_action_type",
            "dipl_id", "dipl_action_type",
            "gov_action_type", "tech_action_type"
        ]:
            actions[key] = np.array([0])
        
        print(f"    Sample action: {actions}")
        print(f"    Stepping environment...")
        obs, reward, done, info, available_actions = env.step(actions)
        print(f"    ✓ Step successful! Reward: {reward}")
        
        env.close()
        print("\n✓ All tests passed!")
        
    except Exception as e:
        print(f"\n✗ Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = train_with_tensor_env()
    sys.exit(0 if success else 1)
