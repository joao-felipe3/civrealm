"""Debug script to understand CivRealm action format."""
import os
os.environ["RAY_DEDUP_LOGS"] = "0"

import civrealm
import gymnasium as gym
import json

# Create environment
env = gym.make(
    "civrealm/FreecivTensorMinitask-v0",
    minitask_pattern={"type": "development_build_city", "level": "easy"}
)

print("=" * 70)
print("CIVREALM ACTION FORMAT DEBUGGING")
print("=" * 70)

obs, info = env.reset()

print("\n[1] Observation keys:")
for key in sorted(obs.keys()):
    if isinstance(obs[key], (list, tuple)):
        print(f"    {key}: {len(obs[key])} items")
    else:
        print(f"    {key}: shape {getattr(obs[key], 'shape', 'N/A')}")

print("\n[2] Available actions in observation:")
if "available_actions" in obs:
    avail = obs["available_actions"]
    print(f"    Type: {type(avail)}")
    if isinstance(avail, list) and len(avail) > 0:
        print(f"    Count: {len(avail)}")
        print(f"    First 5 actions:")
        for i, action in enumerate(avail[:5]):
            print(f"      {i}: {action}")

print("\n[3] Action space:")
print(f"    Type: {type(env.action_space)}")
if hasattr(env.action_space, 'spaces'):
    for key, space in env.action_space.spaces.items():
        print(f"    {key}: {space}")

print("\n[4] Sample action from gym.make():")
sampled = env.action_space.sample()
print(f"    Type: {type(sampled)}")
print(f"    Keys: {list(sampled.keys())}")
for k, v in sampled.items():
    print(f"      {k}: {v} (type: {type(v).__name__})")

print("\n[5] Try step with sampled action:")
try:
    next_obs, reward, terminated, truncated, info = env.step(sampled)
    print(f"    ✓ Step succeeded")
    print(f"    Reward: {reward}")
    print(f"    Terminated: {terminated}, Truncated: {truncated}")
except Exception as e:
    print(f"    ✗ Step failed: {type(e).__name__}: {e}")

print("\n[6] Try step with valid action from available_actions:")
if "available_actions" in obs:
    avail = obs["available_actions"]
    if avail:
        # Try first available action
        first_action = avail[0]
        print(f"    First available action: {first_action}")
        try:
            next_obs, reward, terminated, truncated, info = env.step(first_action)
            print(f"    ✓ Step with available_actions[0] succeeded")
            print(f"    Reward: {reward}")
        except Exception as e:
            print(f"    ✗ Step failed: {type(e).__name__}: {e}")

env.close()
print("\n[OK] Debug complete!")
