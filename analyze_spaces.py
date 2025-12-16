"""
Script to understand the tensor formats (observation_space and action_space)
for the chosen mini-game: development_build_city (easy)
"""
import os
os.environ["RAY_DEDUP_LOGS"] = "0"

import civrealm
import gymnasium as gym
import numpy as np

print("=" * 70)
print("OBSERVATION_SPACE AND ACTION_SPACE ANALYSIS")
print("Mini-game: development_build_city (easy)")
print("=" * 70)

# Create environment
print("\n[1] Creating environment...")
env = gym.make(
    "civrealm/FreecivTensorMinitask-v0",
    minitask_pattern={"type": "development_build_city", "level": "easy"}
)
print("    [OK] Environment created!")

print("\n[2] OBSERVATION SPACE")
print("=" * 70)
print(f"Type: {type(env.observation_space)}")
print(f"\nStructure: {env.observation_space}")

# Reset to obtain a real observation
obs, info = env.reset()

print("\n[3] EXAMPLE OBSERVATION AFTER RESET")
print("=" * 70)
print(f"Observation type: {type(obs)}")

if isinstance(obs, dict):
    print(f"\nAvailable keys ({len(obs.keys())} total):")
    for key in sorted(obs.keys()):
        value = obs[key]
        if isinstance(value, np.ndarray):
            print(f"  {key:25s} -> shape: {str(value.shape):20s} dtype: {value.dtype}")
        else:
            print(f"  {key:25s} -> type: {type(value)}")

print("\n[4] DETAILED ANALYSIS OF KEY TENSORS")
print("=" * 70)

# Analyze the most important tensors
important_keys = ['map', 'unit', 'city', 'player', 'actor_type_mask', 
                  'unit_id_mask', 'unit_action_type_mask']

for key in important_keys:
    if key in obs:
        tensor = obs[key]
        print(f"\n{key}:")
        print(f"  Shape: {tensor.shape}")
        print(f"  Dtype: {tensor.dtype}")
        print(f"  Range: [{tensor.min():.2f}, {tensor.max():.2f}]")
        print(f"  Unique values: {len(np.unique(tensor))}")
        if tensor.size < 50:
                print(f"  Values: {tensor}")

print("\n[5] ACTION SPACE")
print("=" * 70)
print(f"Type: {type(env.action_space)}")
print(f"\nStructure: {env.action_space}")

# Try sampling an action
action = env.action_space.sample()
print(f"\n[6] SAMPLED ACTION EXAMPLE")
print("=" * 70)
print(f"Type: {type(action)}")
if isinstance(action, dict):
     print(f"\nAction keys ({len(action.keys())} total):")
     for key, value in action.items():
          print(f"  {key:25s} -> value: {value}")

print("\n[7] PROJECT INTERPRETATION")
print("=" * 70)
print("""
NOTES:
1. The state is a DICTIONARY of tensors (not a single matrix)
2. Each tensor carries specific meaning:
    - 'map': Map representation (likely 2D or 3D)
    - 'unit': Unit information
    - 'city': City information
    - 'player': Player information
    - '*_mask': Masks indicating valid actions

3. Actions are also a DICTIONARY (not a single number)
    - actor_type: Actor type (unit/city/etc.)
    - *_id: ID of the acting entity
    - *_action_type: Action type to execute

IMPLICATIONS FOR AGENTS:
- DQN: Needs a network that processes multiple tensors
  * Possible architecture: CNN for 'map', MLPs for other features
  * Concatenate features and pass through dense layers
  
- REINFORCE: Same input challenge, but the output is a policy
  
- PPO: Same input, but with actor-critic (policy + value)

CHALLENGE: The action_space is DISCRETE but STRUCTURED (Dict)
- It is not just choosing a number from 0-N
- It is choosing actor type + ID + action
- This makes agent implementation more complex!
""")

print("\n[8] SAVING INFORMATION FOR DOCUMENTATION")
print("=" * 70)

# Save structures for reference
with open("observation_action_analysis.txt", "w") as f:
    f.write("OBSERVATION SPACE\n")
    f.write("=" * 70 + "\n")
    f.write(str(env.observation_space) + "\n\n")
    
    f.write("OBSERVATION KEYS AND SHAPES\n")
    f.write("=" * 70 + "\n")
    for key in sorted(obs.keys()):
        value = obs[key]
        if isinstance(value, np.ndarray):
            f.write(f"{key}: {value.shape} ({value.dtype})\n")
    
    f.write("\n\nACTION SPACE\n")
    f.write("=" * 70 + "\n")
    f.write(str(env.action_space) + "\n\n")
    
    f.write("ACTION KEYS\n")
    f.write("=" * 70 + "\n")
    if isinstance(action, dict):
        for key in action.keys():
            f.write(f"{key}\n")

print("    [OK] Analysis saved to 'observation_action_analysis.txt'")

env.close()
print("\n[OK] Analysis complete!")
