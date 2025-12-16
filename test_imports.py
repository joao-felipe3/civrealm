"""Ultra-minimal test - check if we can even import the modules."""
print("Step 1: Importing torch...")
import torch
print("  ✓ torch OK")

print("Step 2: Importing numpy...")
import numpy as np
print("  ✓ numpy OK")

print("Step 3: Importing gymnasium...")
import gymnasium as gym
print("  ✓ gymnasium OK")

print("Step 4: Importing civrealm...")
import civrealm
print("  ✓ civrealm OK")

print("\nStep 5: Checking CivRealm registration...")
from gymnasium import registry
if "civrealm/FreecivTensorMinitask-v0" in registry:
    print("  ✓ civrealm/FreecivTensorMinitask-v0 is registered")
else:
    print("  ✗ civrealm/FreecivTensorMinitask-v0 NOT registered")
    print(f"  Available CivRealm envs: {[e for e in registry if 'civrealm' in e]}")

print("\nAll imports successful!")
