"""Minimal environment test - no heavy network processing."""
import os
import sys

# Set timeout very high to avoid port checking issues
os.environ['CIVREALM_DISABLE_DEV_CHECK'] = '1'

print("Importing CivRealm...", flush=True)
import civrealm
print("✓ CivRealm imported", flush=True)

print("Importing gymnasium...", flush=True)
import gymnasium as gym
print("✓ gymnasium imported", flush=True)

print("Creating environment...", flush=True)
try:
    env = gym.make(
        "civrealm/FreecivTensorMinitask-v0",
        minitask_pattern={"type": "development_build_city", "level": "easy"}
    )
    print("✓ Environment created!", flush=True)
    
    print("\nResetting environment...", flush=True)
    obs, info = env.reset()
    print("✓ Environment reset successful!", flush=True)
    
    print("\nObservation keys:", list(obs.keys()), flush=True)
    
    print("\nTaking 3 random actions...", flush=True)
    for i in range(3):
        action = env.action_space.sample()
        try:
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"  Step {i+1}: reward={reward}, terminated={terminated}, truncated={truncated}", flush=True)
        except Exception as e:
            print(f"  Step {i+1} failed: {e}", flush=True)
    
    print("\nClosing environment...", flush=True)
    env.close()
    print("✓ Environment closed!", flush=True)
    
except Exception as e:
    print(f"✗ Error: {type(e).__name__}: {e}", flush=True)
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✓ All tests passed!")
