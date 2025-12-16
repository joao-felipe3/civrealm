"""Test environment creation with custom socket timeout."""
import socket
import sys

# Increase socket timeout for all connections
socket.setdefaulttimeout(30)

print(f"Socket timeout set to: {socket.getdefaulttimeout()}s")

print("\nImporting CivRealm...")
import civrealm
import gymnasium as gym

print("Creating environment (this may take a moment)...")
try:
    env = gym.make(
        "civrealm/FreecivTensorMinitask-v0",
        minitask_pattern={"type": "development_build_city", "level": "easy"}
    )
    print("✓ Environment created successfully!")
    print("\nChecking observation...")
    obs, info = env.reset()
    print(f"✓ Reset successful! Got observation with keys: {list(obs.keys())}")
    env.close()
    print("✓ Closed successfully!")
except KeyboardInterrupt:
    print("\n✗ Interrupted by user (socket timeout issue)")
    sys.exit(1)
except Exception as e:
    print(f"\n✗ Error: {type(e).__name__}")
    print(f"  Message: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
