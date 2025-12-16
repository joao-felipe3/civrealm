"""Test if we can monkey-patch CivRealm port checking to use offline ports."""
import os
import sys

print("Setting up environment...")

# Try to patch Ports class before importing civrealm
try:
    print("Attempting to patch CivRealm Ports class...")
    from civrealm.freeciv.utils import port_utils
    
    # Save original Ports class
    OriginalPorts = port_utils.Ports
    
    class OfflinePorts:
        """Offline port manager that doesn't check port server."""
        def __init__(self):
            self.port_range = range(6300, 6400)  # 100 ports available
            self.used_ports = set()
        
        def get(self):
            """Get an available port."""
            for port in self.port_range:
                if port not in self.used_ports:
                    self.used_ports.add(port)
                    print(f"  Allocated port: {port}")
                    return port
            raise RuntimeError("No available ports!")
        
        def release(self, port):
            """Release a port."""
            self.used_ports.discard(port)
            print(f"  Released port: {port}")
    
    # Replace Ports class
    port_utils.Ports = OfflinePorts
    print("✓ Successfully patched Ports class!")
    
except Exception as e:
    print(f"✗ Failed to patch Ports class: {e}")
    import traceback
    traceback.print_exc()

print("\nNow trying to import civrealm...")
import civrealm
import gymnasium as gym

print("✓ CivRealm imported successfully!")

print("\nCreating environment...")
try:
    env = gym.make(
        "civrealm/FreecivTensorMinitask-v0",
        minitask_pattern={"type": "development_build_city", "level": "easy"}
    )
    print("✓ Environment created!")
    print("✓ All tests passed!")
except Exception as e:
    print(f"✗ Failed: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
