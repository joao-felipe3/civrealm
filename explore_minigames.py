"""
Script to explore the available CivRealm mini-games.
"""
import civrealm
from civrealm.envs.freeciv_minitask_env import MinitaskType

print("=" * 70)
print("CIVREALM MINI-GAMES")
print("=" * 70)

all_tasks = MinitaskType.list()
print(f"\nTotal de mini-games: {len(all_tasks)}\n")

# Categorize by type
categories = {
    'battle': [],
    'development': [],
    'diplomacy': []
}

for task in all_tasks:
    if 'battle' in task:
        categories['battle'].append(task)
    elif 'development' in task:
        categories['development'].append(task)
    elif 'diplomacy' in task:
        categories['diplomacy'].append(task)

print("CATEGORIES:")
print("\n1. COMBAT (Battle Tasks):")
for task in categories['battle']:
    print(f"   - {task}")

print("\n2. DEVELOPMENT (Development Tasks):")
for task in categories['development']:
    print(f"   - {task}")

print("\n3. DIPLOMACY (Diplomacy Tasks):")
for task in categories['diplomacy']:
    print(f"   - {task}")



print("\nSimplest to start:")
print("  1. development_build_city - Build cities (expansion focus)")
print("  2. development_build_infra - Build infrastructure")
print("  3. battle_ancient_era - Ancient era combat (limited units)")

print("\nFinal recommendation: 'development_build_city' (easy)")
print("Reasons:")
print("  - Clear, bounded objective")
print("  - Less complexity than combat")
print("  - Simpler action space")
print("  - More manageable training time")
