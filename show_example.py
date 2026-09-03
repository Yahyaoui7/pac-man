import json

with open("/home/nyahyaou/goinfre/pac_man_intra/AI_arena/data/CNN_DATA.jsonl", "r") as f:
    line = f.readline()
    
data = json.loads(line)

print("Dataset Record Example:")
print("-" * 30)
print(f"1. grid (Image Data): shape = ({len(data['grid'])}, {len(data['grid'][0])}, {len(data['grid'][0][0])})")
print("   Channel 0: Maze walls")
print("   Channel 1: Normal pellets")
print("   Channel 2: Power pellets")
print("   Channel 3: Pac-Man's position")
print("   Channel 4: Blinky's position")
print("   Channel 5: Pinky's position")
print("   Channel 6: Inky's position")
print("   Channel 7: Clyde's position")
print("   Channel 8: Walkable area")

print(f"\n2. extra_features (Scalar Data): shape = ({len(data['extra_features'])},)")
print(f"   Includes player direction, distances to each ghost, etc.")

print(f"\n3. valid_actions (Masks): shape = ({len(data['valid_actions'])}, {len(data['valid_actions'][0])})")
print("   (True means the ghost can move in that direction without hitting a wall)")
print(f"   Blinky (Ghost 0): {data['valid_actions'][0]}")
print(f"   Pinky  (Ghost 1): {data['valid_actions'][1]}")
print(f"   Inky   (Ghost 2): {data['valid_actions'][2]}")
print(f"   Clyde  (Ghost 3): {data['valid_actions'][3]}")

print(f"\n4. labels (Target Moves):")
directions = ["UP", "DOWN", "LEFT", "RIGHT"]
print(f"   Blinky (Ghost 0) target: {directions[data['labels'][0]]} ({data['labels'][0]})")
print(f"   Pinky  (Ghost 1) target: {directions[data['labels'][1]]} ({data['labels'][1]})")
print(f"   Inky   (Ghost 2) target: {directions[data['labels'][2]]} ({data['labels'][2]})")
print(f"   Clyde  (Ghost 3) target: {directions[data['labels'][3]]} ({data['labels'][3]})")
