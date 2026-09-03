import json

with open("/home/nyahyaou/goinfre/pac_man_intra/AI_arena/data/CNN_DATA.jsonl", "r") as f:
    line = f.readline()
    
data = json.loads(line)
grid = data["grid"]

print("A 7x7 patch of the Maze that contains solid walls:")
print("==================================================")
print("Channel 0 (Raw Bitmask / 15.0):")
print("  (Values < 1.0 are open paths with specific shapes. 1.0 is a solid wall block.)")
print("-" * 50)
# Find a 7x7 patch that contains a 1.0 (wall)
found = False
for start_y in range(len(grid[0])-7):
    for start_x in range(len(grid[0][0])-7):
        # check if this 7x7 patch has a wall (1.0)
        patch = [row[start_x:start_x+7] for row in grid[0][start_y:start_y+7]]
        if any(1.0 in row for row in patch):
            for row in patch:
                print(" ".join(f"{val:4.2f}" for val in row))
            found = True
            
            print("\nChannel 8 (Walkable Binary Mask):")
            print("  (Values are 1.0 for open paths, 0.0 for solid walls/padding.)")
            print("-" * 50)
            patch8 = [row[start_x:start_x+7] for row in grid[8][start_y:start_y+7]]
            for row in patch8:
                print(" ".join(f"{val:4.1f}" for val in row))
            break
    if found:
        break
