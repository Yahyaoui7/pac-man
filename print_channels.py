import json

with open("/home/nyahyaou/goinfre/pac_man_intra/AI_arena/data/CNN_DATA.jsonl", "r") as f:
    line = f.readline()
    
data = json.loads(line)
grid = data["grid"]

# Let's find the player's position from Channel 3 so we can show a relevant 7x7 patch
player_channel = grid[3]
py, px = 0, 0
for y in range(len(player_channel)):
    for x in range(len(player_channel[0])):
        if player_channel[y][x] == 1.0:
            py, px = y, x
            break

print("Top-left corner (7x7 patch) of the Maze:")
print("="*40)
print("Channel 0 (Raw Bitmask / 15.0):")
print("-" * 40)
for y in range(7):
    row = grid[0][y][:7]
    print(" ".join(f"{val:4.2f}" for val in row))

print("\nChannel 8 (Walkable Binary Mask):")
print("-" * 40)
for y in range(7):
    row = grid[8][y][:7]
    print(" ".join(f"{val:4.1f}" for val in row))
