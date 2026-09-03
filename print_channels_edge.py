import json

with open("/home/nyahyaou/goinfre/pac_man_intra/AI_arena/data/CNN_DATA.jsonl", "r") as f:
    line = f.readline()
    
data = json.loads(line)
grid = data["grid"]
width = data["maze_width"]
height = data["maze_height"]

print(f"Maze is {width}x{height} inside a 50x25 grid.")
print("\nBottom-Right corner of the maze showing the padding border:")
print("=========================================================")
print("Channel 0 (Raw Bitmask / 15.0):")
print("  (Notice how it drops to 0.00 outside the maze boundary)")
print("-" * 50)
start_y = height - 4
start_x = width - 4
for y in range(start_y, start_y + 8):
    row = grid[0][y][start_x:start_x+8]
    print(" ".join(f"{val:4.2f}" for val in row))

print("\nChannel 8 (Active Map Mask):")
print("  (1.0 inside the maze, 0.0 in the padding area)")
print("-" * 50)
for y in range(start_y, start_y + 8):
    row = grid[8][y][start_x:start_x+8]
    print(" ".join(f"{val:4.1f}" for val in row))

