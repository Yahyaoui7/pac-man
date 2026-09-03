import sys
with open("/home/nyahyaou/goinfre/pac_man_intra/AI_arena/data/constants.py", "r") as f:
    content = f.read()

content = content.replace("CNN_CHANNEL_COUNT = 6", "CNN_CHANNEL_COUNT = 9")

with open("/home/nyahyaou/goinfre/pac_man_intra/AI_arena/data/constants.py", "w") as f:
    f.write(content)
print("Success")
