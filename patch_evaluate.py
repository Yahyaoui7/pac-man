import sys
with open("/home/nyahyaou/goinfre/pac_man_intra/AI_arena/ghosts/ghost_training.py", "r") as f:
    content = f.read()

target_evaluate = """            predictions = logits.masked_fill(
                ~valid_actions,
                float("-inf"),
            ).argmax(dim=-1)
            correct += (predictions == labels).sum().item()
            prediction_count += labels.numel()"""

replacement_evaluate = """            valid_mask = valid_actions.any(dim=-1)
            predictions = logits.masked_fill(
                ~valid_actions,
                float("-inf"),
            ).argmax(dim=-1)
            correct += ((predictions == labels) & valid_mask).sum().item()
            prediction_count += valid_mask.sum().item()"""

if target_evaluate in content:
    content = content.replace(target_evaluate, replacement_evaluate)
    with open("/home/nyahyaou/goinfre/pac_man_intra/AI_arena/ghosts/ghost_training.py", "w") as f:
        f.write(content)
    print("Success")
else:
    print("Failed to find target_evaluate")
