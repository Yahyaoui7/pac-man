import sys
with open("/home/nyahyaou/goinfre/pac_man_intra/AI_arena/ghosts/ghost_training.py", "r") as f:
    content = f.read()

target_loss = """    masked_logits = logits.masked_fill(~safe_mask, float("-inf"))
    return nn.functional.cross_entropy(
        masked_logits.reshape(-1, ACTION_COUNT),
        labels.reshape(-1),
    )"""

replacement_loss = """    masked_logits = logits.masked_fill(~safe_mask, float("-inf"))
    loss = nn.functional.cross_entropy(
        masked_logits.reshape(-1, ACTION_COUNT),
        labels.reshape(-1),
        reduction="none",
    )
    valid_ghosts = ~all_masked.reshape(-1)
    if valid_ghosts.any():
        return (loss * valid_ghosts.float()).sum() / valid_ghosts.float().sum()
    return loss.sum() * 0.0"""

target_train = """                predictions = logits.masked_fill(
                    ~valid_actions,
                    float("-inf"),
                ).argmax(dim=-1)
                correct += (predictions == labels).sum().item()
                prediction_count += labels.numel()"""

replacement_train = """                valid_mask = valid_actions.any(dim=-1)
                predictions = logits.masked_fill(
                    ~valid_actions,
                    float("-inf"),
                ).argmax(dim=-1)
                correct += ((predictions == labels) & valid_mask).sum().item()
                prediction_count += valid_mask.sum().item()"""

if target_loss in content:
    content = content.replace(target_loss, replacement_loss)
else:
    print("Failed to find target_loss")
    sys.exit(1)

if target_train in content:
    content = content.replace(target_train, replacement_train)
else:
    print("Failed to find target_train")
    sys.exit(1)

with open("/home/nyahyaou/goinfre/pac_man_intra/AI_arena/ghosts/ghost_training.py", "w") as f:
    f.write(content)
print("Success")
