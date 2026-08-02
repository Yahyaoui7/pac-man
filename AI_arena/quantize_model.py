"""Quantize the GhostCNN model for faster CPU inference.

Applies dynamic INT8 quantization to Linear layers and wraps in TorchScript.
Produces ghost_ai_quantized.pt (~30x faster than FP32 on CPU).
"""

from __future__ import annotations

import torch
import torch.ao.quantization as quant

from AI_arena.models.cnn_ghost import GhostCNN

WEIGHTS_PATH = "AI_arena/models/ghost_ai.pt"
OUTPUT_PATH = "AI_arena/models/ghost_ai_quantized.pt"


def quantize(source: str = WEIGHTS_PATH, dest: str = OUTPUT_PATH) -> None:
    device = torch.device("cpu")
    model = GhostCNN().to(device)
    model.load_state_dict(torch.load(source, map_location=device, weights_only=True))
    model.eval()

    # Dynamic INT8 quantization (Linear layers only — Conv2d stays FP32)
    quantized = quant.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    quantized.eval()

    # TorchScript trace eliminates Python overhead
    grid_example = torch.randn(1, 12, 50, 25)
    extra_example = torch.randn(1, 37)
    traced = torch.jit.trace(quantized, (grid_example, extra_example))
    traced.eval()

    traced.save(dest)
    print(f"Quantized model saved to {dest}")


if __name__ == "__main__":
    quantize()
