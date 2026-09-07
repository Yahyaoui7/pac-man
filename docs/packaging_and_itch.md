# Project Packaging & Itch.io Deployment

Per Chapter VII of the subject, the game is packaged into a self-contained, standalone distribution ready for distribution on platforms such as Itch.io.

---

## 1. Zero-PyTorch Lightweight Bundle

Standard deep learning frameworks like PyTorch result in binaries exceeding 1.5–2 GB. To achieve a lightweight, fast-loading bundle without losing neural capability:
- The inference engine runs exclusively on **ONNX Runtime** (`onnxruntime`).
- The neural network weights are packaged as an optimized **FP32 ONNX model** (`AI_arena/models/player_model.onnx`, 19.5 MB).
- Observation pipelines were authored in **pure NumPy** with zero PyTorch runtime dependencies.

---

## 2. Packaging Workflow

The packaging is automated via **PyInstaller**:

```bash
uv run pyinstaller --noconfirm PacMan_AI.spec
```

The resulting distribution is located in `dist/PacMan_AI/`:
```
dist/PacMan_AI/
├── PacMan_AI           # Native Linux executable binary
├── config.json         # User-editable configuration file
└── _internal/          # Bundled assets, Python runtime, ONNX model
    ├── assets/
    │   ├── ghost_sprites/
    │   ├── pacman_sprites/
    │   └── sounds/
    └── AI_arena/
        └── models/
            └── player_model.onnx
```

---

## 3. Creating the Release Archive for Itch.io

To generate the single ZIP upload file for Itch.io:

```bash
cd dist
zip -r PacMan_AI_Linux.zip PacMan_AI
```

### Resulting Artifact:
- **Archive**: `dist/PacMan_AI_Linux.zip`
- **Total Compressed Size**: **~82 MB** (well within standard indie game upload limits).
- **In-Game Documentation**: Config file and control cheats are bundled directly inside the archive.
