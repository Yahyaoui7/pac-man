"""
sprite_library.py
------------------
Central place that owns every Pac-Man sprite frame + animation timing.

Why a shared library instead of loading frames straight in Player:
  - the PNG frames only need to be loaded and scaled from disk ONCE, no
    matter how many Players/Ghosts end up wanting them
  - Player/Ghost each get their own lightweight `Animation` *playback
    state* (current index/timer) that references the SAME shared Surface
    list -- cheap per-entity, expensive load happens exactly once

Usage from Player (see entity.py):

    self.sprites = SpriteLibrary.instance()
    self.animation = self.sprites.new_animation(PacmanMode.NORMAL)
    ...
    self.animation.update(dt_ms)
    screen.blit(self.animation.current_frame, rect)
"""

from __future__ import annotations

import json
import os
from enum import Enum
from typing import Optional


import pygame

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

ASSET_ROOT = os.path.join(PROJECT_ROOT, "assets")

GHOST_SPRITE_ROOT = os.path.join(ASSET_ROOT, "ghost_sprites")
# Folder that holds normal/, puncher/, kicker/ subfolders produced by
# extract_frames.py (each with frame_00.png, frame_01.png, ... + meta.json)
SPRITE_ROOT = os.path.join(ASSET_ROOT, "pacman_sprites")

# Folder holding the ghost frames + meta.json (color/direction/state map)
GHOST_SPRITE_ROOT = os.path.join(ASSET_ROOT, "ghost_sprites")

# How big the largest side of a sprite frame should be, relative to
# CELL_SIZE. Punch/kick frames include outstretched arms/legs so they're
# naturally bigger than the plain chomp circle -- this keeps them from
# overlapping neighboring cells too aggressively. Tune to taste.
SPRITE_SCALE = 0.9
# Puncher and kicker sprites need to be bigger (outstretched arms/legs)
# so they read clearly as attacks rather than tiny overlays.
SPRITE_SCALE_ATTACK = SPRITE_SCALE * 2
GHOST_SPRITE_SCALE = 1.4


class PacmanMode(Enum):
    """One entry per sprite-sheet folder. The .value must match the
    folder name under SPRITE_ROOT exactly."""

    NORMAL = "normal"
    PUNCH = "puncher"
    KICK = "kicker"


PACMAN_DIRS = {
    PacmanMode.NORMAL: f"{ASSET_ROOT}/pacman_sprites/normal",
    PacmanMode.PUNCH: f"{ASSET_ROOT}/pacman_sprites/puncher",
    PacmanMode.KICK: f"{ASSET_ROOT}/pacman_sprites/kicker",
}


class Facing(Enum):
    """Horizontal facing only -- see the note on Pac-Man/ghost drawing
    for why we deliberately never flip or rotate vertically."""

    LEFT = "l"
    RIGHT = "r"


class GhostColor(Enum):
    RED = "red"  # Blinky
    PINK = "pinc"  # Pinky (folder uses "pinc")
    CYAN = "cyan"  # Inky
    ORANGE = "orange"  # Clyde


class GhostState(Enum):
    HUNT = "hunt"  # normal colored body, chasing
    FRIGHTENED = "frightened"  # blue, edible, running away
    EATEN = "eaten"  # just the eyes, heading home


# Per-mode playback tuning: default ms/frame, whether it loops, and any
# frame indices that should be held longer (impact frames read much
# better on screen with an extra beat -- "hit stop").
_MODE_TIMING = {
    PacmanMode.NORMAL: dict(frame_duration_ms=160, loop=True, overrides={}),
    PacmanMode.PUNCH: dict(frame_duration_ms=70, loop=False, overrides={7: 220}),
    PacmanMode.KICK: dict(frame_duration_ms=80, loop=False, overrides={4: 220}),
}


class Animation:
    """Per-entity playback state over a shared list of Surfaces."""

    def __init__(self, frames, frame_duration_ms=90, loop=True, overrides=None):
        self.frames = frames
        self.default_duration = frame_duration_ms
        self.overrides = overrides or {}
        self.loop = loop
        self.index = 0
        self.timer = 0.0
        self.finished = False

    def reset(self):
        self.index = 0
        self.timer = 0.0
        self.finished = False

    def _duration_for(self, index):
        return self.overrides.get(index, self.default_duration)

    def update(self, dt_ms: float) -> None:
        if self.finished:
            return
        self.timer += dt_ms
        while self.timer >= self._duration_for(self.index):
            self.timer -= self._duration_for(self.index)
            if self.index + 1 < len(self.frames):
                self.index += 1
            else:
                if self.loop:
                    self.index = 0
                else:
                    self.finished = True
                    break

    @property
    def current_frame(self) -> pygame.Surface:
        return self.frames[self.index]


class SpriteLibrary:
    """Singleton cache of scaled Surface frames, keyed by PacmanMode."""

    # Number of walking frames shared by all modes (including NORMAL).
    WALK_FRAME_COUNT = 3

    _instance: Optional["SpriteLibrary"] = None

    def __init__(self):
        self._frames: dict[PacmanMode, list[pygame.Surface]] = {}
        self._walk_frames: dict[PacmanMode, list[pygame.Surface]] = {}
        self._attack_frames: dict[PacmanMode, list[pygame.Surface]] = {}
        self._loaded = False

        # ghost_frames["hunt"][GhostColor][Facing]      -> [Surface, Surface]
        # ghost_frames["frightened"]                     -> [Surface x3]
        # ghost_frames["eaten"]["up"|"down"]              -> [Surface, Surface]
        # ghost_frames["eaten"]["side"][Facing]           -> [Surface, Surface]  (mirrored)
        self._ghost_frames: dict = {}
        self._ghosts_loaded = False

    @classmethod
    def instance(cls) -> "SpriteLibrary":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------------ pacman --
    def load(self, cell_size: int) -> None:
        """Load + scale every Pac-Man mode's frames. Safe to call more
        than once (e.g. if CELL_SIZE changes on a resize) -- it just
        reloads.

        Frames are also split into walk (first WALK_FRAME_COUNT) and
        attack (remaining) subsets for punch/kick powered-mode playback."""
        target = int(cell_size * SPRITE_SCALE)
        target_attack = int(cell_size * SPRITE_SCALE_ATTACK)
        self._frames = {
            PacmanMode.NORMAL: self._load_mode(PacmanMode.NORMAL, target),
            PacmanMode.PUNCH: self._load_mode(PacmanMode.PUNCH, target_attack),
            PacmanMode.KICK: self._load_mode(PacmanMode.KICK, target_attack),
        }
        for mode in PacmanMode:
            frames = self._frames[mode]
            self._walk_frames[mode] = frames[: self.WALK_FRAME_COUNT]
            self._attack_frames[mode] = frames[self.WALK_FRAME_COUNT :]
        self._loaded = True

    def _load_mode(self, mode: PacmanMode, target_size: int) -> list[pygame.Surface]:
        folder = PACMAN_DIRS[mode]

        with open(os.path.join(folder, "meta.json")) as f:
            meta = json.load(f)

        frames = []

        for filename in meta["frames"]:
            path = os.path.join(folder, os.path.basename(filename))
            raw = pygame.image.load(path).convert_alpha()
            frames.append(self._scale_to(raw, target_size))

        return frames

    def new_animation(self, mode: PacmanMode) -> Animation:
        """Factory for a fresh, independent playback state. Call this
        once per entity per mode-switch (not once per frame)."""
        if not self._loaded:
            raise RuntimeError(
                "SpriteLibrary.load(cell_size) must be called once at "
                "startup before requesting animations."
            )
        timing = _MODE_TIMING[mode]
        return Animation(
            self._frames[mode],
            frame_duration_ms=timing["frame_duration_ms"],
            loop=timing["loop"],
            overrides=timing["overrides"],
        )

    def new_walk_animation(self, mode: PacmanMode) -> Animation:
        """Looping walk animation using the first WALK_FRAME_COUNT frames."""
        timing = _MODE_TIMING[mode]
        return Animation(
            self._walk_frames[mode],
            frame_duration_ms=timing["frame_duration_ms"],
            loop=True,
            overrides={},
        )

    def new_attack_animation(self, mode: PacmanMode) -> Animation:
        """One-shot attack animation using frames after the walk subset."""
        timing = _MODE_TIMING[mode]
        # Re-key overrides to account for the removed walk-frame prefix
        rekeyed = {}
        for idx, dur in timing["overrides"].items():
            new_idx = idx - self.WALK_FRAME_COUNT
            if new_idx >= 0:
                rekeyed[new_idx] = dur
        return Animation(
            self._attack_frames[mode],
            frame_duration_ms=timing["frame_duration_ms"],
            loop=False,
            overrides=rekeyed,
        )

    # ------------------------------------------------------------- ghosts --
    def load_ghosts(self, cell_size: int) -> None:
        """Load + scale every ghost color/direction/state frame. Reads
        assets/ghost_sprites/meta.json (rebuilt by rebuild_ghost_meta.py
        if it's ever lost/corrupted again)."""
        target = int(cell_size * GHOST_SPRITE_SCALE)
        meta_path = os.path.join(GHOST_SPRITE_ROOT, "meta.json")
        with open(meta_path) as f:
            meta = json.load(f)

        def load_list(filenames):
            out = []
            for name in filenames:
                raw = pygame.image.load(
                    os.path.join(GHOST_SPRITE_ROOT, name)
                ).convert_alpha()
                out.append(self._scale_to(raw, target))
            return out

        hunt = {}
        for color in GhostColor:
            entry = meta["colors"][color.value]
            hunt[color] = {
                Facing.LEFT: load_list(entry["l"]),
                Facing.RIGHT: load_list(entry["r"]),
            }

        frightened = load_list(meta["frightened"])

        eaten_side_r = load_list(meta["eaten"]["side"])
        eaten_side_l = [pygame.transform.flip(s, True, False) for s in eaten_side_r]

        eaten = {
            "up": load_list(meta["eaten"]["up"]),
            "down": load_list(meta["eaten"]["down"]),
            "side": {Facing.RIGHT: eaten_side_r, Facing.LEFT: eaten_side_l},
        }

        self._ghost_frames = {"hunt": hunt, "frightened": frightened, "eaten": eaten}
        self._ghosts_loaded = True

    def new_ghost_animation(
        self,
        state: GhostState,
        color: GhostColor = None,
        facing: Facing = Facing.RIGHT,
        vertical: str = None,
    ) -> Animation:
        """
        state:    HUNT / FRIGHTENED / EATEN
        color:    required for HUNT (which colored ghost)
        facing:   LEFT/RIGHT, used for HUNT and EATEN-side
        vertical: "up" or "down", used only for EATEN when moving vertically
                  (overrides `facing` since eyes have dedicated up/down art)
        """
        if not self._ghosts_loaded:
            raise RuntimeError(
                "SpriteLibrary.load_ghosts(cell_size) must be called once "
                "at startup before requesting ghost animations."
            )

        if state == GhostState.HUNT:
            if color is None:
                raise ValueError("GhostState.HUNT requires a color")
            frames = self._ghost_frames["hunt"][color][facing]
            return Animation(frames, frame_duration_ms=180, loop=True)

        if state == GhostState.FRIGHTENED:
            frames = self._ghost_frames["frightened"]
            return Animation(frames, frame_duration_ms=150, loop=True)

        if state == GhostState.EATEN:
            if vertical in ("up", "down"):
                frames = self._ghost_frames["eaten"][vertical]
            else:
                frames = self._ghost_frames["eaten"]["side"][facing]
            return Animation(frames, frame_duration_ms=250, loop=True)

        raise ValueError(f"Unknown ghost state: {state}")

    # --------------------------------------------------------------- util --
    @staticmethod
    def _scale_to(raw: pygame.Surface, target_size: int) -> pygame.Surface:
        scale = target_size / max(raw.get_width(), raw.get_height())
        new_size = (
            max(1, int(raw.get_width() * scale)),
            max(1, int(raw.get_height() * scale)),
        )
        return pygame.transform.smoothscale(raw, new_size)
