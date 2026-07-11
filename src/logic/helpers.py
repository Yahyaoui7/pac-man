"""Coordinate and geometry helpers for grid/pixel/screen conversions."""

from src.logic.config import CELL_SIZE, PADDING, TOP_BAR_HEIGHT


def grid_to_pixel(row: int, col: int) -> tuple[int, int]:
    """Convert grid cell to pixel-space center of that cell."""
    x = col * CELL_SIZE + CELL_SIZE // 2
    y = row * CELL_SIZE + CELL_SIZE // 2
    return x, y


def pixel_to_screen(px: int, py: int) -> tuple[int, int]:
    """Convert entity pixel position to screen position (adds padding + top bar)."""
    sx = PADDING // 2 + px
    sy = TOP_BAR_HEIGHT + PADDING // 2 + py
    return sx, sy


def cell_to_screen(row: int, col: int) -> tuple[int, int]:
    """Convert grid cell to screen corner position (for maze wall drawing)."""
    x = PADDING // 2 + col * CELL_SIZE
    y = PADDING // 2 + row * CELL_SIZE + TOP_BAR_HEIGHT
    return x, y


def pellet_screen_pos(col: int, row: int) -> tuple[int, int]:
    """Convert grid cell to screen center (for pellet drawing)."""
    x = PADDING // 2 + col * CELL_SIZE + CELL_SIZE // 2
    y = PADDING // 2 + row * CELL_SIZE + CELL_SIZE // 2 + TOP_BAR_HEIGHT
    return int(x), int(y)


def screen_center(width: int, height: int) -> tuple[int, int]:
    """Return (center_x, center_y) of the screen."""
    return width // 2, height // 2
