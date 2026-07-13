"""Shared UI drawing helpers: fonts, colors, overlays, text, buttons."""

import pygame

COLOR_NEON_YELLOW = (255, 238, 0)
COLOR_NEON_CYAN = (0, 238, 255)
COLOR_BG_DARK = (5, 5, 10)
COLOR_BG_PANEL = (10, 10, 20)
COLOR_PELLET = (255, 184, 151)
COLOR_RED = (255, 0, 0)
COLOR_WHITE = (255, 255, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_DIM_CYAN = (0, 70, 80)
COLOR_HUD_TOP = (16, 16, 30)
COLOR_HUD_BOTTOM = (8, 8, 16)

fonts: dict[int, pygame.font.Font] = {}


def _get_font(size: int) -> pygame.font.Font:
    if size not in fonts:
        fonts[size] = pygame.font.Font(None, size)
    return fonts[size]


def get_scaled_font(
    text: str,
    max_width: int,
    base_size: int = 28,
    min_size: int = 14,
    step: int = 2,
) -> pygame.font.Font:
    """Return the largest cached font (<= base_size, >= min_size) that renders
    *text* within *max_width* pixels. Prevents HUD text from overlapping on
    narrow/small windows."""
    size = base_size
    font = _get_font(size)
    while size > min_size and font.size(text)[0] > max_width:
        size -= step
        font = _get_font(size)
    return font


def _font_attr(name: str, size: int) -> pygame.font.Font:
    """Return a cached font; also set it as a module-level attr on first call."""
    f = _get_font(size)
    globals()[name] = f
    return f


FONT_TITLE = None
FONT_TITLE_LARGE = None
FONT_BTN = None
FONT_TEXT = None
FONT_HUD = None
FONT_SCORE = None
FONT_INPUT = None
FONT_LOSING = None


def _init_fonts():
    """Call once after pygame.init() to populate font constants."""
    global FONT_TITLE, FONT_TITLE_LARGE, FONT_BTN, FONT_TEXT
    global FONT_HUD, FONT_SCORE, FONT_INPUT, FONT_LOSING
    FONT_TITLE = _get_font(48)
    FONT_TITLE_LARGE = _get_font(64)
    FONT_BTN = _get_font(36)
    FONT_TEXT = _get_font(24)
    FONT_HUD = _get_font(28)
    FONT_SCORE = _get_font(32)
    FONT_INPUT = _get_font(36)
    FONT_LOSING = _get_font(42)


def draw_overlay(screen: pygame.Surface, alpha: int = 150) -> None:
    """Draw a semi-transparent dark overlay over the full screen."""
    overlay = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, alpha))
    screen.blit(overlay, (0, 0))


def draw_text_centered(
    screen: pygame.Surface,
    font: pygame.font.Font,
    text: str,
    y: int,
    color=COLOR_WHITE,
) -> pygame.Rect:
    """Render text centered horizontally at *y*, blit it, return rect."""
    surf = font.render(text, True, color)
    rect = surf.get_rect(center=(screen.get_width() // 2, y))
    screen.blit(surf, rect)
    return rect
