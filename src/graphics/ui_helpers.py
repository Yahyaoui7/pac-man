"""Shared UI drawing helpers: fonts, colors, overlays, text, buttons."""

import pygame

from src.UI.button import Button

# ---------------------------------------------------------------------------
# Color constants
# ---------------------------------------------------------------------------
COLOR_NEON_YELLOW = (255, 238, 0)
COLOR_NEON_CYAN = (0, 238, 255)
COLOR_BG_DARK = (5, 5, 10)
COLOR_BG_PANEL = (10, 10, 20)
COLOR_PELLET = (255, 184, 151)
COLOR_RED = (255, 0, 0)
COLOR_WHITE = (255, 255, 255)
COLOR_GREEN = (0, 255, 0)

# ---------------------------------------------------------------------------
# Font constants — lazily created after pygame.init()
# ---------------------------------------------------------------------------
_fonts: dict[int, pygame.font.Font] = {}


def _get_font(size: int) -> pygame.font.Font:
    if size not in _fonts:
        _fonts[size] = pygame.font.Font(None, size)
    return _fonts[size]


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


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

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


def make_button_centered(
    screen: pygame.Surface,
    label: str,
    y: int,
    font: pygame.font.Font | None = None,
    width: int = 200,
    height: int = 45,
) -> Button:
    """Create a Button centered horizontally on screen."""
    if font is None:
        font = FONT_BTN
    cx = screen.get_width() // 2
    return Button(cx - width // 2, y, width, height, label, font)
