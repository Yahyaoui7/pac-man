import pygame


def now() -> int:
    """Current time in milliseconds since pygame started."""
    return pygame.time.get_ticks()


def after(ms: int) -> int:
    """Returns the timestamp when a timer should expire."""
    return now() + ms


def expired(end_time: int) -> bool:
    """True if the timer has expired."""
    return now() >= end_time
