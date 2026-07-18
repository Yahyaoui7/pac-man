"""Implements the State Pattern for game screens and UI rendering."""

import pygame
from typing import Any, List


class State:
    """Base class for all screen states."""

    def __init__(self, game: Any) -> None:
        self.game = game

    def enter(self) -> None:
        pass

    def exit(self) -> None:
        pass

    def update(
        self,
        input_state: Any,
        events: List[pygame.event.Event],
    ) -> None:
        pass

    def draw(self, screen: pygame.Surface) -> None:
        pass


class StateManager:
    """Manages switching and updating the active screen state."""

    def __init__(self, game: Any) -> None:
        self.game = game
        self.stack: list[State] = []

    @property
    def current(self) -> State | None:
        if self.stack:
            return self.stack[-1]
        return None

    def change_state(self, state: State) -> None:
        while self.stack:
            self.stack.pop().exit()
        self.stack.append(state)
        state.enter()

    def push_state(self, state: State) -> None:
        self.stack.append(state)
        state.enter()

    def pop_state(self) -> None:
        if not self.stack:
            return
        state = self.stack.pop()
        state.exit()

    def update(
        self, input_state: Any, events: list[pygame.event.Event]
    ) -> None:
        if self.current:
            self.current.update(input_state, events)

    def draw(self, screen: pygame.Surface) -> None:
        if self.current:
            self.current.draw(screen)
