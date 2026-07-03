import pygame

from ..UI.button import Button


class GameState:

    def enter(self):
        """Called when the state becomes active."""
        pass

    def exit(self):
        """Called when leaving the state."""
        pass

    def update(self, input_state):
        pass

    def draw(self, screen):
        pass


class State:

    def __init__(self, game):
        self.game = game

    def enter(self):
        pass

    def exit(self):
        pass

    def update(self, input_state):
        pass

    def draw(self, screen):
        pass


class StateManager:

    def __init__(self, game):
        self.game = game
        self.current = None

    def change_state(self, state):

        if self.current:
            self.current.exit()

        self.current = state
        self.current.enter()

    def update(self, input_state):
        self.current.update(input_state)

    def draw(self, screen):
        self.current.draw(screen)


class PauseState(State):

    def __init__(self, game):

        super().__init__(game)

        self.resume = Button(...)
        self.home = Button(...)

    def update(self, input_state):

        if self.resume.update(input_state):

            self.game.state_manager.change_state(PlayingState(self.game))

        elif self.home.update(input_state):

            self.game.state_manager.change_state(HomeState(self.game))

    def draw(self, screen):

        self.game.draw_maze(self.game.maze.maze)

        self.resume.draw(screen)
        self.home.draw(screen)


class PlayingState(State):

    def enter(self):

        # Create maze here
        self.game.load_level(6)

    def update(self, input_state):

        if input_state.pause_pressed:

            self.game.state_manager.change_state(PauseState(self.game))

        if input_state.move_left:
            print("Move left")

    def draw(self, screen):

        screen.fill("black")

        self.game.draw_maze(self.game.maze.maze)


class HighScoreState(State):

    def __init__(self, game):

        super().__init__(game)

        self.back = Button(...)

    def update(self, input_state):

        if self.back.update(input_state):

            self.game.state_manager.change_state(HomeState(self.game))

    def draw(self, screen):

        screen.fill("black")

        # Draw scores

        self.back.draw(screen)


class HomeState(State):

    def __init__(self, game):

        super().__init__(game)

        used_font = pygame.font.Font(None, 36)
        self.play_button = Button(
            50,
            100,
            70,
            20,
            "first",
            used_font,
        )
        self.score_button = Button(
            50,
            140,
            70,
            20,
            "second",
            used_font,
        )
        self.quit_button = Button(
            50,
            180,
            70,
            20,
            "thired",
            used_font,
        )

    def update(self, input_state):

        if self.play_button.update(input_state):

            self.game.state_manager.change_state(PlayingState(self.game))

        elif self.score_button.update(input_state):

            self.game.state_manager.change_state(HighScoreState(self.game))

        elif self.quit_button.update(input_state):

            self.game.running = False

    def draw(self, screen):

        screen.fill("black")

        self.play_button.draw(screen)
        self.score_button.draw(screen)
        self.quit_button.draw(screen)


class GameState(GameState):

    def update(self, input_state):

        if input_state.pause_pressed:
            self.manager.change_state(PauseState(self.manager))

        # Update player
        # Update enemies
        # Update maze

    def draw(self, screen):
        pass
