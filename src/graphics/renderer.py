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


class StateManager:

    def __init__(self):
        self.states = []

    def push_state(self, state):
        self.states.append(state)

    def pop_state(self):
        self.states.pop()

    @property
    def current(self):
        return self.states[-1]

    def update(self, input_state):
        self.current.update(input_state)

    def draw(self, screen):
        self.current.draw(screen)


class PauseState(GameState):

    def __init__(self, manager):

        self.manager = manager

        self.resume_button = Button(...)
        self.home_button = Button(...)

    def update(self, input_state):

        if self.resume_button.update(input_state):
            self.manager.pop_state()

        elif self.home_button.update(input_state):
            self.manager.change_state(HomeState(self.manager))


class HomeState(GameState):

    def __init__(self, manager):
        self.manager = manager

        self.play_button = Button(...)
        self.score_button = Button(...)
        self.quit_button = Button(...)

    def update(self, input_state):

        if self.play_button.update(input_state):
            self.manager.change_state(GameState(self.manager))

        elif self.score_button.update(input_state):
            self.manager.change_state(HighScoreState(self.manager))

        elif self.quit_button.update(input_state):
            self.manager.running = False

    def draw(self, screen):
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
        # Draw maze
        # Draw player