import json


class ScoreManager:
    """Manage player score."""

    def __init__(self, config) -> None:
        self.score = 0

        self.normal_pellet_points = config.points_per_pacgum
        self.super_pacgum_points = config.points_per_super_pacgum
        self.ghost_points = config.points_per_ghost

    def reset(self) -> None:
        """Reset score when starting new game."""
        self.score = 0

    def add_normal_pellet(self) -> None:
        """Add score when player eats normal pellet."""
        self.score += self.normal_pellet_points

    def add_super_pacgum(self) -> None:
        """Add score when player eats super pacgum."""
        self.score += self.super_pacgum_points

    def add_ghost(self) -> None:
        """Add score when player eats edible ghost."""
        self.score += self.ghost_points

    def add_time_bonus(self, time):
        self.score += time

    def get_score(self) -> int:
        """Return current score."""
        return self.score


class HighScoreManager:
    def __inti__(self, file_path):
        self.file_path = file_path
        self.highscores = []
        self.max_scores = 10

    def load_scores(self):
        try:
            with open(self.file_path, "r", encoding="utf-8") as file:
                self.highscores = json.load(file)
        except FileNotFoundError:
            self.highscores = []
        except json.JSONDecodeError:
            self.highscores = []

    def save_score(self):
        with open(self.file_path, "w") as file:
            json.dump(self.highscores, file)
