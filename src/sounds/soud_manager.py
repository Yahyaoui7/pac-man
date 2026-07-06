import pygame


class SoundManager:
    def __init__(self):
        pygame.mixer.init()
        self.normal_music_volume = 0.4
        self.duck_music_volume = 0.15
        self.duck_music_timer = 0.0
        self.duck_music_duration = 0.3

        # ---------- Sound Effects ----------
        self.sounds = {
            "eat_normal_pellet": {
                "sound": pygame.mixer.Sound("src/sounds/fahhh_KcgAXfs.mp3"),
                "volume": 0.3,
            },
            # "eat_ghost": {
            #     "sound": pygame.mixer.Sound("assets/sounds/eat_ghost.wav"),
            #     "volume": 0.7,
            # },
            "Super-pacgum": {
                "sound": pygame.mixer.Sound(
                    "src/sounds/i-got-this-fahhhh.mp3"
                ),
                "volume": 0.7,
            },
            # # "power_pellet": {
            # #     "sound": pygame.mixer.Sound("assets/sounds/power_pellet.wav"),
            # #     "volume": 0.7,
            # # },
            # "death": {
            #     "sound": pygame.mixer.Sound("src/sounds/pacman_death.wav"),
            #     "volume": 0.8,
            # },
            # "extra_life": {
            #     "sound": pygame.mixer.Sound("src/sounds/pacman_extrapac.wav"),
            #     "volume": 0.8,
            # },
            # # "victory": {
            # #     "sound": pygame.mixer.Sound("assets/sounds/victory.wav"),
            # #     "volume": 0.8,
            # # },
        }

        # Apply volumes
        for data in self.sounds.values():
            data["sound"].set_volume(data["volume"])

        # ---------- Music ----------
        self.music = {
            "menu_intro": {
                "file": "src/sounds/merhba-biiiik.mp3",
                "volume": 0.3,
            },
            "menu_music": {
                "file": "src/sounds/pacmantng.mp3",
                "volume": 0.3,
            },
            "game_intro": {
                "file": "src/sounds/lslm-lykm-b-d-hy-lwl.mp3",
                "volume": 0.4,
            },
            "game_music": {
                "file": "src/sounds/hzym-lr-d_3Kc3wxM.mp3",
                "volume": 0.4,
            },
            # "game_over": {
            #     "file": "assets/music/game_over.ogg",
            #     "volume": 0.5,
            # },
        }

    def play_sound(self, name: str):
        if name in self.sounds:
            return self.sounds[name]["sound"].play()
        return None

    def play_music(self, name, loop=True):
        if name not in self.music:
            return

        pygame.mixer.music.load(self.music[name]["file"])
        self.current_music_volume = self.music[name]["volume"]
        pygame.mixer.music.set_volume(self.music[name]["volume"])
        pygame.mixer.music.play(-1 if loop else 0)

    def stop_music(self):
        pygame.mixer.music.stop()

    def pause_music(self):
        pygame.mixer.music.pause()

    def resume_music(self):
        pygame.mixer.music.unpause()

    def play_sound_with_duck(self, name: str) -> None:
        if pygame.mixer.music.get_busy():
            pygame.mixer.music.set_volume(self.duck_music_volume)

        self.play_sound(name)
        self.duck_music_timer = self.duck_music_duration

    def update(self, dt: float) -> None:
        if self.duck_music_timer > 0:
            self.duck_music_timer -= dt

            if self.duck_music_timer <= 0:
                pygame.mixer.music.set_volume(self.current_music_volume)
