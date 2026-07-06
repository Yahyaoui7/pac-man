import pygame


class SoundManager:
    def __init__(self):
        pygame.mixer.init()

        # ---------- Sound Effects ----------
        # self.sounds = {
        #     "chomp": {
        #         "sound": pygame.mixer.Sound(
        #             "src/sounds/merhba-biiiik.mp3"
        #         ),
        #         "volume": 0.5,
        #     },
        #     # "eat_ghost": {
        #     #     "sound": pygame.mixer.Sound("assets/sounds/eat_ghost.wav"),
        #     #     "volume": 0.7,
        #     # },
        #     "eat_fruit": {
        #         "sound": pygame.mixer.Sound("src/sounds/pacman_eatfruit.wav"),
        #         "volume": 0.7,
        #     },
        #     # "power_pellet": {
        #     #     "sound": pygame.mixer.Sound("assets/sounds/power_pellet.wav"),
        #     #     "volume": 0.7,
        #     # },
        #     "death": {
        #         "sound": pygame.mixer.Sound("src/sounds/pacman_death.wav"),
        #         "volume": 0.8,
        #     },
        #     "extra_life": {
        #         "sound": pygame.mixer.Sound("src/sounds/pacman_extrapac.wav"),
        #         "volume": 0.8,
        #     },
        #     # "victory": {
        #     #     "sound": pygame.mixer.Sound("assets/sounds/victory.wav"),
        #     #     "volume": 0.8,
        #     # },
        # }

        # Apply volumes
        # for data in self.sounds.values():
        #     data["sound"].set_volume(data["volume"])

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

    def play_sound(self, name):
        if name in self.sounds:
            self.sounds[name]["sound"].play()

    def play_music(self, name, loop=True):
        if name not in self.music:
            return

        pygame.mixer.music.load(self.music[name]["file"])
        pygame.mixer.music.set_volume(self.music[name]["volume"])
        pygame.mixer.music.play(-1 if loop else 0)

    def stop_music(self):
        pygame.mixer.music.stop()

    def pause_music(self):
        pygame.mixer.music.pause()

    def resume_music(self):
        pygame.mixer.music.unpause()
