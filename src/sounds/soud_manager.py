import pygame


class SoundManager:
    def __init__(self):
        pygame.mixer.init()
        self.normal_music_volume = 0.4
        self.duck_music_volume = 0.1
        self.duck_music_timer = 0.0
        self.duck_music_duration = 0.3
        self.current_music_volume = self.normal_music_volume
        # ---------- Sound Effects ----------

        self.sounds = {
            # Pacgum / food
            "eat_normal_pellet": {
                "sound": pygame.mixer.Sound("src/sounds/munch_pac_man.mp3"),
                "volume": 0.3,
            },
            "eat_super_pacgum": {
                "sound": pygame.mixer.Sound(
                    "src/sounds/mshyn-fyh-y-wldyk.mp3"
                ),
                "volume": 0.6,
            },
            # Ghost / collision
            "eat_ghost": {
                "sound": pygame.mixer.Sound("src/sounds/fahhh_KcgAXfs.mp3"),
                "volume": 0.7,
            },
            "player_death": {
                "sound": pygame.mixer.Sound("src/sounds/ayeh-ayeh-ayeh.mp3"),
                "volume": 0.8,
            },
            # Game states
            "level_complete": {
                "sound": pygame.mixer.Sound("src/sounds/omgwow.mp3"),
                "volume": 0.7,
            },
            "victory": {
                "sound": pygame.mixer.Sound(
                    "src/sounds/anime-wow-sound-effect.mp3"
                ),
                "volume": 0.7,
            },
            "game_over": {
                "sound": pygame.mixer.Sound(
                    "src/sounds/man-screaming-aaaah.mp3"
                ),
                "volume": 0.8,
            },
            # UI / menu
            "menu_select": {
                "sound": pygame.mixer.Sound("src/sounds/pop_7e9Is8L.mp3"),
                "volume": 0.4,
            },
            "menu_confirm": {
                "sound": pygame.mixer.Sound(
                    "src/sounds/punch-gaming-sound-effect-hd_RzlG1GE.mp3"
                ),
                "volume": 0.5,
            },
            "pause": {
                "sound": pygame.mixer.Sound("src/sounds/w9af-3and-hadek.mp3"),
                "volume": 0.5,
            },
        }
        # Apply volumes
        for data in self.sounds.values():
            data["sound"].set_volume(data["volume"])

        self.music = {
            # Menus
            "menu_intro": {
                "file": "src/sounds/merhba-biiiik.mp3",
                "volume": 0.3,
            },
            "menu_music": {
                "file": "src/sounds/pacmantng.mp3",
                "volume": 0.3,
            },
            # Game
            "game_intro": {
                "file": "src/sounds/lslm-lykm-b-d-hy-lwl.mp3",
                "volume": 0.4,
            },
            "game_music": {
                "file": "src/sounds/hzym-lr-d_3Kc3wxM.mp3",
                "volume": 0.15,
            },
            # End states
            "game_over_music": {
                "file": "src/sounds/n-llh-wd-hmd.mp3",
                "volume": 0.4,
            },
            "victory_music": {
                "file": "src/sounds/b9afiya-lhal.mp3",
                "volume": 0.4,
            },
        }

    def play_sound(self, name: str):
        if name in self.sounds:
            return self.sounds[name]["sound"].play(0)
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

    # def update(self, dt: float) -> None:
    #     if self.duck_music_timer > 0:
    #         self.duck_music_timer -= dt

    #         if self.duck_music_timer <= 0:
    #             pygame.mixer.music.set_volume(self.current_music_volume)
