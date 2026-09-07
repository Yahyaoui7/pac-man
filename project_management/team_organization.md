# Team Organization

## Team Members

| Member | Intra Login | Primary Responsibilities |
|---|---|---|
| **Mouad Mennioui** | mmenniou | Game Engine, UI/UX, Rendering, Maze Integration, RL Training Pipeline |
| **Nabil Yahyaoui Idrissi** | nyahyaou | Ghosts AI, Player Logic, Sound/Score Systems, CNN/SL Models, Ghost Neural Model |

## Contribution Summary (from Git)

- **Mouad Mennioui**: ~220 commits — Led the game engine architecture (`game_loop.py`), state management (Home, Pause, GameOver, Victory screens), UI rendering, cheat modes, entity system, pellet/collision integration, maze generation with A-Maze-ing, keyboard/mouse input manager, custom sprite animations, and the RL training pipeline for the Pac-Man player model.
- **Nabil Yahyaoui Idrissi**: ~100 commits — Led the ghost AI system (movement, chase/flee BFS, edible mode, respawn logic), player movement and wall collision, score management, highscore persistence (save/load/validate), sound management, CNN data collection, supervised learning (SL) player model training, and the adversarial ghost neural model (SL_ghosts_model branch).

## How We Split the Work

The project was tracked using **Jira Kanban** with tasks tagged by category (e.g., `[Player]`, `[Ghosts]`, `[UI]`, `[Maze system]`). Tasks were assigned on Jira and each member worked on their assigned tickets. The split was roughly:

### Mouad Mennioui
- **Engine & Infrastructure:** Project structure setup, game loop, level system, timer, pause/resume, input manager (`SCRUM-30` to `SCRUM-33`, `SCRUM-81` to `SCRUM-85`)
- **Maze System:** A-Maze-ing integration, random level generation, error handling (`SCRUM-27` to `SCRUM-29`)
- **UI/UX:** Main menu, pause menu, game over screen, animations, sprite creation, button manager (`SCRUM-54`, `SCRUM-56`, `SCRUM-57`, `SCRUM-98` to `SCRUM-106`)
- **Game Mechanics:** Entity system, pellet eating, ghost catching, collision system, cheat modes, leveling (`SCRUM-87`, `SCRUM-88`, `SCRUM-93`, `SCRUM-95`)
- **AI Training:** RL training pipeline setup, reward system tuning, CNN architecture, model checkpoint management

### Nabil Yahyaoui Idrissi
- **Player System:** Pac-Man creation, WASD/arrow movement, wall collision prevention (`SCRUM-34` to `SCRUM-36`)
- **Ghost AI:** 4 ghost creation, movement logic, BFS chase behavior, edible/flee mode, respawn after being eaten, ghost eyes animation (`SCRUM-39` to `SCRUM-42`, `SCRUM-109`, `SCRUM-113`)
- **Score & Highscore:** Score tracking (pacgums, super-pacgums, ghosts), highscore JSON persistence, name validation, top-10 list (`SCRUM-46`, `SCRUM-48` to `SCRUM-53`)
- **Sound:** Sound integration across game states (menu, gameplay, ghost eating, game over, pause) (`SCRUM-89`)
- **AI Models:** CNN data collection pipeline, SL player model training, adversarial ghost model on dedicated branch (`SL_ghosts_model`)

## How Decisions Were Made

- **Daily coordination at 1337 campus:** The team worked side by side at the 1337 coding school (visible from Git commit emails: `@c1r1p5.1337.ma`, `@c2r9p2.1337.ma`, etc.), enabling rapid in-person decision making.
- **Jira tickets for task tracking:** All features and bugs were tracked as Jira issues with descriptions and acceptance criteria.
- **Git branches for parallel work:** Major features were developed on separate branches (`Player_RL`, `SL_ghosts_model`, `polished-game`, `cnn_collect_data`) to avoid conflicts, then merged into master.

## How Issues Were Handled

- **Bugs were tracked on Jira:** Issues like collision detection bugs (`SCRUM-93`, `SCRUM-96`, `SCRUM-105`, `SCRUM-106`), movement bugs (`SCRUM-94`), and ghost respawn bugs (`SCRUM-107`, `SCRUM-113`) were logged as `[BUG]` tickets, assigned, and resolved.
- **Merge conflicts were resolved collaboratively:** Git history shows multiple merge commits where Mouad and Nabil resolved conflicts together (e.g., `merging the bug fix`, `conflict the merge`, `fixing the merge conflicts`).
