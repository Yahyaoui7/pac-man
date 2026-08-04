uv run python -u -m AI_arena.player.player_training \
    --stage 1 \
    --num-updates 4000 \
    --save-interval 50 \
    2>&1 | awk '/^Upd/ { print >> "RL_logs.txt"; fflush("RL_logs.txt") } { print; fflush() }'

    