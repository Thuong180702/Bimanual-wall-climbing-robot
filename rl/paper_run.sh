#!/usr/bin/env bash
# One of the paper's training runs, by its name alone. The name is the run's role in
# rl.paper_eval and paper/figsrc/tables.py, so it is also what sets the run's settings here:
# typed by hand beside the name, one forgotten variable is a day of GPU on the wrong run.
#
#     mkdir -p out && nohup bash rl/paper_run.sh p_s0 > out/p_s0.console 2>&1 &
#     DRY=1 bash rl/paper_run.sh a_rfcl_s1          # print what the name means, run nothing
#
#   p_s<seed>              the method, 1.5B steps (seeds 0..4)
#   a_none_s<seed>         the method under the ablations' budget, 0.75B (seeds 0..2)
#   a_plain_s<seed>        PPO alone: every family at its last levels, from the hang
#   a_<ablation>_s<seed>   rsi frontier rfcl clone cloneall curriculum taught route hold
#   b<family>_s<seed>      family 1..7's specialist
#   w_<weight><05|2>_s<seed>   K_BANK, K_REACH or K_SLIDE x0.5 or x2: w_bank05_s0, w_slide2_s0
#   x_<rule>               none oneway world grab refund route hold: families 1-3, 100M steps
#
# Every run: families 1-7 (a specialist its own, an exploit run 1-3), dynamics randomization,
# and no scoring on the GPU box (rl.paper_eval scores every run, on CPUs). HOURS is only a cap
# -- TOTAL ends the run -- and a second call with the same name carries the run on.
set -euo pipefail
NAME=${1:?usage: bash rl/paper_run.sh <run name> [<run name>.tgz, packed on another box]}
FAMILIES=1-7 TOTAL=750000000 SEED=0 ABLATE= WEIGHTS= WALLS=adr
A='rsi|frontier|rfcl|clone|cloneall|curriculum|taught|route|hold'
if   [[ $NAME =~ ^p_s([0-9]+)$ ]];            then SEED=${BASH_REMATCH[1]} TOTAL=1500000000
elif [[ $NAME =~ ^a_none_s([0-9]+)$ ]];       then SEED=${BASH_REMATCH[1]}
elif [[ $NAME =~ ^a_plain_s([0-9]+)$ ]];      then SEED=${BASH_REMATCH[1]} WALLS=random
elif [[ $NAME =~ ^a_($A)_s([0-9]+)$ ]];       then ABLATE=${BASH_REMATCH[1]} SEED=${BASH_REMATCH[2]}
elif [[ $NAME =~ ^b([1-7])_s([0-9]+)$ ]];     then FAMILIES=${BASH_REMATCH[1]} SEED=${BASH_REMATCH[2]}
elif [[ $NAME =~ ^w_(bank|reach|slide)(05|2)_s([0-9]+)$ ]]; then
    SEED=${BASH_REMATCH[3]}
    WEIGHTS="K_$(tr a-z A-Z <<< "${BASH_REMATCH[1]}")=$([ "${BASH_REMATCH[2]}" = 05 ] && echo 0.5 || echo 2)"
elif [[ $NAME =~ ^x_(none|oneway|world|grab|refund|route|hold)$ ]]; then
    FAMILIES=1-3 TOTAL=100000000
    [ "${BASH_REMATCH[1]}" = none ] || ABLATE=${BASH_REMATCH[1]}
else
    echo "$NAME is not one of the paper's run names:"; sed -n '9,15p' "$0"; exit 1
fi
export FAMILIES TOTAL SEED ABLATE WEIGHTS WALLS SCORE=0 DR=1 HOURS=${HOURS:-72}
echo "$NAME: FAMILIES=$FAMILIES TOTAL=$TOTAL SEED=$SEED WALLS=$WALLS ABLATE=${ABLATE:--} WEIGHTS=${WEIGHTS:--} HOURS=$HOURS"
[ -z "${DRY:-}" ] || exit 0
exec bash "$(dirname "$0")/gpu_run.sh" "$@"
