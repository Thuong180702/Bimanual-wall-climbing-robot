#!/usr/bin/env bash
# A whole GPU training run, unattended, on a rented box (after rl/gpu_setup.sh):
#
#     mkdir -p out && nohup bash rl/gpu_run.sh gpu1 > out/gpu1.console 2>&1 &
#     tail -f out/gpu1.console                      # or: .venv/bin/python -m rl.status --name gpu1
#
#   1. preflight, once per commit, ENVS and WALLS: JAX and torch both on the card in one process, the
#      JAX task still ClimbTask's on THIS card (rl.mjx_check task physics), a batch of
#      ENVS fits and how fast it runs, and a tiny training run that saves and resumes.
#      Anything wrong stops here, before the hours are paid for.
#   2. train -- on generated walls with ADR (rl.walls), or WALLS=random for the
#      no-curriculum ablation -- resuming by itself: a crash restarts from the last checkpoint, at most
#      MAX_FAILS times in a row; it stops at HOURS or TOTAL steps, whichever is first
#      (SIGTERM, so it saves on the way out).
#   3. score the best checkpoint on the FULL robot in C MuJoCo -- the benchmark,
#      lessons 1..9 both ways round, EPISODES each, against doing nothing (rl.eval) -- and replay it on the full
#      robot, the simplified one and MJX (rl.mjx_check policy), then pack out/<name>/
#      into out/<name>.tgz and print the croc line that sends it home.
#
# What out/<name>/ keeps, for an audit and for the paper's plots (python -m rl.plot):
#   train.log        everything the trainer printed, crashes and their stacks included
#   progress.jsonl   PPO's table, one line an update (reward, KL, rate, std, losses)
#   metrics.jsonl    one line a checkpoint (success and ways of ending, per lesson; the
#                    pool; best; rate; KL target; contact cap) and one per event (start,
#                    unlock, ADR move, best, rollback, exit -- with the error if it crashed)
#   gpu.csv          the card every minute: load, memory, temperature, power, clock
#   gpu_run.log      this script's own lines: restarts and why, budgets, scoring
#   eval.txt/.jsonl  the full-robot score per lesson with its 95 % interval
#   sim2sim.txt      the same policy on full robot, simplified robot, MJX
#   provenance.txt   commit, uncommitted changes, nvidia-smi, every package version
#   snaps/           the policy every 50M steps: rl.curve scores them at home (the curve)
#
# Knobs, as environment variables:
#     HOURS=24 TOTAL=3000000000 ENVS=4096 SEED=0 EPISODES=50 MAX_FAILS=3 SCORE=1 WALLS=adr
#     STALL=1800 (s without a PPO update before a trainer counts as hung)
#     CKPT=<steps> (checkpoint interval; the trainer's own default, 1M, if unset)
# ENVS is halved by the preflight, down to 1024, until a batch fits the card, and the
# run keeps the size it found (out/<name>/envs) unless ENVS is given again.
# A second call with the same name carries the same run on (its preflight skipped),
# so a short first call measures the card: HOURS=2 SCORE=0 bash rl/gpu_run.sh gpu1,
# then HOURS=22 bash rl/gpu_run.sh gpu1. SCORE=0 skips step 3, 30-60 min of CPU work
# while the GPU sits idle.
set -uo pipefail
cd "$(dirname "$0")/.."
NAME=${1:?usage: bash rl/gpu_run.sh <run name>}
RUN=out/$NAME
[ -z "${ENVS:-}" ] && [ -f "$RUN/envs" ] && ENVS=$(cat "$RUN/envs")   # what the preflight found
HOURS=${HOURS:-24} TOTAL=${TOTAL:-3000000000} ENVS=${ENVS:-4096} SEED=${SEED:-0}
EPISODES=${EPISODES:-50} MAX_FAILS=${MAX_FAILS:-3} WALLS=${WALLS:-adr} STALL=${STALL:-1800}
PY=.venv/bin/python
[[ "$HOURS" =~ ^[0-9]+$ ]] || { echo "HOURS=$HOURS: whole hours only"; exit 1; }
# a restart recompiles nothing it has compiled before
export JAX_COMPILATION_CACHE_DIR=${JAX_COMPILATION_CACHE_DIR:-$HOME/.cache/jax_climb}
trap '' HUP                          # a dropped SSH session must not end the run
mkdir -p "$RUN"
deadline=$(( $(date +%s) + HOURS * 3600 ))
say() { echo "[gpu_run $(date '+%F %T')] $*" | tee -a "$RUN/gpu_run.log"; }

{ echo "== $(date)  HOURS=$HOURS TOTAL=$TOTAL ENVS=$ENVS SEED=$SEED WALLS=$WALLS"
  git rev-parse HEAD; git status --short
  nvidia-smi
  $PY -c "import importlib.metadata as m; print('\n'.join(sorted(f\"{d.metadata['Name']}=={d.version}\" for d in m.distributions())))"
} >> "$RUN/provenance.txt" 2>&1

# ---- 1. preflight ---------------------------------------------------------------
stamp="$(git rev-parse HEAD) $ENVS $WALLS"
if [ "$(cat "$RUN/preflight.ok" 2>/dev/null)" != "$stamp" ]; then
    say "preflight"
    $PY -m rl.hardware > "$RUN/preflight.txt" 2>&1
    cat "$RUN/preflight.txt"
    grep -q "via JAX" "$RUN/preflight.txt" \
        || { say "JAX cannot use the GPU: not training on the CPU here (see rl/gpu_setup.sh)"; exit 1; }
    $PY - <<'EOF' || { say "JAX and torch cannot share the GPU in one process"; exit 1; }
import jax, jax.numpy as jp, torch
x = jp.ones((1024, 1024))
assert float((x @ x).sum()) == 1024.0 ** 3
assert torch.cuda.is_available(), "torch sees no CUDA: is it the cu126 build?"
y = torch.ones(1024, 1024, device='cuda')
assert float((y @ y).sum()) == 1024.0 ** 3
print("JAX on", jax.devices()[0].device_kind, "| torch on", torch.cuda.get_device_name(0))
EOF
    # each step at most an hour: a preflight that hangs must not eat the budget
    timeout 3600 $PY -m rl.lessons >> "$RUN/preflight.txt" 2>&1 || { say "the lesson scenes failed: $RUN/preflight.txt"; exit 1; }
    timeout 3600 $PY -m rl.mjx_check task physics 2>&1 | tee -a "$RUN/preflight.txt" \
        || { say "the GPU path is not the CPU path's task on this card: $RUN/preflight.txt"; exit 1; }
    # the batch the run will train with: halved until it fits the card
    until timeout 3600 $PY -m rl.mjx_task --envs "$ENVS" --steps 50 2>&1 | tee -a "$RUN/preflight.txt"; do
        [ "$ENVS" -gt 1024 ] || { say "not even 1024 envs run on this card: $RUN/preflight.txt"; exit 1; }
        ENVS=$(( ENVS / 2 )); say "that batch does not run here: trying ENVS=$ENVS"
    done
    echo "$ENVS" > "$RUN/envs"
    rm -rf out/_preflight
    { timeout 3600 $PY -m rl.train_ppo --name _preflight --backend mjx --envs 256 --walls "$WALLS" --steps 16384 --ckpt-every 8192 \
      && timeout 3600 $PY -m rl.train_ppo --name _preflight --backend mjx --envs 256 --walls "$WALLS" --steps 8192 --resume; } \
        >> "$RUN/preflight.txt" 2>&1 \
        || { say "a short training run did not save and resume: $RUN/preflight.txt"; exit 1; }
    rm -rf out/_preflight
    stamp="$(git rev-parse HEAD) $ENVS $WALLS"
    echo "$stamp" > "$RUN/preflight.ok"
    say "preflight passed"
fi

# ---- 2. train -------------------------------------------------------------------
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw,clocks.sm \
    --format=csv -l 60 >> "$RUN/gpu.csv" 2>/dev/null &
MON=$!
trap 'kill $MON 2>/dev/null' EXIT
fails=0
while :; do
    left=$(( deadline - $(date +%s) ))
    [ "$left" -gt 600 ] || { say "time budget reached"; break; }
    have=$($PY -c "import json, sys; print(json.load(open(sys.argv[1]))['timesteps'])" \
           "$RUN/state.json" 2>/dev/null || echo 0)
    need=$(( TOTAL - have ))
    [ "$need" -gt 0 ] || { say "step budget reached: $have"; break; }
    resume=; [ -f "$RUN/state.json" ] && resume=--resume
    say "train: $have steps so far, $need to go, $(( left / 60 )) min left, $ENVS envs"
    t0=$(date +%s)
    $PY -m rl.train_ppo --name "$NAME" --backend mjx --envs "$ENVS" --seed "$SEED" \
                        --walls "$WALLS" --steps "$need" $resume ${CKPT:+--ckpt-every "$CKPT"} &
    pid=$! why=
    # Watched: at the deadline it is told to stop (SIGTERM: it saves, then exits), and so
    # it is when no PPO update has reached progress.jsonl for STALL s -- a trainer hung
    # on the card never ends by itself. Whatever does not go in time is killed.
    while kill -0 $pid 2>/dev/null; do
        sleep 10
        if [ "$(date +%s)" -ge "$deadline" ]; then why=deadline grace=600
        elif [ $(( $(date +%s) - t0 )) -gt "$STALL" ] && { [ ! -f "$RUN/progress.jsonl" ] \
             || [ -n "$(find "$RUN/progress.jsonl" -mmin +$(( STALL / 60 )))" ]; }; then why=stalled grace=120
        else continue; fi
        say "stopping the trainer: $why"
        kill -TERM $pid 2>/dev/null
        for _ in $(seq $(( grace / 10 ))); do kill -0 $pid 2>/dev/null || break; sleep 10; done
        kill -KILL $pid 2>/dev/null
        break                        # a killed child stays a zombie until the wait below
    done
    wait $pid
    rc=$?
    [ "$why" = deadline ] && { say "time budget reached (saved on the way out)"; break; }
    [ -z "$why" ] && [ $rc -eq 0 ] && { say "training ran its steps"; break; }
    [ $(( $(date +%s) - t0 )) -gt 3600 ] && fails=0      # an hour in: not a crash loop
    fails=$(( fails + 1 ))
    say "train ${why:-exited with code $rc}: failure $fails of $MAX_FAILS in a row, see $RUN/train.log"
    [ "$fails" -lt "$MAX_FAILS" ] || { say "giving up on training: the same failure keeps coming back"; break; }
    sleep 60
done
kill $MON 2>/dev/null
pack() {
    tar czf "out/$NAME.tgz" -C out "$NAME"
    # its full path: ~/.local/bin is on PATH only in shells started after gpu_setup.sh
    croc=$(command -v croc || echo "$HOME/.local/bin/croc")
    say "packed: out/$NAME.tgz ($(du -h "out/$NAME.tgz" | cut -f1)). Send it home with"
    say "    $croc send out/$NAME.tgz        and at home:  CROC_SECRET=<the code it prints> croc --yes"
    say "then at home: tar xzf $NAME.tgz -C out && python -m rl.curve --name $NAME && python -m rl.plot --name $NAME"
}

# ---- 3. score the best checkpoint on the full robot -----------------------------
[ "${SCORE:-1}" = 1 ] || { say "SCORE=0: not scoring; carry on with the same name"; pack; exit 0; }
[ -f "$RUN/best_model.zip" ] || { say "no best checkpoint to score"; pack; exit 1; }
say "scoring best_ on the full robot, the benchmark 1..9 both ways round: $EPISODES episodes each, one process each"
jobs="1 1m 2 2m 3 3m 4 4m 5 5m 6 6m 7 7m 8 8m 9 9m"   # 'm': mirrored, toward the right
job() {
    k=${1%m}; m=; [ "$1" != "$k" ] && m=--mirror
    $PY -m rl.eval --name "$NAME" --from-best --lessons="$k" $m --episodes "$EPISODES" \
        --jsonl "$RUN/eval.jsonl" > "$RUN/eval_$1.txt" 2>&1
}
export -f job; export PY NAME EPISODES RUN
rm -f "$RUN/eval.jsonl"
printf '%s\n' $jobs | xargs -P "$(getconf _NPROCESSORS_ONLN)" -I{} bash -c 'job {}'
{ head -3 "$RUN/eval_1.txt"
  for j in $jobs; do grep -E "^ *${j} " "$RUN/eval_$j.txt" || echo "$j: eval failed, see eval_$j.txt"; done
} > "$RUN/eval.txt"
cat "$RUN/eval.txt"
$PY -m rl.mjx_check policy --run "$RUN" > "$RUN/sim2sim.txt" 2>&1 \
    || say "sim2sim FAILED -- the policy climbs differently on the full robot: $RUN/sim2sim.txt"
cat "$RUN/sim2sim.txt"
$PY -m rl.plot --name "$NAME" > "$RUN/plot.log" 2>&1 \
    || say "no plots here ($RUN/plot.log says why; this box has no matplotlib): make them at home, python -m rl.plot --name $NAME"
pack
say "done: $RUN"
