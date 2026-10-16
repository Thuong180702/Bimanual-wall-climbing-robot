#!/usr/bin/env bash
# A whole GPU training run, unattended, on a rented box (after rl/gpu_setup.sh):
#
#     mkdir -p out && nohup bash rl/gpu_run.sh gpu1 > out/gpu1.console 2>&1 &
#     tail -f out/gpu1.console                      # or: .venv/bin/python -m rl.status --name gpu1
#
#   1. preflight, once per commit, ENVS, WALLS and card: JAX and torch both on the card in one process, the
#      JAX task still ClimbTask's on THIS card (rl.mjx_check task physics), a batch of
#      ENVS fits, which of MJX's backends the run steps on (Warp only if it is C
#      MuJoCo's physics by the same checks: rl.mjx_check warp;
#      out/<name>/physics) -- these four at once, each its own process, the GPU near
#      idle through them -- and a tiny training run that saves and resumes.
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
#     NET=64,64 (the policy and value nets' hidden sizes, a fresh run only; a run
#                carried on keeps its checkpoint's shape; unset: the trainer's 256,256)
# ENVS is halved by the preflight, down to 1024, until a batch fits the card, and the
# run keeps the size it found (out/<name>/envs) unless ENVS is given again.
# A second call with the same name carries the same run on (its preflight skipped),
# so a short first call measures the card: HOURS=2 SCORE=0 bash rl/gpu_run.sh gpu1,
# then HOURS=22 bash rl/gpu_run.sh gpu1. SCORE=0 skips step 3, 30-60 min of CPU work
# while the GPU sits idle. On another box, the run's .tgz carries it on (croc'd over,
# wherever it landed): HOURS=22 bash rl/gpu_run.sh gpu1 ~/gpu1.tgz -- unpacked into
# out/gpu1 by this script, never over another run, the preflight again on the new card.
set -uo pipefail
NAME=${1:?usage: bash rl/gpu_run.sh <run name> [<run name>.tgz, packed on another box]}
PKG=${2:-}; [ -z "$PKG" ] || [ "${PKG#/}" != "$PKG" ] || PKG=$PWD/$PKG   # before the cd
cd "$(dirname "$0")/.."
RUN=out/$NAME
PY=.venv/bin/python
steps() { $PY -c "import json, sys; print(json.load(open(sys.argv[1]))['timesteps'])" "$1" 2>/dev/null || echo 0; }

# ---- 0. a run packed on another box, carried on here ------------------------------
# Unpacked by this script, not by hand: unpacked by hand anywhere but out/, the package
# left out/<name> without a checkpoint and the 22 h run began again at step 0 without a
# word (gpu1, 2026-09-26). Never over a run already in out/<name>, whichever run it is:
# the call stops and says how to carry either one on.
if [ -n "$PKG" ]; then
    [ -f "$PKG" ] || { echo "no such file: $PKG"; exit 1; }
    if [ -f "$RUN/state.json" ]; then
        echo "$RUN already holds a run, at $(steps "$RUN/state.json") steps. To carry that one on, call this"
        echo "without the .tgz; to carry the package on instead, first: mv $RUN $RUN.aside"
        exit 1
    fi
    grep -qx '[^/]*/state.json' <<< "$(tar tzf "$PKG")" \
        || { echo "$PKG is not a run packed by this script (no <name>/state.json in it)"; exit 1; }
    mkdir -p "$RUN" && tar xzf "$PKG" -C "$RUN" --strip-components=1 || { echo "cannot unpack $PKG"; exit 1; }
    echo "unpacked $PKG into $RUN: carrying it on from $(steps "$RUN/state.json") steps"
fi
[ -z "${ENVS:-}" ] && [ -f "$RUN/envs" ] && ENVS=$(cat "$RUN/envs")   # what the preflight found
HOURS=${HOURS:-24} TOTAL=${TOTAL:-3000000000} ENVS=${ENVS:-4096} SEED=${SEED:-0}
EPISODES=${EPISODES:-50} MAX_FAILS=${MAX_FAILS:-3} WALLS=${WALLS:-adr} STALL=${STALL:-1800}
[[ "$HOURS" =~ ^[0-9]+$ ]] || { echo "HOURS=$HOURS: whole hours only"; exit 1; }
# a restart recompiles nothing it has compiled before
export JAX_COMPILATION_CACHE_DIR=${JAX_COMPILATION_CACHE_DIR:-$HOME/.cache/jax_climb}
# Every Python line reaches the console as it is printed. Piped, Python held all of a
# process's output until it ended, so the preflight's MJX check -- minutes of one robot
# at a time, the GPU near idle -- showed nothing at all and looked hung (gpu1, 2026-09-25).
export PYTHONUNBUFFERED=1
# ...and a Python process ended with SIGABRT (a step's timeout, the watchdog) first
# prints where each of its threads was
export PYTHONFAULTHANDLER=1
# a preflight step: an hour at most, and if it hangs, its stacks on the way out
step() { timeout --signal=ABRT --kill-after=60 3600 "$@"; }
trap '' HUP                          # a dropped SSH session must not end the run
mkdir -p "$RUN"
deadline=$(( $(date +%s) + HOURS * 3600 ))
say() { echo "[gpu_run $(date '+%F %T')] $*" | tee -a "$RUN/gpu_run.log"; }

{ echo "== $(date)  HOURS=$HOURS TOTAL=$TOTAL ENVS=$ENVS SEED=$SEED WALLS=$WALLS"
  git rev-parse HEAD; git status --short
  echo "threads allowed: ulimit -u $(ulimit -u), cgroup pids.max $(cat /sys/fs/cgroup/pids.max 2>/dev/null || echo -)"
  nvidia-smi
  $PY -c "import importlib.metadata as m; print('\n'.join(sorted(f\"{d.metadata['Name']}=={d.version}\" for d in m.distributions())))"
} >> "$RUN/provenance.txt" 2>&1
# gpu6 (2026-10-10) shared its card from the start -- 5.2 GB used, 100 % busy, 270 W before it
# began -- and ran 9k env-steps/s against gpu5's 14.3k on the same card; nothing said so. Another
# container's processes do not show in nvidia-smi from inside this one; its memory and load do
read -r used util <<< "$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ',')"
if [ "${used:-0}" -gt 1000 ] || [ "${util:-0}" -gt 20 ]; then
    say "WARNING: before this run the GPU already has ${used} MiB in use and is ${util} % busy -- another job shares it and training will be slower. Stop it, or take another box"
fi

# ---- 1. preflight ---------------------------------------------------------------
# the card is in the stamp: a run carried to another box (its .tgz unpacked there)
# is checked again on the card it lands on
card=$(nvidia-smi --query-gpu=uuid --format=csv,noheader 2>/dev/null | head -1)
stamp="$(git rev-parse HEAD) $ENVS $WALLS $card"
if [ "$(cat "$RUN/preflight.ok" 2>/dev/null)" != "$stamp" ]; then
    say "preflight: 4 steps on this card, each at most an hour. Harmless in the console:" \
        "'numa_hwloc ... hwloc_set_cpubind() failed' (XLA may not pin its threads in a" \
        "container), and Warp's 'MULTICCD ... CYLINDER' warning (one contact a motor pair)"
    say "preflight 1/4: the GPU JAX and torch see"
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
    say "preflight 2/4: the lesson scenes"
    step $PY -m rl.lessons >> "$RUN/preflight.txt" 2>&1 || { say "the lesson scenes failed: $RUN/preflight.txt"; exit 1; }
    # Four at once, each its own process. Each is one robot at a time or XLA compiling
    # -- a core apiece, the GPU near idle -- so one after another they were most of the
    # preflight (gpu4's RTX 3090, 2026-10-07: the task and physics checks 23 min, the
    # batch 10, of 42). The batch's speed, timed beside them, is only a rough number:
    # what it settles is that ENVS fits.
    # Each on its own quarter of the cores: XLA, LLVM and OpenBLAS size their thread
    # pools by the cores a process may run on (a JAX compile: 63 threads on 12 cores, 18
    # on 3), and four sized for all 96 ran gpu6's box out of threads (2026-10-09: 'LLVM
    # ERROR: pthread_create failed', the physics check aborted at its first compile).
    # A check that still dies beside the others runs again alone before it counts.
    say "preflight 3/4: at once -- the JAX task against ClimbTask, MJX physics against C," \
        "MJX's Warp backend against both, and $ENVS envs in one batch (halved until it fits)," \
        "then step 4's short training run on Warp in its place"
    read -ra quarter <<< "$($PY -c "import os; c = sorted(os.sched_getaffinity(0)); n = max(len(c) // 4, 1)
print(*(','.join(map(str, c[i * n:(i + 1) * n] or c)) for i in range(4)))")"
    pin=; taskset -c "${quarter[0]}" true 2>/dev/null && pin=1    # no taskset, or not allowed: unpinned
    on() { local q=${quarter[$1]}; shift; if [ -n "$pin" ]; then step taskset -c "$q" "$@"; else step "$@"; fi; }
    # every line a check prints, stamped with its seconds in: where the preflight's time goes
    # (gpu9's box: step 3 took 19.7 min and nothing said which check, or which part of it)
    stamped() { local t0=$SECONDS; while IFS= read -r l; do printf '[%4ss] %s\n' $(( SECONDS - t0 )) "$l"; done; }
    check() { local i=$1; shift; on "$i" $PY -m rl.mjx_check "$@" 2>&1 | stamped | tee "$RUN/check_$1.txt"; }
    again() { say "the $1 check died beside the others: again, alone"; step $PY -m rl.mjx_check "$@" 2>&1 | stamped | tee "$RUN/check_$1.txt"; }
    # beside the checks only ENVS itself: their memory is not the card's, and a batch that failed
    # there was halved for good (gpu11's 3090, 2026-10-14: 4096 -> 1024); halved only alone
    fit() {
        local n=$ENVS
        until "$@" $PY -m rl.mjx_task --envs "$n" --steps 10 2>&1 | stamped >> "$RUN/check_envs.txt"; do
            [ "$1" = step ] && [ "$n" -gt 1024 ] || return 1
            n=$(( n / 2 )); say "that batch does not run here: trying ENVS=$n"
        done
        echo "$n" > "$RUN/envs"
    }
    # step 4, a short training run that saves and resumes, on physics $1, launched by the rest
    # (`on 3` beside the checks, or `step` alone)
    short() {
        local phy=$1; shift
        rm -rf out/_preflight
        { "$@" $PY -m rl.train_ppo --name _preflight --backend mjx --physics "$phy" --envs 256 --walls "$WALLS" --steps 16384 --ckpt-every 8192 ${NET:+--net "$NET"} \
          && "$@" $PY -m rl.train_ppo --name _preflight --backend mjx --physics "$phy" --envs 256 --walls "$WALLS" --steps 8192 --resume; } \
            2>&1 | stamped > "$RUN/short_$phy.txt"
        local r=$?
        rm -rf out/_preflight
        return $r
    }
    echo jax > "$RUN/physics"            # JAX unless Warp proves itself, and if it never finishes
    : > "$RUN/check_envs.txt"
    rm -f "$RUN"/short_*.txt
    check 0 task & p1=$!
    check 1 physics & p2=$!
    check 2 warp --out "$RUN/physics" & p3=$!
    # the fit's quarter then runs step 4 on Warp, what every card but one has trained on, while
    # the checks still run (gpu9's box: step 4 took 9.4 min after step 3's 19.7): exit 2 the
    # fit failed, 3 the short run did. It counts only if step 3 then says Warp; else it runs
    # again alone on what step 3 says, as a check that dies beside the others does
    { fit on 3 || exit 2; short warp on 3 || exit 3; } & p4=$!
    wait $p1; r1=$?; wait $p2; r2=$?; wait $p4; r4=$?; wait $p3; r3=$?
    [ $r4 -ne 2 ] || { say "$ENVS envs did not run beside the checks: again, alone"
                      if fit step; then r4=3; else r4=2; fi; }   # 3: step 4 then runs alone
    [ $r1 -eq 0 ] || { again task; r1=$?; }
    [ $r2 -eq 0 ] || { again physics; r2=$?; }
    [ $r3 -eq 0 ] || again warp --out "$RUN/physics"   # a wrong Warp ends 0, saying jax
    cat "$RUN"/check_task.txt "$RUN"/check_physics.txt "$RUN"/check_warp.txt "$RUN"/check_envs.txt >> "$RUN/preflight.txt"
    grep -E "control-steps/s|build \+ compile" "$RUN/check_envs.txt"
    [ $r1 -eq 0 ] && [ $r2 -eq 0 ] \
        || { say "the GPU path is not the CPU path's task on this card: $RUN/check_task.txt, $RUN/check_physics.txt"; exit 1; }
    [ $r4 -ne 2 ] || { say "not even 1024 envs run on this card: $RUN/check_envs.txt"; exit 1; }
    ENVS=$(cat "$RUN/envs")
    # Warp wherever it is C MuJoCo's physics (step 3), untimed: gpu5's RTX 3090 stepped it
    # 2.82x JAX's speed (14,300 env-steps/s training), and timing both batches cost ~7 min
    # and once sent a run to JAX on a crash of the timing itself (gpu6, 2026-10-10).
    # `python -m rl.mjx_check warp --speed --envs N` still times them, by hand.
    say "physics: $(cat "$RUN/physics")"
    PHY=$(cat "$RUN/physics")
    if [ "$PHY" = warp ] && [ $r4 -eq 0 ]; then
        say "preflight 4/4: a short training run on warp saved and resumed, beside step 3 ($RUN/short_warp.txt)"
    else
        say "preflight 4/4: a short training run that saves and resumes, on $PHY, alone (its log: $RUN/short_$PHY.txt)"
        short "$PHY" step || { say "a short training run did not save and resume: $RUN/short_$PHY.txt"; exit 1; }
    fi
    cat "$RUN"/short_*.txt >> "$RUN/preflight.txt"
    # gpu9 (2026-10-12) passed with one XLA kernel 16 min to compile here, and its training then
    # sat 18+ min at 0 % GPU on a recompile: a step graph XLA compiles that slowly is a bug to fix
    grep -q "Very slow compile" "$RUN/preflight.txt" \
        && say "WARNING: XLA reported a very slow GPU compile in the preflight ($RUN/preflight.txt): a recompile in training may stall the same way"
    stamp="$(git rev-parse HEAD) $ENVS $WALLS $card"
    echo "$stamp" > "$RUN/preflight.ok"
    say "preflight passed"
fi

pack() {
    tar czf "out/$NAME.tgz" -C out "$NAME"
    # its full path: ~/.local/bin is on PATH only in shells started after gpu_setup.sh
    croc=$(command -v croc || echo "$HOME/.local/bin/croc")
    say "packed: out/$NAME.tgz ($(du -h "out/$NAME.tgz" | cut -f1)). Send it home with"
    say "    $croc send out/$NAME.tgz        and at home:  CROC_SECRET=<the code it prints> croc --yes"
    say "    (croc $($croc --version 2>&1 | awk '{print $3}') here: the one at home must be the same -- brew upgrade croc)"
    say "then at home: tar xzf $NAME.tgz -C out && python -m rl.curve --name $NAME && python -m rl.plot --name $NAME"
}

# ---- 2. train -------------------------------------------------------------------
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw,clocks.sm \
    --format=csv -l 60 >> "$RUN/gpu.csv" 2>/dev/null &
MON=$!
trap 'kill $MON 2>/dev/null' EXIT
fails=0
while :; do
    left=$(( deadline - $(date +%s) ))
    [ "$left" -gt 600 ] || { say "time budget reached"; break; }
    have=$(steps "$RUN/state.json")
    need=$(( TOTAL - have ))
    [ "$need" -gt 0 ] || { say "step budget reached: $have"; break; }
    resume=; [ -f "$RUN/state.json" ] && resume=--resume
    net=; [ -z "$resume" ] && [ -n "${NET:-}" ] && net="--net $NET"
    # gpu9 (2026-10-12): the card fell off the bus at 176.6M ('GPU is lost' in gpu.csv, cuInit
    # failing), and the restart, forced onto --backend mjx, trained on the CPU at 90 env-steps/s
    # for 7.5 h -- an update every 22 min, inside STALL. The preflight's own GPU check, every start,
    # its whole output first: piped, `grep -q` left at the first line and the next print's broken
    # pipe failed the pipeline (pipefail), and gpu10 (dfc5a31) stopped with its GPU fine
    grep -q "via JAX" <<< "$($PY -m rl.hardware 2>&1)" || {
        say "JAX cannot use the GPU any more ($(nvidia-smi -L 2>&1 | head -1)): not training on the CPU." \
            "Reboot the box or take another, then carry the run on from its .tgz"
        pack; exit 1; }
    say "train: $have steps so far, $need to go, $(( left / 60 )) min left, $ENVS envs"
    t0=$(date +%s)
    $PY -m rl.train_ppo --name "$NAME" --backend mjx --physics "$(cat "$RUN/physics" 2>/dev/null || echo jax)" \
                        --envs "$ENVS" --seed "$SEED" \
                        --walls "$WALLS" --steps "$need" $resume $net ${CKPT:+--ckpt-every "$CKPT"} &
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
        # still there: where it hangs, into train.log (faulthandler), then gone
        kill -0 $pid 2>/dev/null && { kill -ABRT $pid 2>/dev/null; sleep 5; }
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

# ---- 3. score the best checkpoint on the full robot -----------------------------
[ "${SCORE:-1}" = 1 ] || { say "SCORE=0: not scoring; carry on with the same name"; pack; exit 0; }
[ -f "$RUN/best_model.zip" ] || { say "no best checkpoint to score"; pack; exit 1; }
say "scoring best_ on the full robot, the benchmark both ways round but 5 and 6 (rl.walls.COMPOSED): $EPISODES episodes each, one process each"
jobs="1 1m 2 2m 3 3m 4 4m 7 7m 8 8m 9 9m"   # 'm': mirrored, toward the right; 5 and 6 are composed
# One thread each: an episode is one robot in C MuJoCo, and unset, every one of the 18
# would size numpy's, scipy's and torch's pools for all the cores -- thousands of
# threads, past what gpu6's box allowed four JAX processes (preflight, 2026-10-09)
job() {
    k=${1%m}; m=; [ "$1" != "$k" ] && m=--mirror
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
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
