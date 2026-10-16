"""The climbing task and its policy. See rl/README.md.

    python -m rl.lessons      # the benchmark scenes (1..9) + their checks
    python -m rl.walls        # the generated walls training draws, and ADR's ranges
    python -m rl.climb_task   # the task's own invariants on every lesson
    python -m rl.rsi          # the reference climbs' fixture
    python -m rl.train_ppo    # train on MJX (needs gymnasium + stable-baselines3)
    python -m rl.mjx_check    # the GPU path (rl.mjx_task) against the C task
    bash rl/gpu_setup.sh && bash rl/gpu_run.sh <name>   # a rented GPU box, unattended
    python -m rl.plot --name <name>   # a run's figures, from the logs it keeps
    python -m rl.curve --name <name>  # the learning curve: every snapshot on the benchmark
"""
import os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URDF = os.path.join(ROOT, 'urdf')
SCENE = os.path.join(URDF, 'scene.xml')
if URDF not in sys.path:            # climbing_env.py lives next to scene.xml
    sys.path.insert(0, URDF)


def default_envs():
    """Usable cores. `sched_getaffinity` respects taskset and cgroup CPU pinning,
    which `os.cpu_count()` does not -- it reports the host's cores from inside a
    container and would oversubscribe a shared Linux box. Absent on macOS.

    """
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 8
