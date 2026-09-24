"""PPO curriculum scaffold for the wall-climbing robot. See rl/README.md.

    python -m rl.lessons      # generate + smoke-check the lesson scenes (1..9: the benchmark)
    python -m rl.walls        # the generated walls the GPU path trains on, and ADR's ranges
    python -m rl.climb_task   # smoke-check the RL task API on every lesson
    python -m rl.bench        # where training time goes on this machine
    python -m rl.train_ppo    # train (needs gymnasium + stable-baselines3); on the GPU
                              # by itself when JAX can use one (rl.hardware)
    python -m rl.mjx_check    # the GPU path (rl.mjx_task) against the CPU one
    bash rl/gpu_setup.sh && bash rl/gpu_run.sh <name>   # a rented GPU box, unattended
    python -m rl.plot --name <name>   # a run's figures, from the logs it keeps
    python -m rl.curve --name <name>  # the learning curve: every snapshot on the benchmark
"""
import os


def default_envs():
    """Usable cores. `sched_getaffinity` respects taskset and cgroup CPU pinning,
    which `os.cpu_count()` does not -- it reports the host's cores from inside a
    container and would oversubscribe a shared Linux box. Absent on macOS.

    Lives here rather than in train_ppo so `rl.bench` can size its sweep without
    importing the training dependencies -- checking a fresh machine is exactly the
    job bench has before stable-baselines3 is installed on it.
    """
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 8
