"""What this machine can train on, and which simulator backend follows from it.

    python -m rl.hardware        # what is here, and what training will use

`rl.train_ppo` prints this line at startup and picks its backend from it, so a
run says on its own log which machine it was on -- `win17`'s 20M steps had to be
worked out from file timestamps.

The question it answers is narrow: is there an NVIDIA GPU that JAX can actually
use, because that is what MJX -- MuJoCo's own physics on the GPU, `rl.mjx_task`
-- needs. If so the backend is 'mjx', otherwise 'cpu': C MuJoCo in one process
per core, exactly as before. Two near-misses are called out by name rather than
reported as "no GPU":

  * A GPU is present and JAX only sees the CPU. On Windows that is the usual
    case: JAX has had no native CUDA wheels since 0.4.x, and CUDA JAX runs under
    WSL2. Otherwise it is a missing `pip install -U "jax[cuda12]"`.
  * JAX sees a GPU but `mujoco.mjx` is not installed (`pip install mujoco-mjx`).

With no NVIDIA card at all JAX is not even imported, so the CPU path pays nothing
for the GPU one existing. `--backend` on rl.train_ppo overrides the choice
either way (mjx on a CPU runs, slowly: that is how rl.mjx_check runs on a Mac).
"""
import os
import shutil
import subprocess

# JAX takes 75 % of the GPU's memory the moment it starts, and the policy -- torch
# -- trains on the same card. This module is the first to import JAX.
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')


def nvidia_gpu():
    """Name of the first NVIDIA GPU the driver reports, or None. `nvidia-smi` is
    the one check that works the same on Windows and Linux, with no CUDA Python
    stack installed."""
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name",
                              "--format=csv,noheader"], capture_output=True,
                             text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return None
    name = (out.stdout or "").strip().splitlines()
    return name[0].strip() if out.returncode == 0 and name else None


def jax_gpu():
    """The first GPU device JAX can put an array on, or None."""
    try:
        import jax
    except ImportError:
        return None
    try:
        return next(iter(jax.devices("gpu")), None)
    except RuntimeError:          # jax raises when no backend of that kind exists
        return None


def has_mjx():
    try:
        from mujoco import mjx                                   # noqa: F401
        return True
    except ImportError:
        return False


def backend():
    """('mjx' | 'cpu', one line saying why)"""
    card = nvidia_gpu()
    if card is None:
        return "cpu", "no NVIDIA GPU: training on the CPU simulator"
    dev = jax_gpu()
    if dev is None:
        return "cpu", (f"{card} is here but JAX cannot use it "
                       f"(on Windows CUDA JAX needs WSL2; on Linux run "
                       f"rl/gpu_setup.sh): training on the CPU")
    if not has_mjx():
        return "cpu", (f"JAX sees {dev.device_kind} but mujoco-mjx is missing "
                       f"(pip install mujoco-mjx): training on the CPU")
    return "mjx", f"{dev.device_kind} via JAX: physics, task and policy on the GPU (rl.mjx_task)"


def report():
    from . import default_envs
    name, why = backend()
    return (f"[hardware] {default_envs()} usable cores, "
            f"{os.cpu_count()} on the machine; {why}")


if __name__ == "__main__":
    print(report())
    print("  nvidia-smi:", nvidia_gpu() or "not found")
    print("  jax gpu:   ", jax_gpu() or "none")
    print("  mujoco-mjx:", "installed" if has_mjx() else "not installed")
