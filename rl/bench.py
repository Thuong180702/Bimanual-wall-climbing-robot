"""Measure where PPO training time actually goes on this machine.

The answer decides whether the GPU is worth using at all: PPO here is a MuJoCo
simulator feeding a small MLP, so if the simulator dominates, the policy device is
noise. Measured, do not guess -- run it on your own machine.

    python -m rl.bench            # env scaling; policy device too if torch is present
    python -m rl.bench --envs 1,4,8,12
"""
import argparse, os, time
import multiprocessing as mp
import numpy as np
from . import default_envs

STEPS = 200


def _worker(lesson, steps, q):
    os.environ['OMP_NUM_THREADS'] = '1'          # no BLAS thread contention
    from .climb_task import ClimbTask
    t = ClimbTask(lesson, visual=False)          # as training builds it
    t.reset()
    a = np.zeros(17)
    t.step(a)                                     # warm up
    t0 = time.perf_counter()
    for _ in range(steps):
        t.step(a)
    q.put(steps / (time.perf_counter() - t0))


def env_scaling(counts, lesson=1):
    """total control-steps/s with n envs running in parallel processes"""
    ctx = mp.get_context('spawn')
    out = {}
    for n in counts:
        q = ctx.Queue()
        ps = [ctx.Process(target=_worker, args=(lesson, STEPS, q)) for _ in range(n)]
        for p in ps:
            p.start()
        rates = [q.get() for _ in ps]
        for p in ps:
            p.join()
        # in-worker rates only: an end-to-end number here would be dominated by
        # process spawn, which a real run pays once and this bench would pay per row
        out[n] = sum(rates)
        per_env = out[n] / n
        solo = out[counts[0]] / counts[0]
        print(f"  {n:>2} envs: {out[n]:>7.0f} steps/s total, "
              f"{per_env:>6.0f} per env ({per_env / solo * 100:.0f}% of solo speed)")
    return out


def policy_devices(obs=135, act=17, batch=512, iters=200):
    """one PPO-sized forward+backward on each available torch device"""
    try:
        import torch, torch.nn as nn
    except ImportError:
        print("  torch not installed here -- run this in the training venv")
        return {}
    devs = ['cpu']
    if torch.cuda.is_available():
        devs.append('cuda')
        print(f"  cuda device: {torch.cuda.get_device_name(0)}")
    if torch.backends.mps.is_available():
        devs.append('mps')

    def sync(dev):
        # GPU kernels launch asynchronously, so without this the timer measures how
        # fast Python can queue work, not how fast the device does it
        if dev == 'cuda':
            torch.cuda.synchronize()
        elif dev == 'mps':
            torch.mps.synchronize()

    out = {}
    for dev in devs:
        net = nn.Sequential(nn.Linear(obs, 64), nn.Tanh(), nn.Linear(64, 64),
                            nn.Tanh(), nn.Linear(64, act)).to(dev)
        opt = torch.optim.Adam(net.parameters(), 3e-4)
        x = torch.randn(batch, obs, device=dev)
        for _ in range(10):                        # warm up / compile
            net(x).square().mean().backward()
            opt.step(); opt.zero_grad()
        sync(dev)
        t0 = time.perf_counter()
        for _ in range(iters):
            net(x).square().mean().backward()
            opt.step(); opt.zero_grad()
        sync(dev)
        dt = time.perf_counter() - t0
        out[dev] = iters / dt
        print(f"  {dev:>4}: {iters / dt:>7.0f} updates/s "
              f"({dt / iters * 1000:.2f} ms per batch of {batch})")
    return out


def main():
    n = default_envs()
    ap = argparse.ArgumentParser()
    # scale the sweep to the machine, not to the one this was written on
    ap.add_argument('--envs', default=','.join(
        str(x) for x in sorted({1, n // 2, n, 2 * n}) if x))
    ap.add_argument('--lesson', type=int, default=1)
    a = ap.parse_args()
    counts = [int(x) for x in a.envs.split(',')]

    print(f"machine: {n} usable cores")
    print("simulator throughput (this is the PPO bottleneck):")
    env = env_scaling(counts, a.lesson)
    print("\npolicy update, MlpPolicy-sized net:")
    pol = policy_devices()

    best = max(env.values())
    print(f"\nbest simulator rate: {best:.0f} control-steps/s")
    if pol:
        # SB3 PPO does n_epochs * (rollout / batch) updates per rollout; with the
        # defaults in train_ppo.py that is ~10 updates per 256*n_envs env steps.
        for dev, rate in pol.items():
            share = (best / 256 * 10) / rate
            print(f"  on {dev}, policy updates would cost {share * 100:.1f}% of "
                  f"wall clock at that simulator rate")
        print("\nWhichever device is fastest above, the simulator share is what caps "
              "training. Spend cores on envs first.")


if __name__ == '__main__':
    main()
