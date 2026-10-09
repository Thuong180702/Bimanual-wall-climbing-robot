#!/usr/bin/env bash
# One-time setup of a Linux + NVIDIA box for the GPU path, from a fresh clone:
#
#     bash rl/gpu_setup.sh
#
# Asks nothing and needs no root. Builds .venv with Python 3.12 and every version in
# requirements-gpu.txt (the file says why each one: a V100 needs JAX's CUDA 12 wheels
# and torch's cu126 build), installs croc to send the results home, then says what
# JAX and torch can see. Safe to run again: it only fills in what is missing.
set -euo pipefail
cd "$(dirname "$0")/.."

command -v nvidia-smi >/dev/null || { echo "nvidia-smi not found: no NVIDIA driver on this machine"; exit 1; }
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
drv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
[ "$drv" -ge 525 ] || { echo "NVIDIA driver $drv: JAX's CUDA 12 wheels need >= 525"; exit 1; }
free_gb=$(df -Pk . | awk 'NR == 2 {print int($4 / 1048576)}')
[ "$free_gb" -ge 20 ] || echo "WARNING: $free_gb GB free here; the environment takes ~12 GB and a run a few more"

export PATH="$HOME/.local/bin:$PATH"
mkdir -p "$HOME/.local/bin"
if ! command -v uv >/dev/null; then
    if command -v curl >/dev/null; then
        curl -LsSf https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh
    else
        python3 -m pip install --user uv
    fi
fi

# croc, the LATEST release, checked against that release's own checksums: `croc send
# <file>` here, `CROC_SECRET=<code> croc` at home. Both ends must run the same version
# (a pinned 11.5.3 here fell behind the Mac's brew croc), so this updates it on every
# run, and at home: brew upgrade croc. The latest is where releases/latest redirects:
# GitHub's API allows 60 calls an hour an address, and a rented box shares its address
CROC=$(curl -fsSLI -o /dev/null -w '%{url_effective}' https://github.com/schollz/croc/releases/latest \
       | sed -n 's|.*/tag/v\([0-9][0-9.]*\)$|\1|p') || CROC=
have=$(croc --version 2>/dev/null | awk '{print $3}' | sed 's/^v//') || have=
if [ -z "$CROC" ]; then
    echo "WARNING: GitHub did not say which croc is the latest; keeping ${have:-no croc at all}"
elif [ "$have" != "$CROC" ]; then
    tmp=$(mktemp -d)
    url="https://github.com/schollz/croc/releases/download/v$CROC"
    tgz="croc_v${CROC}_Linux-64bit.tar.gz"
    curl -LsSf -o "$tmp/$tgz" "$url/$tgz"
    curl -LsSf -o "$tmp/sums.txt" "$url/croc_v${CROC}_checksums.txt"
    (cd "$tmp" && grep " $tgz\$" sums.txt | sha256sum -c --quiet)
    tar xzf "$tmp/$tgz" -C "$tmp" croc
    install -m 755 "$tmp/croc" "$HOME/.local/bin/croc"
    rm -rf "$tmp"
fi

uv venv --python 3.12 --allow-existing .venv
# unsafe-best-match: the torch index also carries numpy & co, often older; take
# each pinned version from whichever index has it
uv pip install --python .venv/bin/python --index-strategy unsafe-best-match -r requirements-gpu.txt

.venv/bin/python - <<'EOF'
import jax, torch
print("jax", jax.__version__, jax.devices())
print("torch", torch.__version__, "cuda" if torch.cuda.is_available() else "NO CUDA",
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
EOF
.venv/bin/python -m rl.hardware
echo "croc $(croc --version 2>&1 | awk '{print $3}') in ~/.local/bin -- at home the same: brew upgrade croc"
echo "setup done. Next:  mkdir -p out && nohup bash rl/gpu_run.sh <run name> > out/<run name>.console 2>&1 &"
