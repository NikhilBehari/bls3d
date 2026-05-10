#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${BLS3D_ENV_NAME:-bls3d}"
TORCH_ARCH="${TORCH_CUDA_ARCH_LIST:-5.2 6.0 6.1 7.0 7.5 8.0 8.6+PTX}"

conda env remove -n "$ENV_NAME" -y 2>/dev/null || true
conda create -n "$ENV_NAME" python=3.10 -y
conda install -n "$ENV_NAME" -c nvidia/label/cuda-11.8.0 cuda-toolkit -y
conda install -n "$ENV_NAME" -c conda-forge gcc=11 gxx=11 -y

ENV_PREFIX="$(conda info --base)/envs/$ENV_NAME"
ACT_DIR="$ENV_PREFIX/etc/conda/activate.d"
DEACT_DIR="$ENV_PREFIX/etc/conda/deactivate.d"
mkdir -p "$ACT_DIR" "$DEACT_DIR"
cat > "$ACT_DIR/bls3d_cuda_env.sh" <<EOF
export CUDA_HOME="\$CONDA_PREFIX"
export CUDACXX="\$CONDA_PREFIX/bin/nvcc"
export CC="\$CONDA_PREFIX/bin/gcc"
export CXX="\$CONDA_PREFIX/bin/g++"
export TORCH_CUDA_ARCH_LIST="$TORCH_ARCH"
EOF
cat > "$DEACT_DIR/bls3d_cuda_env.sh" <<'EOF'
unset CUDA_HOME CUDACXX CC CXX TORCH_CUDA_ARCH_LIST
EOF

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

PIP="$ENV_PREFIX/bin/pip"

"$PIP" install --no-cache-dir \
    torch==2.1.0 torchvision==0.16.0 \
    --index-url https://download.pytorch.org/whl/cu118

"$PIP" install --no-cache-dir -r requirements.txt
"$PIP" install --no-cache-dir 'numpy<2'

"$PIP" install --no-cache-dir --no-build-isolation \
    "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.5"

"$PIP" install --no-cache-dir --no-build-isolation ./submodules/diff-gaussian-rasterization
"$PIP" install --no-cache-dir --no-build-isolation ./submodules/simple-knn

echo
echo "Done. Activate with: conda activate $ENV_NAME"
