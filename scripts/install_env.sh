#!/bin/bash
# 安装 llm-se-v2 训练环境依赖
set -e
source /root/miniconda3/etc/profile.d/conda.sh
conda activate llm-se-v2

# 使用清华/阿里镜像加速 pip（AutoDL 内网可达）
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple || true

echo "=== [1/6] 安装 PyTorch 2.1.0 + CUDA 12.1（走清华镜像，默认 linux wheel 即 cu121 版）==="
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0

echo "=== [2/6] 安装 transformers / peft / accelerate / trl ==="
pip install transformers==4.46.0 peft==0.14.0 accelerate==1.2.0 trl==0.12.0

echo "=== [3/6] 安装 datasets / 音频处理 ==="
pip install datasets==3.2.0 librosa==0.10.2 soundfile==0.13.0

echo "=== [4/6] 安装训练辅助 / 评估指标（numpy 锁 1.x，否则 torch2.1.0 的 numpy 桥接会报 'Numpy is not available'）==="
pip install tensorboard scipy "numpy==1.26.4" pandas tqdm pesq==0.0.4 pystoi==0.4.1

echo "=== [5/6] 安装 gradio / modelscope ==="
pip install gradio==5.20.0 modelscope

echo "=== [6/6] 清理 pip 缓存 ==="
pip cache purge || true

echo "=== 验证 ==="
python -c "import torch; print('PyTorch', torch.__version__, 'CUDA', torch.version.cuda, 'avail', torch.cuda.is_available())"
python -c "import transformers, peft, trl, datasets; print('transformers', transformers.__version__, 'peft', peft.__version__, 'trl', trl.__version__, 'datasets', datasets.__version__)"
echo "ALL_INSTALL_DONE"
