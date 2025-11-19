### DeepSeek-OCR-vllm
* vllm 版本

### install
```bash

docker run -it --name tmp ubuntu:22.04

apt update
apt install -y ubuntu-standard build-essential vim less curl wget gnupg sudo
apt install -y redis-server # remote dict server: 用于存储任务状态
systemctl start redis-server

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash ./Miniconda3-latest-Linux-x86_64.sh

# 重启 bash 让环境变量生效

# conda env remove -n deepseek-ocr 删除之前的环境
echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc # 注意使用 >> 是 append, > 是覆盖

conda create -n deepseek-ocr python=3.12.9 -y
conda activate deepseek-ocr

pip config set --global global.index-url https://mirrors.aliyun.com/pypi/simple/ # 保持镜像源一致

pip install -r requirements.txt # 先安装指定版本的, 减少依赖分析

pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118

# github.akams.cn 可以将 github 资源下载 url 转成 从国内高速下载
# https://ghproxy.cfd/ 加到 github 资源 https://github.com/... 前面即可
wget https://ghproxy.cfd/https://github.com/vllm-project/vllm/releases/download/v0.8.5/vllm-0.8.5+cu118-cp38-abi3-manylinux1_x86_64.whl
pip install vllm-0.8.5+cu118-cp38-abi3-manylinux1_x86_64.whl

# pip install flash-attn==2.7.3 --no-build-isolation
# 编译太慢了, 从 发布页 下载预编译好的, 用 github 国内镜像
wget https://ghproxy.cfd/https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu11torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
pip install flash_attn-2.7.3+cu11torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
```