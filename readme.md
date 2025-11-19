### DeepSeek-OCR-vllm
* vllm 版本

### install
```bash

# # 让容器可以使用宿主机的 GPU 资源, 以下方法已经弃用, deprecated
# distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
# curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
# curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list # T-shaped pipe fitting（T 形管道接头）把标准输入（standard input, stdin）同时输出到终端和文件
# sudo apt update
# sudo apt install -y nvidia-docker2 # 用来让 Docker 容器 可以直接访问 NVIDIA GPU
# sudo systemctl restart docker

# 卸载参考
# sudo apt-get remove --purge nvidia-docker2 # --purge：彻底删除，包括配置文件
# sudo apt-get autoremove


# 现在已经被 NVIDIA Container Toolkit 取代
# sudo apt install nvidia-container-toolkit 就行 # 可能会因为网络问题下载特别慢, 不用这个方法
# 直接从 官方 release 页下载整合包, 分别安装 .deb, 这些文件都不是特别大, 所以可以在代理网络的情况下下载好, 上传到服务器上进行解压
wget https://github.com/NVIDIA/nvidia-container-toolkit/releases/download/v1.18.0/nvidia-container-toolkit_1.18.0_deb_amd64.tar.gz
tar -xzvf nvidia-container-toolkit_1.18.0_deb_amd64.tar.gz
# 安装其中的四个就好
sudo dpkg -i libnvidia-container1_1.14.1-1_amd64.deb
sudo dpkg -i libnvidia-container-tools_1.14.1-1_amd64.deb
sudo dpkg -i nvidia-container-toolkit-base_1.14.1-1_amd64.deb
sudo dpkg -i nvidia-container-toolkit_1.14.1-1_amd64.deb
# 
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

docker run --gpus=all -it --name tmp ubuntu:22.04

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