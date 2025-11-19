### DeepSeek-OCR-vllm
* vllm 版本

### install
```bash
# conda env remove -n deepseek-ocr 删除之前的环境
conda create -n deepseek-ocr python=3.12.9 -y
conda activate deepseek-ocr

pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# github.akams.cn 可以将 github 资源下载 url 转成 从国内高速下载
# https://ghproxy.cfd/ 加到 github 资源 https://github.com/... 前面即可
wget https://ghproxy.cfd/https://github.com/vllm-project/vllm/releases/download/v0.8.5/vllm-0.8.5+cu118-cp38-abi3-manylinux1_x86_64.whl
pip install vllm-0.8.5+cu118-cp38-abi3-manylinux1_x86_64.whl

# pip install flash-attn==2.7.3 --no-build-isolation
# 编译太慢了, 从 发布页 下载预编译好的, 用 github 国内镜像
wget https://ghproxy.cfd/https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu11torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
pip install flash_attn-2.7.3+cu11torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
```