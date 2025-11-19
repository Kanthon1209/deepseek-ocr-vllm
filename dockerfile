# 选择基础镜像
FROM ubuntu:22.04

# 设置工作目录
WORKDIR /app

# 拷贝项目文件到容器
COPY . /app

# 安装依赖（根据你的服务语言和环境）
RUN apt-get update && \
    apt-get install -y python3 python3-pip

# 暴露端口（你的服务监听的端口）
EXPOSE 8000

# 设置容器启动命令
CMD ["python3", "main.py"]
