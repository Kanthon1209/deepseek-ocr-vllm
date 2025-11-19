from transformers import HfArgumentParser
from dataclasses import dataclass, field
import redis


@dataclass
class ServerConfig:
    host: str = field(default="0.0.0.0")
    port: int = field(default=12138)

@dataclass 
class ModelConfig:
    CUDA_VISIBLE_DEVICES: str = field(default='0') # 有的机器只有一张卡
    gpu_memory_utilization: float = field(default=0.8)
    parallel_size: int = field(default=1) # 默认用一张卡并行跑
    model_path: str = field(default='/root/.cache/modelscope/hub/models/deepseek-ai/DeepSeek-OCR')

@dataclass
class RedisConfig:
    redis_host: str = field(default='localhost')
    redis_port: int = field(default=6379)


if __name__ == '__main__':
    # 先进行参数解析再进行导入, 导入的时候就会把模块中的部分代码执行
    parser = HfArgumentParser([ServerConfig, ModelConfig, RedisConfig]) # 要传入 DataClass 数组
    server_config, model_config, redis_config = parser.parse_args_into_dataclasses(return_remaining_strings=False) # 设置成为 True 的时候会返回一个 未解析参数 数组
    import os 
    os.environ['CUDA_VISIBLE_DEVICES'] = model_config.CUDA_VISIBLE_DEVICES # 在 torch 加载前赋值好环境变量
    
    import config # import 模块, 对模块中的 常量 进行赋值操作
    config.GPU_UTILIZATION = model_config.gpu_memory_utilization
    config.PARALLEL_SIZE = model_config.parallel_size
    config.MODEL_PATH = model_config.model_path # 设置成为本地模型路径, 注意使用绝对路径, / 开头

    import uvicorn
    from main import app # 不能放到参数解析前, from main import ... 会把 main 中所有的顶层代码执行
    app.state.redis = redis.Redis(host=redis_config.redis_host, port=redis_config.redis_port, decode_responses=True)
    
    uvicorn.run(app=app, host=server_config.host, port=server_config.port)
