import os
from dotenv import load_dotenv

# 1. 自动寻找并加载 .env 文件中的变量到系统环境变量
load_dotenv()

class Config:
    # 2. 使用 os.getenv 获取变量，并设置默认值以防文件丢失
    API_URL = os.getenv("LANGGRAPH_server_API_URL", "http://localhost:2024")

# 实例化，方便其他文件直接 import
settings = Config()