import os
import subprocess

def start_dev_server(host="0.0.0.0", port=2024):
    """
    独立启动函数，自动处理 Docker 环境下的监听地址
    """
    cmd = ["langgraph", "dev", "--host", host, "--port", str(port)]
    print(f"正在启动 LangGraph 远程开发服务器: {host}:{port}")
    subprocess.run(cmd)

if __name__ == "__main__":
    # 可以通过环境变量控制是否开启远程监听
    listen_host = os.getenv("LANGGRAPH_HOST", "0.0.0.0")
    start_dev_server(host=listen_host)