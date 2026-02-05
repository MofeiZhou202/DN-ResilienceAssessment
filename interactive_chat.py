import requests
import json
import os
import sys
from datetime import datetime

# 配置
COZE_URL = "https://6rjyqxwr5h.coze.site/run"  # 使用非流式端点
COZE_TOKEN = "eyJhbGciOiJSUzI1NiIsImtpZCI6ImYxYjFjNTdlLWNlZDItNGVhMC05ZmE4LWI2MjU1MTM2MzYyNyJ9.eyJpc3MiOiJodHRwczovL2FwaS5jb3plLmNuIiwiYXVkIjpbIjMzYUVQRDRDUm15WW1CYmRvZEhTQ0YzZ3gxTDFYY243Il0sImV4cCI6ODIxMDI2Njg3Njc5OSwiaWF0IjoxNzY5MTc0NDE3LCJzdWIiOiJzcGlmZmU6Ly9hcGkuY296ZS5jbi93b3JrbG9hZF9pZGVudGl0eS9pZDo3NTk3MDM2NDk5Mzk3MjQ3MDE5Iiwic3JjIjoiaW5ib3VuZF9hdXRoX2FjY2Vzc190b2tlbl9pZDo3NTk4NTQ2MjYyMjcwNDEwNzkwIn0.FFtllPErUcwiSEPswaSJcHYyBugKglaS2upODuTSZOgvQv8-QQLc_GUMJu40JLts0L2BSYPv5vIkrxI1Dt3nCwjIP2MR57yauPfWHHx12drGYpqcac_I-qRo_39Im_hWlJaGkKscywUo_njkRTV2sq5wBk3QhYRwev3RcwrSezMcQphT_Yvhj4mo6sjyY5drvOJMsNPDuye3FaQo3umOLm1dGhBsOAlKeZs7fJGvgjZT6U7EfAHAN3ehbiDTOMEPVW-dKEYe5LcxsWE5Z6bJucti2c7M7jrSwfZ999vlcJ2TKaIy1N1b0VMiMl924Z1jyXJOOqe_fHghe1aLoTfT5A"  # 请替换成你的实际token
PROJECT_ID = 7597026881757757476
DATA_FOLDER = "data"
SESSION_ID = f"interactive-session-{datetime.now().strftime('%Y%m%d%H%M%S')}"


def upload_file_to_coze(file_path):
    """
    上传文件到 Coze 平台
    """
    upload_url = "https://api.coze.cn/v1/files/upload"

    headers = {
        "Authorization": f"Bearer {COZE_TOKEN}"
    }

    file_name = os.path.basename(file_path)

    try:
        with open(file_path, 'rb') as f:
            files = {
                'file': (file_name, f, 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
            }
            response = requests.post(upload_url, headers=headers, files=files, timeout=60)
            response.raise_for_status()

            result = response.json()
            file_id = result.get('data', {}).get('id')
            file_url = result.get('data', {}).get('url')

            print(f"✅ 文件上传成功: {file_name}")

            return file_id, file_url

    except Exception as e:
        print(f"❌ 文件上传失败: {e}")
        return None, None


def send_message(prompt_text=None, file_paths=None):
    """
    发送消息到智能体（非流式，更稳定）
    """
    headers = {
        "Authorization": f"Bearer {COZE_TOKEN}",
        "Content-Type": "application/json"
    }

    # 构建消息
    prompt_parts = []

    # 添加文件
    if file_paths:
        for file_path in file_paths:
            file_id, file_url = upload_file_to_coze(file_path)

            if file_id:
                prompt_parts.append({
                    "type": "file",
                    "content": {
                        "upload_file": {
                            "file_name": os.path.basename(file_path),
                            "url": file_url,
                            "file_id": file_id
                        }
                    }
                })

    # 添加文本消息
    if prompt_text:
        prompt_parts.append({
            "type": "text",
            "content": {
                "text": prompt_text
            }
        })

    # 构建请求数据
    data = {
        "content": {
            "query": {
                "prompt": prompt_parts
            }
        },
        "type": "query",
        "session_id": SESSION_ID,  # 使用固定的session_id保持上下文
        "project_id": PROJECT_ID
    }

    print("📤 发送请求...")
    print("⏳ 评估通常需要 10-30 分钟，请耐心等待...\n")

    try:
        # 发送请求（增加超时到 2 小时）
        response = requests.post(
            COZE_URL,
            headers=headers,
            json=data,
            timeout=(60, 7200)  # 连接超时60秒，读取超时7200秒（2小时）
        )
        response.raise_for_status()

        result = response.json()

        print("=" * 60)
        print("🤖 智能体回复：")
        print("=" * 60 + "\n")

        # 解析响应
        if "content" in result:
            content = result["content"]

            # 提取答案
            if "answer" in content and content["answer"]:
                print(content["answer"])

            # 提取工具调用信息
            if "tool_request" in content and content["tool_request"]:
                print("\n[智能体正在调用工具...]\n")

            # 提取工具响应
            if "tool_response" in content and content["tool_response"]:
                print("\n[工具执行结果]\n")
                tool_resp = content["tool_response"]
                if isinstance(tool_resp, dict):
                    print(str(tool_resp))
                else:
                    print(str(tool_resp))

        print("\n" + "=" * 60 + "\n")

    except requests.exceptions.Timeout as e:
        print(f"\n❌ 请求超时: 评估过程超过 2 小时")
        print(f"💡 建议：评估可能仍在后台运行")
        print(f"💡 建议：直接在 Coze 网页界面查看结果")
    except requests.exceptions.RequestException as e:
        print(f"\n❌ 请求失败: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"状态码: {e.response.status_code}")
            try:
                print(f"响应: {e.response.text[:1000]}")
            except:
                pass


def list_files():
    """列出 data 文件夹中的文件"""
    if not os.path.exists(DATA_FOLDER):
        print(f"❌ 文件夹不存在: {DATA_FOLDER}")
        return []

    files = [f for f in os.listdir(DATA_FOLDER) if f.endswith('.xlsx')]

    if not files:
        print(f"📁 {DATA_FOLDER} 文件夹中没有 .xlsx 文件")
    else:
        print(f"\n📁 {DATA_FOLDER} 文件夹中的文件:")
        for i, filename in enumerate(files, 1):
            file_path = os.path.join(DATA_FOLDER, filename)
            size = os.path.getsize(file_path)
            print(f"  {i}. {filename} ({size} 字节)")

    return files


def show_help():
    """显示帮助信息"""
    print("\n" + "=" * 60)
    print("📖 帮助信息")
    print("=" * 60)
    print("可用命令:")
    print("  直接输入文本      - 发送消息给智能体")
    print("  /upload 文件名    - 上传指定文件并评估")
    print("  /upload-all       - 上传所有 .xlsx 文件")
    print("  /list             - 列出可用文件")
    print("  /clear            - 清屏")
    print("  /help             - 显示帮助")
    print("  /exit 或 /quit    - 退出程序")
    print()
    print("示例:")
    print("  你好")
    print("  /upload TowerSeg.xlsx ac_dc_real_case.xlsx")
    print("  /upload-all")
    print("  /list")
    print("=" * 60 + "\n")


def main():
    print("\n" + "=" * 60)
    print("🤖 规划仿真智能体 - 交互式对话终端")
    print("=" * 60)
    print(f"💡 会话ID: {SESSION_ID}")
    print(f"💡 数据文件夹: {DATA_FOLDER}")
    print("💡 输入 /help 查看可用命令\n")

    # 首次列出文件
    list_files()

    show_help()

    while True:
        try:
            # 获取用户输入
            user_input = input("👤 你: ").strip()

            if not user_input:
                continue

            # 退出命令
            if user_input.lower() in ['/exit', '/quit']:
                print("\n👋 再见！")
                break

            # 清屏命令
            if user_input.lower() == '/clear':
                os.system('cls' if os.name == 'nt' else 'clear')
                continue

            # 帮助命令
            if user_input.lower() == '/help':
                show_help()
                continue

            # 列出文件命令
            if user_input.lower() == '/list':
                list_files()
                continue

            # 上传所有文件命令
            if user_input.lower() == '/upload-all':
                files = list_files()
                if files:
                    file_paths = [os.path.join(DATA_FOLDER, f) for f in files]
                    send_message("请对这些文件进行弹性评估", file_paths)
                continue

            # 上传指定文件命令
            if user_input.lower().startswith('/upload'):
                parts = user_input.split()

                if len(parts) < 2:
                    print("⚠️  用法: /upload 文件名1 文件名2")
                    print("示例: /upload TowerSeg.xlsx ac_dc_real_case.xlsx")
                    continue

                file_names = parts[1:]
                file_paths = [os.path.join(DATA_FOLDER, f) for f in file_names]

                # 检查文件是否存在
                missing_files = [f for f in file_paths if not os.path.exists(f)]
                if missing_files:
                    print(f"⚠️  以下文件不存在:")
                    for f in missing_files:
                        print(f"   - {f}")
                    continue

                send_message("请对这些文件进行弹性评估", file_paths)
                continue

            # 其他输入作为消息处理
            send_message(user_input, None)

        except KeyboardInterrupt:
            print("\n\n👋 再见！")
            break
        except Exception as e:
            print(f"❌ 发生错误: {e}")


if __name__ == "__main__":
    main()