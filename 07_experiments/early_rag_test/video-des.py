
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import base64
from pathlib import Path
from openai import OpenAI

# ==== 1. 基本配置 ====
API_KEY = "sk-wcxgnlhmttndxzccprdqdgkbrlfakjkxjkbdtlbpilnfeevy"  # ←←← 换成你的硅基流动 API Key
BASE_URL = "https://api.siliconflow.cn/v1"

# 本地视频路径（带中文没问题）
VIDEO_PATH = Path("/home/rainyfog/code/multi-modal-fusion/测试数据/视频数据/3ada4222967f421d_part_000.mp4")

# 选择支持视频理解的模型：Qwen3-Omni 或 Qwen3-VL 系列都可以
MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"
# 如果你只想用纯视觉模型，也可以试：
# MODEL_NAME = "Qwen/Qwen3-VL-72B-Instruct"，"Qwen/Qwen3-Omni-30B-A3B-Instruct"，

# ==== 2. 读取并 base64 编码本地视频 ====
if not VIDEO_PATH.exists():
    raise FileNotFoundError(f"视频文件不存在: {VIDEO_PATH}")

with VIDEO_PATH.open("rb") as f:
    video_bytes = f.read()

video_b64 = base64.b64encode(video_bytes).decode("utf-8")
# 按文档要求，使用 data URL 格式传本地文件:contentReference[oaicite:1]{index=1}
video_data_url = f"data:video/mp4;base64,{video_b64}"

# ==== 3. 初始化客户端 ====
client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
)

# ==== 4. 构造请求并调用视频理解 ====
prompt_text = (

    # """
    # 你是一名消化内镜科的辅助诊断助手。现在给你一段 1 分钟的胃肠镜检查视频，请你完成以下任务：

    # 1. 通读整段视频，找出所有明确异常或可疑非良性病变出现的关键时间点。

    # 2. 每个关键时间点，请给出：
    # - "time_sec": 对应的时间（以秒为单位，可以是整数或一位小数）；
    # - "finding": 使用客观、描述性的语言总结该时刻可见的异常情况，
    #                 例如黏膜颜色改变、充血、水肿、糜烂、溃疡、息肉样突起等。
    #                 **不得做医学诊断推断（如“不典型增生、癌”），仅做肉眼可见的结构性描述。**
    #                 可以提及解剖部位（食管、贲门、胃体、胃角、胃窦、幽门、十二指肠）。

    # 3. 若整个视频未见明显异常，请返回空数组 []。

    # 4. 输出必须严格符合 JSON 数组格式。例如：
    # [
    #     {"time_sec": 12.5, "finding": "胃窦部黏膜充血伴轻度水肿"},
    #     {"time_sec": 35,   "finding": "胃体小弯可见息肉样黏膜隆起，表面略不规则"}
    # ]

    # 5. 只输出合法 JSON，不要输出任何解释性文字，不要输出多余内容，不要在 JSON 外添加文字。
    # """
    
    """
    你是一名消化内镜科的辅助诊断助手。现在给你一段 1 分钟的胃肠镜检查视频，请你给出内镜诊断描述
    """
)

response = client.chat.completions.create(
    model=MODEL_NAME,
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "video_url",
                    "video_url": {
                        "url": video_data_url,
                        "detail": "high",   # 细节级别：auto / low / high:contentReference[oaicite:2]{index=2}
                        "max_frames": 16,   # 建议 8–16 帧即可，一般够用:contentReference[oaicite:3]{index=3}
                        "fps": 1            # 每秒提取 1 帧
                    }
                },
                {
                    "type": "text",
                    "text": prompt_text,
                }
            ]
        }
    ],
    max_tokens=5000,
    temperature=0.7,
)

# ==== 5. 打印结果 ====
content = response.choices[0].message.content
# 有时候 SDK 返回的是 list/str，这里做个兼容
if isinstance(content, list):
    # 多段 content 的情况，拼成一个字符串
    text_out = "".join([part.get("text", "") for part in content if isinstance(part, dict)])
else:
    text_out = str(content)

print("\n====== 模型视频分析结果 ======\n")
print(text_out.strip())
