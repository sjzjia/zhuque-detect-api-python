"""图片检测示例：URL / Base64 两种传参 + 手动轮询演示。

跑法：
    export AIGC_DETECT_API_KEY=sk-zq-xxxx
    python examples/detect_image.py https://example.com/a.png
"""

import os
import sys
import time

from aigc_detect import AigcDetectClient


def main() -> int:
    if len(sys.argv) < 2:
        print(f"用法: python {sys.argv[0]} <图片URL>")
        return 1

    api_key = os.environ.get("AIGC_DETECT_API_KEY")
    if not api_key:
        print("请先设置环境变量 AIGC_DETECT_API_KEY")
        return 1

    client = AigcDetectClient(api_key=api_key)

    # 方式一：一条调用搞定（提交 + 自动轮询）
    task = client.detect_image(image_url=sys.argv[1])
    result = task.image_result()
    print(f"AI 生成置信度: {result.confidence}")

    # 方式二：手动提交 + 自己控制轮询节奏
    # task = client.submit_image(image_url=sys.argv[1])
    # while not task.is_done:
    #     time.sleep(2)
    #     task = client.get_task(task.task_id)
    # if task.is_succeeded:
    #     print(task.image_result().confidence)

    return 0


if __name__ == "__main__":
    sys.exit(main())
