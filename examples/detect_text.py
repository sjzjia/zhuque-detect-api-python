"""文本检测示例：三分类分段标注 + 标签占比。

跑法：
    export AIGC_DETECT_API_KEY=sk-zq-xxxx   # Windows: set / $env:
    python examples/detect_text.py
"""

import os
import sys

from aigc_detect import AigcDetectClient

TEXT = """大模型的输出有一个典型特征：句式均匀、缺少具体细节。
而这一段是人随手写的——今天早上出门的时候，楼下的猫又蹲在快递柜上面，
看见我也不跑，就那么看着，好像我欠它一条小鱼干。"""


def main() -> int:
    api_key = os.environ.get("AIGC_DETECT_API_KEY")
    if not api_key:
        print("请先设置环境变量 AIGC_DETECT_API_KEY")
        return 1

    client = AigcDetectClient(api_key=api_key)

    # is_merge=False 得到细分段：每段独立置信度，落在 segment_labels
    task = client.detect_text(TEXT, is_merge=False)
    result = task.text_result()

    print(f"task_id:        {task.task_id}")
    print(f"扣费:           {task.cost} 元")
    print(f"整体置信度:      softmax={result.softmax_confidence}  ratio={result.ratio_confidence}")
    print(f"标签占比:        {result.labels_ratio}")
    print(f"AI（含疑似）合计: {result.ai_ratio:.0%}")
    print("-" * 50)

    for seg in result.segment_labels:
        start, length = seg.position  # [start, length]，第二项是段长
        original = TEXT[start : start + length]
        print(f"  [{seg.label_name} conf={seg.conf:.2f}] {original}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
