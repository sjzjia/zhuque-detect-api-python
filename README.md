<div align="center">

# aigc-detect

**AIGC 检测 API 的 Python SDK —— 文本 / 图片 AI 生成内容检测**

提交 · 幂等重试 · 自动轮询 · 结果解析 · 错误分类，几行代码拿到检测结果

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org)
[![PyPI](https://img.shields.io/badge/PyPI-aigc--detect-3775C9?style=flat-square&logo=pypi&logoColor=white)](https://pypi.org/project/aigc-detect/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

[官网与在线体验](https://sbnew.top) · [接口文档](https://sbnew.top/docs) · [获取 API Key](https://sbnew.top/console/keys)

</div>

## 这是什么

这是 [sbnew.top](https://sbnew.top) 提供的 AIGC 检测 API 的 Python 客户端。它把「提交检测 → 轮询结果」的异步流程、幂等重试与错误处理封装好，你只需要关心检测结果本身：

- **AI 文本检测**：三分类逐段标注（人工 / AI / 疑似 AI），返回整体置信度与每段在原文中的位置
- **AI 图片检测**：返回 AI 生成置信度，支持公网 URL 与 Base64 两种传参

> **声明**：本 SDK 对接的是 [sbnew.top](https://sbnew.top) 的第三方接口服务。该服务不是腾讯官方产品，也不代表腾讯；它基于腾讯云 EdgeOne（Makers Model Pro AI 网关）接入的朱雀 AIGC 检测能力，在其之上实现了 API Key 鉴权、异步任务队列、按量计费与用量查询。

## 安装

```bash
pip install aigc-detect
```

依赖 `requests>=2.25`，Python 3.8+。

## 快速开始

先到 [控制台 - API 密钥](https://sbnew.top/console/keys) 创建一个 Key（形如 `sk-zq-xxxx`），注册即有体验金：

```python
from aigc_detect import AigcDetectClient

client = AigcDetectClient(api_key="sk-zq-xxxx")

# 文本检测：is_merge=False 得到细分段（每段独立置信度）
task = client.detect_text("把待检测的文本粘在这里，支持中英文混排。", is_merge=False)
result = task.text_result()

print(f"整体 AI 置信度: {result.softmax_confidence}")
print(f"标签占比: {result.labels_ratio}")          # {"0": 0.3, "1": 0.55, "2": 0.15}
for seg in result.segment_labels:
    print(f"[{seg.label_name}] {seg.text}")        # [人工] / [AI] / [疑似AI]

# 图片检测：URL 或 Base64 二选一
task = client.detect_image(image_url="https://example.com/a.png")
print(task.image_result().confidence)              # AI 生成置信度
```

`detect_text` / `detect_image` 默认会**自动轮询直到出结果**，不用自己写 while 循环。需要自行控制节奏时：

```python
task = client.submit_text("文本")        # 只提交，立即返回
# ... 做别的事 ...
task = client.get_task(task.task_id)      # 幂等查询，可无限次
task = client.wait_for_task(task.task_id) # 或交给 SDK 等到终态
```

## 返回字段怎么读

### labels_ratio（标签占比）

三分类，键是**字符串**数字：`"0"` 人工、`"1"` AI、`"2"` 疑似 AI，值相加为 1。**实际出现哪几个键取决于本次检测命中了哪些标签**——可能只有两个键，不要写死取值逻辑，遍历即可。

### segment_labels（分段标注）

`position` 是 `[start, length]`：第一项是段落在全文中的起始位置（按 Unicode 字符计），**第二项是段长而不是结束下标**：

```python
start, length = seg.position
original = full_text[start : start + length]
```

### softmax_confidence 与 ratio_confidence

都是整体置信度但口径不同：前者是模型对整篇的分类输出，后者与分段标签占比相关。两者明显不一致时，通常说明文本里人工与 AI 内容混杂。上游未公开这两个值的具体计算方式，请以返回值为准，不要当作精确统计公式。

## 幂等与扣费（重要）

两个提交接口**提交即扣费**（按量：文本按千字、图片按张，见[价格](https://sbnew.top/#pricing)）。SDK 的保护机制：

- 每次提交自动生成 `Idempotency-Key`，**网络超时自动重试时复用同一个 Key**，服务端会返回同一个任务，不会重复扣费
- 轮询超时抛出的 `TaskTimeoutError` 任务并没有丢——拿 `task_id` 稍后继续 `get_task()` 即可，**不要重新提交**
- 换内容检测时让 Key 换新的（不传即自动生成）；复用旧 Key 会直接返回上一次的任务

## 错误处理

```python
from aigc_detect import (
    AigcDetectClient,
    InsufficientBalanceError,  # 402 余额不足（未产生任务）
    RateLimitError,            # 429 触发限流配额
    TaskFailedError,           # 任务终态 failed（服务端按规则处理退款）
    TaskTimeoutError,          # 轮询超时（任务未丢，继续查询即可）
)

try:
    task = client.detect_text("文本")
except InsufficientBalanceError:
    ...  # 去充值
except TaskFailedError as e:
    print(e.task_id, e.code)   # 如 "timeout"
except TaskTimeoutError as e:
    ...  # 稍后 get_task(e.task_id)
```

任务失败默认抛异常；想拿到失败任务对象自己处理，传 `raise_on_failure=False`。

## API 一览

| 方法 | 对应接口 | 说明 |
|---|---|---|
| `detect_text(text, is_merge)` | `POST /v1/text/classify` | 提交 + 自动等待 |
| `detect_image(image_url=..., image_base64=...)` | `POST /v1/image/classify` | 提交 + 自动等待 |
| `submit_text` / `submit_image` | 同上 | 只提交不等待 |
| `get_task(task_id)` | `GET /v1/tasks/{id}` | 查询状态与结果 |
| `wait_for_task(task_id)` | — | 轮询到终态 |
| `list_usage(limit, offset)` | `GET /v1/usage` | 分页调用记录（含 total） |
| `get_pricing()` | `GET /v1/pricing` | 当前计价规则 |

文本上限 20000 字符（按 Unicode 字符计）；图片需 >300×300 像素、≤10MB（jpg/png/webp）。

## 本地开发

```bash
git clone https://github.com/sjzjia/aigc-detect-api-python.git
cd aigc-detect-api-python
pip install -e .
python -m unittest discover -s tests -v   # 单元测试，不访问真实服务
```

## License

[MIT](LICENSE)
