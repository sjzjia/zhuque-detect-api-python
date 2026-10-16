"""aigc-detect：AIGC 检测 API（文本 / 图片）的 Python SDK。

接入 https://sbnew.top 的 AIGC 检测能力：提交检测、自动轮询、
幂等重试、结果解析与错误分类，几行代码拿到结果。

>>> from aigc_detect import AigcDetectClient
>>> client = AigcDetectClient(api_key="sk-zq-xxxx")
>>> task = client.detect_text("待检测文本", is_merge=False)
>>> task.text_result().labels_ratio
"""

from .client import AigcDetectClient
from .errors import (
    AigcDetectError,
    AuthenticationError,
    InsufficientBalanceError,
    NotFoundError,
    ProviderDisabledError,
    RateLimitError,
    ServerError,
    TaskFailedError,
    TaskTimeoutError,
    ValidationError,
)
from .models import (
    LABEL_AI,
    LABEL_HUMAN,
    LABEL_NAMES,
    LABEL_SUSPECTED,
    ImageResult,
    PricingRule,
    SegmentLabel,
    Task,
    TextResult,
    UsagePage,
)

__version__ = "0.1.0"

__all__ = [
    "AigcDetectClient",
    "AigcDetectError",
    "AuthenticationError",
    "InsufficientBalanceError",
    "NotFoundError",
    "ProviderDisabledError",
    "RateLimitError",
    "ServerError",
    "TaskFailedError",
    "TaskTimeoutError",
    "ValidationError",
    "Task",
    "TextResult",
    "ImageResult",
    "SegmentLabel",
    "PricingRule",
    "UsagePage",
    "LABEL_HUMAN",
    "LABEL_AI",
    "LABEL_SUSPECTED",
    "LABEL_NAMES",
    "__version__",
]
