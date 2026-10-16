"""SDK 领域模型：任务、检测结果与计价规则。

字段名与 https://sbnew.top/docs 的接口文档一一对应，全部为后端真实返回的
JSON 键名，没有做任何改名或映射——调过 API 的人翻这份文件不需要翻译。

金额与时间刻意保留为字符串：
- cost 是十进制字符串（如 "0.0100"，单位元）。服务端用 DECIMAL 存储并原样输出，
  解析成 float 会引入二进制浮点误差，做账时再四舍五入就会对不上。
- created_at / finished_at 是 RFC3339 字符串，是否解析成 datetime 由调用方决定。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

# 任务状态（与后端 model.Task 状态机一致）。
TASK_PENDING = "pending"
TASK_RUNNING = "running"
TASK_SUCCEEDED = "succeeded"
TASK_FAILED = "failed"

# 分段标签语义：三分类。labels_ratio 的键与 segment_labels[].label 共用这套取值。
LABEL_HUMAN = 0  # 人工
LABEL_AI = 1  # AI
LABEL_SUSPECTED = 2  # 疑似 AI

LABEL_NAMES: Dict[int, str] = {
    LABEL_HUMAN: "人工",
    LABEL_AI: "AI",
    LABEL_SUSPECTED: "疑似AI",
}

# 任务终态：到达后不会再变化，轮询可以停止。
TERMINAL_STATUSES = frozenset({TASK_SUCCEEDED, TASK_FAILED})


@dataclass
class SegmentLabel:
    """文本分段检测结果。"""

    #: 分段原文
    text: str
    #: 标签，见 LABEL_HUMAN / LABEL_AI / LABEL_SUSPECTED
    label: int
    #: 该分段的置信度
    conf: float
    #: 分段序号
    order: int
    #: [start, length]：第一项是全文中的起始位置（按 Unicode 字符计），
    #: 第二项是段长——不是结束下标。取原文：text[ps:p[0] + p[1]]。
    position: List[int]

    @property
    def label_name(self) -> str:
        return LABEL_NAMES.get(self.label, "未知")


@dataclass
class TextResult:
    """文本检测结果（任务 succeeded 后 result 字段的结构）。"""

    #: 上游状态，成功为 "success"
    status: str
    #: 模型对整篇文本的分类置信度
    softmax_confidence: float
    #: 与分段标签占比相关的整体置信度
    ratio_confidence: float
    #: 标签占比，键是字符串数字（"0"/"1"/"2"），全部值相加为 1。
    #: 实际出现哪几个键取决于本次检测命中了哪些标签，不要假设固定两个键。
    labels_ratio: Dict[str, float]
    #: 分段标注，is_merge=False 时为细分段
    segment_labels: List[SegmentLabel]
    #: 上游附加信息，可能不存在
    msg: Optional[str] = None

    @classmethod
    def from_dict(cls, d: dict) -> "TextResult":
        return cls(
            status=d["status"],
            softmax_confidence=d["softmax_confidence"],
            ratio_confidence=d["ratio_confidence"],
            labels_ratio=d.get("labels_ratio") or {},
            segment_labels=[
                SegmentLabel(**seg) for seg in (d.get("segment_labels") or [])
            ],
            msg=d.get("msg"),
        )

    @property
    def ai_ratio(self) -> float:
        """labels_ratio 里 AI（含疑似）的合计占比；缺失的键按 0 计。"""
        return float(self.labels_ratio.get("1", 0.0)) + float(
            self.labels_ratio.get("2", 0.0)
        )


@dataclass
class ImageResult:
    """图片检测结果：AI 生成置信度。"""

    #: 上游状态，成功为 "success"
    status: str
    #: AI 生成置信度
    confidence: float
    #: 上游附加信息，可能不存在
    message: Optional[str] = None

    @classmethod
    def from_dict(cls, d: dict) -> "ImageResult":
        return cls(
            status=d["status"],
            confidence=(d.get("data") or {}).get("confidence", 0.0),
            message=d.get("message"),
        )


@dataclass
class Task:
    """检测任务。提交接口立即返回，succeeded / failed 是终态。

    result 只在 succeeded 时非空：文本任务是 TextResult 的字段集合，
    图片任务是 ImageResult 的字段集合。
    """

    task_id: str
    #: "zhuque-text" 或 "zhuque-image"
    provider: str
    #: pending / running / succeeded / failed
    status: str
    #: 本次实际扣费，十进制字符串，单位元（如 "0.0100"）
    cost: str
    #: 终态前为 None；succeeded 后为结果字典（用 text_result() / image_result() 解析）
    result: Optional[dict] = None
    #: 失败时的错误码（如 "timeout"），成功时为 None
    error_code: Optional[str] = None
    #: 失败时的错误描述
    error_msg: Optional[str] = None
    latency_ms: Optional[int] = None
    created_at: Optional[str] = None
    finished_at: Optional[str] = None

    @classmethod
    def from_dict(cls, d: dict) -> "Task":
        return cls(
            task_id=d["task_id"],
            provider=d.get("provider", ""),
            status=d["status"],
            cost=str(d.get("cost", "")),
            result=d.get("result"),
            error_code=d.get("error_code"),
            error_msg=d.get("error_msg"),
            latency_ms=d.get("latency_ms"),
            created_at=d.get("created_at"),
            finished_at=d.get("finished_at"),
        )

    @property
    def is_done(self) -> bool:
        return self.status in TERMINAL_STATUSES

    @property
    def is_succeeded(self) -> bool:
        return self.status == TASK_SUCCEEDED

    @property
    def is_failed(self) -> bool:
        return self.status == TASK_FAILED

    def text_result(self) -> TextResult:
        """按文本任务解析 result。类型不对或任务未成功时抛 ValueError。"""
        if not self.is_succeeded or not self.result:
            raise ValueError(f"任务 {self.task_id} 没有成功结果（status={self.status}）")
        return TextResult.from_dict(self.result)

    def image_result(self) -> ImageResult:
        """按图片任务解析 result。类型不对或任务未成功时抛 ValueError。"""
        if not self.is_succeeded or not self.result:
            raise ValueError(f"任务 {self.task_id} 没有成功结果（status={self.status}）")
        return ImageResult.from_dict(self.result)


@dataclass
class PricingRule:
    """计价规则，来自 GET /v1/pricing。"""

    provider: str
    #: per_unit（按量：每 unit_size 个字符/张收 unit_price）或 per_call（按次）
    charge_mode: str
    unit_size: int
    #: 单价（元），十进制字符串
    unit_price: str
    #: 保底价（元），十进制字符串；实际扣费 max(min_charge, 应收) 
    min_charge: str
    #: 1 启用 / 0 停用
    enabled: int
    updated_at: Optional[str] = None

    @classmethod
    def from_dict(cls, d: dict) -> "PricingRule":
        return cls(
            provider=d["provider"],
            charge_mode=d["charge_mode"],
            unit_size=int(d.get("unit_size", 0)),
            unit_price=str(d.get("unit_price", "")),
            min_charge=str(d.get("min_charge", "")),
            enabled=int(d.get("enabled", 0)),
            updated_at=d.get("updated_at"),
        )


@dataclass
class UsagePage:
    """GET /v1/usage 的一页调用记录。"""

    list: List[Task]
    total: int
    limit: int
    offset: int

    @classmethod
    def from_dict(cls, d: dict) -> "UsagePage":
        return cls(
            list=[Task.from_dict(t) for t in (d.get("list") or [])],
            total=int(d.get("total", 0)),
            limit=int(d.get("limit", 0)),
            offset=int(d.get("offset", 0)),
        )
