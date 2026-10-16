"""客户端：提交、幂等重试、轮询与查询。

用法::

    from aigc_detect import AigcDetectClient

    client = AigcDetectClient(api_key="sk-zq-xxxx")
    task = client.detect_text("待检测的文本", is_merge=False)
    result = task.text_result()
    print(result.softmax_confidence, result.labels_ratio)

两个提交接口（detect_text / detect_image）默认内置完整流程：

1. 自动生成 Idempotency-Key（提交即扣费，网络重试复用同一个 Key，
   服务端据此去重、不会重复扣费）；
2. 提交成功后自动轮询，直到 succeeded / failed，无需手写 while 循环；
3. 任务失败时抛 TaskFailedError（可关），轮询超时抛 TaskTimeoutError
   （任务不丢，可继续查）。

需要自行控制轮询节奏时，用 submit_text / submit_image + get_task。
"""

import time
import uuid
from typing import Optional

import requests

from .errors import (
    AigcDetectError,
    TaskFailedError,
    TaskTimeoutError,
    from_http_error,
)
from .models import PricingRule, Task, UsagePage

DEFAULT_BASE_URL = "https://sbnew.top"
DEFAULT_TIMEOUT = 30.0  # 单次 HTTP 请求超时（秒）
DEFAULT_POLL_INTERVAL = 2.0  # 轮询间隔（秒）
DEFAULT_MAX_POLL_TIME = 180.0  # 轮询总时长上限（秒）
_SUBMIT_RETRIES = 2  # 提交请求的网络层重试次数（幂等键不变，服务端不会重复扣费）

# 瞬态网络错误：提交请求遇到它们时可以安全重试。
# 根因是 Idempotency-Key 去重——重试命中的是同一条任务记录，返回同一个 task_id。
_TRANSIENT = (requests.exceptions.Timeout, requests.exceptions.ConnectionError)


class AigcDetectClient:
    """AIGC 检测 API（文本 / 图片）客户端。

    :param api_key: 控制台创建的 API Key，形如 ``sk-zq-xxxx``。
    :param base_url: API 根地址。生产环境与站点同源（默认 https://sbnew.top），
        自建 / 私有部署时传自己的地址。
    :param timeout: 单次 HTTP 请求超时秒数。
    :param poll_interval: 自动轮询的间隔秒数。
    :param max_poll_time: 单次等待任务终态的总时长上限秒数，超时抛 TaskTimeoutError。
    :param session: 自定义 requests.Session（代理、连接池、重试策略由你接管）。
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        *,
        timeout: float = DEFAULT_TIMEOUT,
        poll_interval: float = DEFAULT_POLL_INTERVAL,
        max_poll_time: float = DEFAULT_MAX_POLL_TIME,
        session: Optional[requests.Session] = None,
    ):
        if not api_key:
            raise ValueError("api_key 不能为空：在控制台「API 密钥」页创建")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.poll_interval = poll_interval
        self.max_poll_time = max_poll_time
        self._session = session or requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
        )

    # ------------------------------------------------------------------
    # 高层 API：提交 + 自动等待
    # ------------------------------------------------------------------

    def detect_text(
        self,
        text: str,
        is_merge: Optional[bool] = None,
        *,
        idempotency_key: Optional[str] = None,
        wait: bool = True,
        raise_on_failure: bool = True,
        max_poll_time: Optional[float] = None,
    ) -> Task:
        """提交文本检测并等待结果（默认）。

        :param text: 待检测文本，上限 20000 字符（按 Unicode 字符计，不是字节）。
        :param is_merge: 分段粒度，语义容易记反——**False 才是细分段**（每段独立
            置信度，落在 segment_labels）；None（不传）或 True 走上游默认的
            粗粒度合并。要逐段看结果就传 False。
        :param idempotency_key: 不传则自动生成。同一文本的重复检测请换新 Key
            或不传（自动生成），复用旧 Key 会直接返回上一次的任务。
        :param wait: True（默认）提交后自动轮询到终态；False 等价 submit_text。
        :param raise_on_failure: 任务失败时抛 TaskFailedError（默认 True）。
        :raises ValidationError: 文本为空或超长（未扣费）。
        :raises InsufficientBalanceError: 余额不足（未扣费）。
        :raises TaskFailedError: 任务终态为 failed（服务端已按规则处理退款）。
        :raises TaskTimeoutError: 超过 max_poll_time 仍未终态（任务不丢，继续 get_task）。
        """
        task = self.submit_text(text, is_merge, idempotency_key=idempotency_key)
        if not wait:
            return task
        return self.wait_for_task(
            task.task_id, max_poll_time=max_poll_time, raise_on_failure=raise_on_failure
        )

    def detect_image(
        self,
        image_url: Optional[str] = None,
        image_base64: Optional[str] = None,
        *,
        idempotency_key: Optional[str] = None,
        wait: bool = True,
        raise_on_failure: bool = True,
        max_poll_time: Optional[float] = None,
    ) -> Task:
        """提交图片检测并等待结果（默认）。image_url / image_base64 二选一。

        :param image_url: 图片公网地址，http/https。
        :param image_base64: 图片 Base64 编码（不含 data: 前缀也可）。
            解码后需 >300×300 像素、≤10MB，jpg/png/webp。
        :raises ValidationError: 两个参数都传或都没传、地址非 http(s)（未扣费）。
        :raises InsufficientBalanceError: 余额不足（未扣费）。
        :raises TaskFailedError: / :raises TaskTimeoutError: 同 detect_text。
        """
        task = self.submit_image(
            image_url, image_base64, idempotency_key=idempotency_key
        )
        if not wait:
            return task
        return self.wait_for_task(
            task.task_id, max_poll_time=max_poll_time, raise_on_failure=raise_on_failure
        )

    # ------------------------------------------------------------------
    # 低层 API：只提交 / 只查询 / 只等待
    # ------------------------------------------------------------------

    def submit_text(
        self,
        text: str,
        is_merge: Optional[bool] = None,
        *,
        idempotency_key: Optional[str] = None,
    ) -> Task:
        """提交文本检测，立即返回 pending/running 状态的 Task，不等待结果。"""
        body = {"text": text}
        if is_merge is not None:
            body["is_merge"] = is_merge
        data = self._submit("/v1/text/classify", body, idempotency_key)
        return Task.from_dict(data)

    def submit_image(
        self,
        image_url: Optional[str] = None,
        image_base64: Optional[str] = None,
        *,
        idempotency_key: Optional[str] = None,
    ) -> Task:
        """提交图片检测，立即返回 Task，不等待结果。image_url / image_base64 二选一。"""
        if bool(image_url) == bool(image_base64):
            raise ValueError("image_url 与 image_base64 必须二选一")
        body = {}
        if image_url:
            body["image_url"] = image_url
        else:
            body["image_base64"] = image_base64
        data = self._submit("/v1/image/classify", body, idempotency_key)
        return Task.from_dict(data)

    def get_task(self, task_id: str) -> Task:
        """查询单个任务的当前状态与结果。幂等 GET，可无限次调用。"""
        data = self._request("GET", f"/v1/tasks/{task_id}")
        return Task.from_dict(data)

    def wait_for_task(
        self,
        task_id: str,
        *,
        max_poll_time: Optional[float] = None,
        raise_on_failure: bool = True,
    ) -> Task:
        """轮询任务直到终态。

        失败任务默认抛 TaskFailedError；把 raise_on_failure 设为 False
        则返回 status="failed" 的 Task，由调用方自行处理。
        """
        budget = self.max_poll_time if max_poll_time is None else max_poll_time
        started = time.monotonic()
        while True:
            task = self.get_task(task_id)
            if task.is_done:
                if task.is_failed and raise_on_failure:
                    raise TaskFailedError(
                        f"任务 {task_id} 失败：{task.error_code} {task.error_msg or ''}",
                        code=task.error_code or "",
                        task_id=task_id,
                    )
                return task
            elapsed = time.monotonic() - started
            if elapsed >= budget:
                raise TaskTimeoutError(task_id, elapsed)
            time.sleep(self.poll_interval)

    def list_usage(self, limit: int = 20, offset: int = 0) -> UsagePage:
        """分页查询调用记录，响应含 total（判断有没有下一页不要靠数条数）。"""
        data = self._request(
            "GET", "/v1/usage", params={"limit": limit, "offset": offset}
        )
        return UsagePage.from_dict(data)

    def get_pricing(self) -> list:
        """查询当前计价规则（公开接口，Key 无效也能调用）。"""
        data = self._request("GET", "/v1/pricing")
        return [PricingRule.from_dict(r) for r in (data or [])]

    # ------------------------------------------------------------------
    # 内部实现
    # ------------------------------------------------------------------

    def _submit(self, path: str, body: dict, idempotency_key: Optional[str]) -> dict:
        """带网络重试的提交：重试复用同一 Idempotency-Key，服务端返回同一任务。

        只有 Timeout / ConnectionError 会重试——请求确实没有到达或没有得到
        响应，幂等键才能兜底。HTTP 4xx/5xx 是服务端的明确答复，重试没有意义。
        """
        key = idempotency_key or f"py-{uuid.uuid4()}"
        headers = {"Idempotency-Key": key}
        last_exc = None
        for _ in range(_SUBMIT_RETRIES + 1):
            try:
                return self._request("POST", path, json_body=body, headers=headers)
            except _TRANSIENT as exc:  # 仅网络瞬断重试
                last_exc = exc
                time.sleep(0.5)
        raise AigcDetectError(f"提交失败（已重试 {_SUBMIT_RETRIES} 次）：{last_exc}")

    def _request(
        self,
        method: str,
        path: str,
        *,
        json_body=None,
        params=None,
        headers=None,
    ) -> dict:
        """发请求并解包统一响应体 {code, message, data}。

        成功（code == "ok"）返回 data；否则按 HTTP 状态码映射异常。
        带错误响应的 HTTP 状态由服务端保证与 code 一致，但这里以响应体为准
        解析，避免个别网关把 4xx/5xx 改写成自己的 HTML 错误页时误判成功。
        """
        url = self.base_url + path
        try:
            resp = self._session.request(
                method,
                url,
                json=json_body,
                params=params,
                headers=headers,
                timeout=self.timeout,
            )
        except _TRANSIENT as exc:
            raise AigcDetectError(f"网络错误：{exc}") from exc

        # 先按 JSON 解析：拿不到合法 JSON（网关错误页等）按状态码兜底。
        try:
            payload = resp.json()
        except ValueError:
            if resp.ok:
                raise AigcDetectError(
                    f"响应不是合法 JSON（HTTP {resp.status_code}）"
                ) from None
            raise from_http_error(resp.status_code, "bad_gateway", resp.text[:200])

        if payload.get("code") == "ok":
            return payload.get("data") or {}

        message = payload.get("message", "")
        code = payload.get("code", "")
        if resp.ok:  # HTTP 200 但 code != ok：异常状态，不要吞
            raise AigcDetectError(message or "请求失败", code=code, http_status=resp.status_code)
        raise from_http_error(resp.status_code, code, message)
