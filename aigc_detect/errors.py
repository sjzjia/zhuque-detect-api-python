"""异常体系：HTTP 状态码 + 业务错误码 → Python 异常类。

错误码取值与服务端 handler/errors.go 一一对应：
    invalid_param / missing_idempotency_key   400 ValidationError
    invalid_api_key                            401 AuthenticationError
    insufficient_balance                       402 InsufficientBalanceError
    provider_disabled                          403 ProviderDisabledError
    not_found                                  404 NotFoundError
    quota_exceeded                             429 RateLimitError
    timeout / internal_error / 5xx             ServerError

不认识的错误码落回父类 AigcDetectError，code / message / http_status
都挂在异常对象上，不丢信息。
"""

from typing import Optional


class AigcDetectError(Exception):
    """所有 SDK 异常的父类。"""

    def __init__(
        self,
        message: str,
        *,
        code: str = "",
        http_status: int = 0,
    ):
        super().__init__(message)
        self.code = code
        self.http_status = http_status


class ValidationError(AigcDetectError):
    """请求参数不合法（400），含缺少 Idempotency-Key 的情况。"""


class AuthenticationError(AigcDetectError):
    """API Key 无效（401）。"""


class InsufficientBalanceError(AigcDetectError):
    """账户余额不足（402），提交即扣费因此请求被拒绝，未产生任务。"""


class ProviderDisabledError(AigcDetectError):
    """该检测能力当前停用（403）。"""


class NotFoundError(AigcDetectError):
    """task_id 不存在或不属于当前 Key（404）。"""


class RateLimitError(AigcDetectError):
    """触发限流配额（429）。"""


class ServerError(AigcDetectError):
    """服务端错误（5xx），含 timeout / internal_error。"""


class TaskFailedError(AigcDetectError):
    """任务进入 failed 终态。

    与上面的"请求失败"不同：请求本身是成功的，任务也在服务端落了终态。
    error_code / error_msg 挂在异常上；是否退款由服务端策略决定，
    客户端不需要也无法干预。
    """

    def __init__(
        self,
        message: str,
        *,
        code: str = "",
        http_status: int = 0,
        task_id: str = "",
    ):
        super().__init__(message, code=code, http_status=http_status)
        self.task_id = task_id


class TaskTimeoutError(AigcDetectError):
    """轮询超时：超过 max_poll_time 任务仍未到终态。

    任务还在服务端继续执行，不会丢失。拿到 task_id 后随时可以
    client.get_task(task_id) 继续查，不要重新提交——那会再扣一次费。
    """

    def __init__(self, task_id: str, elapsed: float):
        super().__init__(
            f"任务 {task_id} 轮询 {elapsed:.0f}s 未到终态，"
            "可稍后用 get_task() 继续查询，请勿重新提交"
        )
        self.task_id = task_id
        self.elapsed = elapsed


_STATUS_TO_EXC = {
    400: ValidationError,
    401: AuthenticationError,
    402: InsufficientBalanceError,
    403: ProviderDisabledError,
    404: NotFoundError,
    429: RateLimitError,
}


def from_http_error(status: int, code: str, message: str) -> AigcDetectError:
    """把服务端错误响应映射成具体异常；未知组合落回父类。"""
    exc_cls = _STATUS_TO_EXC.get(status)
    if status >= 500:
        exc_cls = ServerError
    if exc_cls is None:
        return AigcDetectError(message or "请求失败", code=code, http_status=status)
    return exc_cls(message or "请求失败", code=code, http_status=status)
