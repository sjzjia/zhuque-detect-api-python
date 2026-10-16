"""单元测试：不访问真实服务，mock requests.Session。

跑法：在仓库根目录执行
    python -m unittest discover -s tests -v
"""

import json
import unittest
from unittest import mock

from aigc_detect import (
    AigcDetectClient,
    InsufficientBalanceError,
    Task,
    TaskFailedError,
    TaskTimeoutError,
)


def _resp(payload, status_code=200):
    """构造一个行为足够像 requests.Response 的 mock。"""
    r = mock.Mock()
    r.status_code = status_code
    r.ok = status_code < 400
    r.text = json.dumps(payload)
    r.json.return_value = payload
    return r


def make_client():
    client = AigcDetectClient(api_key="sk-zq-test", poll_interval=0)
    client._session = mock.Mock()
    return client


TEXT_RESULT = {
    "status": "success",
    "softmax_confidence": 0.93,
    "ratio_confidence": 0.87,
    "labels_ratio": {"0": 0.3, "1": 0.55, "2": 0.15},
    "segment_labels": [
        {"text": "第一段", "label": 0, "conf": 0.91, "order": 0, "position": [0, 3]},
        {"text": "第二段", "label": 1, "conf": 0.88, "order": 1, "position": [3, 3]},
    ],
}


def task_payload(status="succeeded", result=None, error_code=None):
    return {
        "task_id": "01TEST",
        "provider": "zhuque-text",
        "status": status,
        "cost": "0.0100",
        "result": result,
        "error_code": error_code,
        "error_msg": "boom" if error_code else None,
        "latency_ms": 1200,
        "created_at": "2026-10-16T10:00:00+08:00",
        "finished_at": "2026-10-16T10:00:02+08:00",
    }


class TestTextDetection(unittest.TestCase):
    def test_detect_text_full_flow(self):
        """提交 → 轮询 → succeeded，结果完整解析。"""
        client = make_client()
        # 第一次调用是 POST 提交，之后是 GET 轮询
        client._session.request.side_effect = [
            _resp({"code": "ok", "data": task_payload(status="pending", result=None)}),
            _resp({"code": "ok", "data": task_payload(status="succeeded", result=TEXT_RESULT)}),
        ]

        task = client.detect_text("测试文本", is_merge=False)

        self.assertTrue(task.is_succeeded)
        self.assertEqual(task.cost, "0.0100")
        result = task.text_result()
        self.assertEqual(result.softmax_confidence, 0.93)
        # 三分类：labels_ratio 键是字符串数字
        self.assertEqual(result.labels_ratio["2"], 0.15)
        self.assertAlmostEqual(result.ai_ratio, 0.70)
        # position 是 [start, length]
        self.assertEqual(result.segment_labels[1].position, [3, 3])
        self.assertEqual(result.segment_labels[0].label_name, "人工")
        # 提交请求带了自动生成的幂等键
        submit_call = client._session.request.call_args_list[0]
        self.assertIn("Idempotency-Key", submit_call.kwargs["headers"])
        # 提交 body 正确
        self.assertEqual(submit_call.kwargs["json"], {"text": "测试文本", "is_merge": False})

    def test_detect_text_failed_raises(self):
        """终态 failed 默认抛 TaskFailedError，携带 task_id 与 error_code。"""
        client = make_client()
        client._session.request.side_effect = [
            _resp({"code": "ok", "data": task_payload(status="pending")}),
            _resp({"code": "ok", "data": task_payload(status="failed", error_code="timeout")}),
        ]
        with self.assertRaises(TaskFailedError) as ctx:
            client.detect_text("测试文本")
        self.assertEqual(ctx.exception.task_id, "01TEST")
        self.assertEqual(ctx.exception.code, "timeout")

    def test_detect_text_failed_no_raise(self):
        """raise_on_failure=False 时返回 failed 任务对象。"""
        client = make_client()
        client._session.request.side_effect = [
            _resp({"code": "ok", "data": task_payload(status="pending")}),
            _resp({"code": "ok", "data": task_payload(status="failed", error_code="timeout")}),
        ]
        task = client.detect_text("测试文本", raise_on_failure=False)
        self.assertTrue(task.is_failed)
        self.assertEqual(task.error_code, "timeout")

    def test_is_merge_omitted_when_none(self):
        """is_merge=None 时不传该字段（后端用上游默认值）。"""
        client = make_client()
        client._session.request.side_effect = [
            _resp({"code": "ok", "data": task_payload(status="pending")}),
        ]
        client.submit_text("文本")
        body = client._session.request.call_args.kwargs["json"]
        self.assertEqual(body, {"text": "文本"})

    def test_poll_timeout_keeps_task_id(self):
        """轮询超时抛 TaskTimeoutError，提示不要重新提交。"""
        client = make_client()
        pending = _resp({"code": "ok", "data": task_payload(status="pending")})
        client._session.request.side_effect = None
        client._session.request.return_value = pending
        with self.assertRaises(TaskTimeoutError) as ctx:
            client.detect_text("文本", max_poll_time=0.01)
        self.assertEqual(ctx.exception.task_id, "01TEST")


class TestImageDetection(unittest.TestCase):
    def test_url_or_base64_exclusive(self):
        client = make_client()
        with self.assertRaises(ValueError):
            client.submit_image(image_url="https://a.com/1.png", image_base64="xxx")
        with self.assertRaises(ValueError):
            client.submit_image()

    def test_image_result(self):
        client = make_client()
        payload = {
            "task_id": "01IMG",
            "provider": "zhuque-image",
            "status": "succeeded",
            "cost": "0.0200",
            "result": {"status": "success", "data": {"confidence": 0.77}},
            "error_code": None,
            "error_msg": None,
            "latency_ms": 800,
            "created_at": "2026-10-16T10:00:00+08:00",
            "finished_at": "2026-10-16T10:00:01+08:00",
        }
        client._session.request.side_effect = [
            _resp({"code": "ok", "data": payload}),
        ]
        task = client.detect_image(image_url="https://a.com/1.png", wait=False)
        self.assertEqual(task.image_result().confidence, 0.77)
        body = client._session.request.call_args.kwargs["json"]
        self.assertEqual(body, {"image_url": "https://a.com/1.png"})


class TestErrorMapping(unittest.TestCase):
    def test_insufficient_balance(self):
        client = make_client()
        client._session.request.return_value = _resp(
            {"code": "insufficient_balance", "message": "余额不足，请先充值"}, status_code=402
        )
        with self.assertRaises(InsufficientBalanceError):
            client.submit_text("文本")

    def test_non_json_error_page(self):
        """网关返回 HTML 错误页时不崩溃，落回 ServerError 家族。"""
        r = mock.Mock()
        r.status_code = 502
        r.ok = False
        r.text = "<html>Bad Gateway</html>"
        r.json.side_effect = ValueError("no json")
        client = make_client()
        client._session.request.return_value = r
        from aigc_detect import ServerError

        with self.assertRaises(ServerError):
            client.get_task("01TEST")


class TestModels(unittest.TestCase):
    def test_task_from_dict_cost_is_string(self):
        """cost 保持字符串（十进制），不转 float。"""
        task = Task.from_dict(task_payload())
        self.assertEqual(task.cost, "0.0100")
        self.assertIsInstance(task.cost, str)


if __name__ == "__main__":
    unittest.main()
