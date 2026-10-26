"""调用百炼上的千问模型（OpenAI 兼容接口），带重试、磁盘缓存和多模型轮换。

多模型轮换：百炼给每个模型单独的免费额度。在控制台开启“免费额度用完即停”后，
额度用完的模型会返回 AllocationQuota.FreeTierOnly 错误而不是扣费；
这里遇到这个错误就自动换下一个模型继续，全部用完才停止。
"""
import hashlib
import json
import random
import threading
import time
from pathlib import Path
from typing import List, Optional, Tuple

from .config import Config


class LLMError(RuntimeError):
    pass


class _SwitchModel(Exception):
    """当前模型不可用（免费额度用完 / 模型未开通），应换下一个模型。"""


# 换模型的错误：免费额度用完、模型不存在或未开通
_SWITCH_MARKERS = ("FreeTierOnly", "AllocationQuota", "model_not_found", "Model.NotFound",
                   "does not exist", "Model not exist", "ModelNotFound", "not supported")
# 不重试也不换模型的错误：账户级问题、内容审核
_FATAL_MARKERS = ("InvalidApiKey", "invalid_api_key", "401", "Arrearage", "data_inspection_failed")


class QwenClient:
    def __init__(self, cfg: Config, use_cache: bool = True):
        self.cfg = cfg
        self.use_cache = use_cache
        self.cache_dir = cfg.work_dir / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.models: List[str] = list(cfg.models) if cfg.models else [cfg.model]
        self._idx = 0
        self._lock = threading.Lock()
        self._client = None

    # ---------- 缓存 ----------
    def _key(self, model: str, messages, temperature: float, sample: int) -> str:
        d = {"model": model, "thinking": self.cfg.enable_thinking,
             "temperature": temperature, "messages": messages}
        if sample:
            d["sample"] = sample
        payload = json.dumps(d, ensure_ascii=False, sort_keys=True)
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    # ---------- 请求 ----------
    def _get_client(self):
        if self._client is None:
            if not self.cfg.api_key:
                raise LLMError("没有找到 API Key：请在项目根目录的 .env 中设置 DASHSCOPE_API_KEY=sk-xxx")
            from openai import OpenAI
            self._client = OpenAI(api_key=self.cfg.api_key, base_url=self.cfg.base_url,
                                  timeout=self.cfg.timeout, max_retries=0)
        return self._client

    def _call_once(self, model: str, messages, temperature: float) -> Tuple[str, str, dict]:
        """流式调用（思考模式只支持流式），返回 (content, reasoning, usage)。"""
        client = self._get_client()
        kwargs = dict(model=model, messages=messages, stream=True,
                      stream_options={"include_usage": True},
                      max_tokens=self.cfg.max_tokens)
        extra = {"enable_thinking": bool(self.cfg.enable_thinking)}
        if not self.cfg.enable_thinking:
            kwargs["temperature"] = temperature
        stream = client.chat.completions.create(extra_body=extra, **kwargs)
        content, reasoning, usage = [], [], {}
        for chunk in stream:
            if getattr(chunk, "usage", None):
                u = chunk.usage
                usage = {"prompt_tokens": u.prompt_tokens, "completion_tokens": u.completion_tokens}
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if getattr(delta, "content", None):
                content.append(delta.content)
            rc = getattr(delta, "reasoning_content", None)
            if rc:
                reasoning.append(rc)
        return "".join(content), "".join(reasoning), usage

    def _call_with_retries(self, model: str, messages, temperature: float, cp: Path) -> dict:
        last_err: Optional[Exception] = None
        for attempt in range(self.cfg.max_retries):
            t0 = time.time()
            try:
                content, reasoning, usage = self._call_once(model, messages, temperature)
                if not content.strip():
                    raise LLMError("模型返回空内容")
                data = {"content": content, "reasoning": reasoning, "usage": usage,
                        "model": model, "seconds": round(time.time() - t0, 1)}
                cp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
                data["cached"] = False
                return data
            except Exception as e:  # noqa: BLE001
                last_err = e
                msg = str(e)
                if any(s in msg for s in _SWITCH_MARKERS):
                    raise _SwitchModel(msg)
                if any(s in msg for s in _FATAL_MARKERS):
                    break
                time.sleep(min(60, 2 ** attempt * 3) + random.random() * 2)
        raise LLMError(f"{type(last_err).__name__}: {last_err}")

    def chat(self, messages, temperature: Optional[float] = None, sample: int = 0) -> dict:
        """返回 {"content", "reasoning", "usage", "model", "cached", "seconds"}。"""
        temperature = self.cfg.temperature if temperature is None else temperature

        # 任何一个候选模型有缓存就直接用（不花钱）
        if self.use_cache:
            for m in self.models:
                cp = self._cache_path(self._key(m, messages, temperature, sample))
                if cp.exists():
                    data = json.loads(cp.read_text(encoding="utf-8"))
                    data.setdefault("model", m)
                    data["cached"] = True
                    return data

        while True:
            with self._lock:
                idx = self._idx
            if idx >= len(self.models):
                raise LLMError("AllocationQuota.FreeTierOnly：候选模型的免费额度都已用完（"
                               + "、".join(self.models) + "）")
            model = self.models[idx]
            cp = self._cache_path(self._key(model, messages, temperature, sample))
            try:
                return self._call_with_retries(model, messages, temperature, cp)
            except _SwitchModel as e:
                with self._lock:
                    if self._idx == idx:
                        self._idx += 1
                        nxt = self.models[self._idx] if self._idx < len(self.models) else "（没有了）"
                        print(f"\n>>> 模型 {model} 不可用（{str(e)[:80]}），切换到 {nxt}\n", flush=True)
