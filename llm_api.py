"""DeepSeek OpenAI 兼容 API。Tutor 默认用 deepseek-chat（可用 TUTOR_MODEL 覆盖）。"""

from __future__ import annotations

import json
import os
import re

import httpx
import streamlit as st
from openai import OpenAI


def deepseek_model() -> str:
    return (os.environ.get("DEEPSEEK_MODEL") or "deepseek-chat").strip() or "deepseek-chat"


def tutor_model() -> str:
    """对话与分类专用；默认 deepseek-chat，避免 reasoner 慢/空正文拖垮 Tutor。"""
    return (os.environ.get("TUTOR_MODEL") or "deepseek-chat").strip() or "deepseek-chat"


def tutor_model_chain() -> list[str]:
    p = tutor_model()
    chain = [p]
    for alt in ("deepseek-chat", "deepseek-reasoner"):
        if alt not in chain:
            chain.append(alt)
    return chain[:3]


def _is_reasoner_model(model_name: str) -> bool:
    return "reasoner" in (model_name or "").lower()


def _model_max_completion_tokens(model_name: str) -> int:
    if _is_reasoner_model(model_name):
        return int(os.environ.get("DEEPSEEK_REASONER_MAX_TOKENS", "32768"))
    return int(os.environ.get("DEEPSEEK_CHAT_MAX_TOKENS", "8192"))


def _effective_request_timeout(model_name: str, timeout_sec: int) -> float:
    if _is_reasoner_model(model_name):
        return float(max(timeout_sec, 240))
    return float(max(timeout_sec, 60))


def _message_text_from_response_message(msg) -> str:
    refusal = getattr(msg, "refusal", None)
    if isinstance(refusal, str) and refusal.strip():
        return refusal.strip()
    content = (getattr(msg, "content", None) or "").strip()
    if content:
        return content
    rc = getattr(msg, "reasoning_content", None)
    if isinstance(rc, str) and rc.strip():
        return rc.strip()
    extra = getattr(msg, "model_extra", None) or {}
    if isinstance(extra, dict):
        rc2 = extra.get("reasoning_content")
        if isinstance(rc2, str) and rc2.strip():
            return rc2.strip()
    return ""


from tutor_api_keys import get_session_effective_api_key


def _http_proxy_url() -> str | None:
    return (
        (os.environ.get("DEEPSEEK_HTTP_PROXY") or "").strip()
        or (os.environ.get("HTTPS_PROXY") or "").strip()
        or (os.environ.get("HTTP_PROXY") or "").strip()
        or None
    )


def configure_llm() -> None:
    key = get_session_effective_api_key()
    if not key:
        st.error(
            "当前没有可用的 API Key。请返回上一步选择「平台托管」或填写你自己的 Key；"
            "若你是部署方，请在环境变量或 Streamlit Secrets 中配置 `DEEPSEEK_API_KEY`。"
        )
        st.stop()
    base = (os.environ.get("DEEPSEEK_BASE_URL") or "https://api.deepseek.com").strip().rstrip("/")
    proxy = _http_proxy_url()
    long_timeout = httpx.Timeout(connect=60.0, read=300.0, write=120.0, pool=30.0)

    sig = (key, base, proxy or "", os.environ.get("DEEPSEEK_USE_CUSTOM_HTTPX", ""))
    if (
        st.session_state.get("_deepseek_config_sig") == sig
        and st.session_state.get("_deepseek_client") is not None
    ):
        return

    st.session_state["_deepseek_config_sig"] = sig
    st.session_state["_deepseek_api_key"] = key
    st.session_state["_deepseek_base_url"] = base
    st.session_state["_deepseek_http_proxy"] = proxy

    # 默认不用自定义 http_client：与 OpenAI SDK 自带的 httpx 配合更稳（不少「一直有兜底」是连接/超时叠在一起）
    if (os.environ.get("DEEPSEEK_USE_CUSTOM_HTTPX") or "").strip() in ("1", "true", "yes"):
        http_client = httpx.Client(
            proxy=proxy,
            timeout=long_timeout,
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
        )
        st.session_state["_deepseek_client"] = OpenAI(
            api_key=key,
            base_url=base,
            max_retries=2,
            timeout=long_timeout,
            http_client=http_client,
        )
    else:
        st.session_state["_deepseek_client"] = OpenAI(
            api_key=key,
            base_url=base,
            max_retries=2,
            timeout=long_timeout,
        )


def _get_deepseek_client() -> OpenAI:
    c = st.session_state.get("_deepseek_client")
    if c is None:
        configure_llm()
        c = st.session_state.get("_deepseek_client")
    if c is None:
        raise RuntimeError("DeepSeek 客户端未初始化")
    return c


def _messages_payload(
    system_instruction: str | None, user_text: str
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if system_instruction:
        messages.append({"role": "system", "content": system_instruction})
    messages.append({"role": "user", "content": user_text})
    return messages


def _deepseek_post_chat_completions(
    model: str,
    messages: list[dict[str, str]],
    *,
    temperature: float,
    max_tokens: int,
    timeout_sec: float,
    json_object: bool,
) -> tuple[str | None, dict]:
    """SDK 失败或空正文时，用 httpx 直连 DeepSeek OpenAI 兼容接口。"""
    base = str(st.session_state.get("_deepseek_base_url") or "https://api.deepseek.com").rstrip("/")
    key = str(st.session_state.get("_deepseek_api_key") or "")
    url = f"{base}/v1/chat/completions"
    body: dict = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if json_object:
        body["response_format"] = {"type": "json_object"}
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    proxy = st.session_state.get("_deepseek_http_proxy")
    tmo = httpx.Timeout(
        connect=45.0, read=max(float(timeout_sec), 90.0), write=120.0, pool=30.0
    )
    detail: dict = {"via": "httpx", "status_code": None}
    try:
        with httpx.Client(proxy=proxy, timeout=tmo) as client:
            r = client.post(url, headers=headers, json=body)
        detail["status_code"] = r.status_code
        if r.status_code != 200:
            detail["err"] = "HTTPError"
            detail["msg"] = (r.text or "")[:500]
            return None, detail
        data = r.json()
        choices = data.get("choices") or []
        if not choices:
            detail["msg"] = "empty choices"
            return None, detail
        msg = choices[0].get("message") or {}
        ref = msg.get("refusal")
        if isinstance(ref, str) and ref.strip():
            detail["refusal"] = ref[:300]
            return ref.strip(), detail
        content = (msg.get("content") or "").strip()
        if not content:
            rc = msg.get("reasoning_content")
            if isinstance(rc, str) and rc.strip():
                content = rc.strip()
        detail["finish_reason"] = choices[0].get("finish_reason")
        return (content or None), detail
    except Exception as e:
        detail["err"] = type(e).__name__
        detail["msg"] = str(e)[:400]
        return None, detail


def _strip_markdown_json_fence(text: str) -> str:
    t = text.strip()
    m = re.match(r"^```(?:json)?\s*([\s\S]*?)```\s*$", t, re.I)
    if m:
        return m.group(1).strip()
    return t


JSON_ONLY = "只输出一个 JSON 对象，不要 Markdown 代码围栏或其他文字。"


def _parse_json_object(text: str) -> dict:
    text = _strip_markdown_json_fence(text.strip())
    if not text:
        return {}
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else {}
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            obj = json.loads(m.group(0))
            return obj if isinstance(obj, dict) else {}
        except json.JSONDecodeError:
            pass
    return {}


def _coerce_message_value(data: object) -> str | None:
    if not isinstance(data, dict):
        return None
    for k in ("message", "reply", "text", "content", "student_message", "answer_to_student"):
        v = data.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
        if isinstance(v, dict):
            inner = v.get("text") or v.get("content") or v.get("message")
            if isinstance(inner, str) and inner.strip():
                return inner.strip()
    for key, v in data.items():
        if not isinstance(key, str):
            continue
        if key.lower() in ("message", "reply", "text") and isinstance(v, str) and v.strip():
            return v.strip()
    return None


def _coerce_message_any(data: object) -> str | None:
    msg = _coerce_message_value(data)
    if msg:
        return msg
    if not isinstance(data, dict):
        return None
    for k in ("回复", "消息", "内容", "输出", "学生回复", "导师回复", "assistant"):
        v = data.get(k)
        if isinstance(v, str) and len(v.strip()) > 2:
            return v.strip()
    best = None
    best_len = 0
    for v in data.values():
        if isinstance(v, str) and len(v.strip()) > best_len:
            best = v.strip()
            best_len = len(best)
    if best_len >= 8:
        return best
    return None


def _generate_raw(
    model_names: list[str],
    system_instruction: str | None,
    user_text: str,
    temperature: float,
    timeout_sec: int,
    *,
    retries_per_variant: int = 2,
    json_mode: bool = False,
) -> tuple[str | None, dict]:
    meta: dict = {"attempts": []}
    client = _get_deepseek_client()
    json_flags = (True, False) if json_mode else (False,)
    eff_timeout = lambda mn: _effective_request_timeout(mn, timeout_sec)
    max_tok = _model_max_completion_tokens

    for mname in model_names:
        for use_json_fmt in json_flags:
            for _ in range(max(1, retries_per_variant)):
                try:
                    messages = _messages_payload(system_instruction, user_text)
                    kwargs: dict = {
                        "model": mname,
                        "messages": messages,
                        "temperature": temperature,
                        "timeout": eff_timeout(mname),
                        "max_tokens": max_tok(mname),
                    }
                    if use_json_fmt:
                        kwargs["response_format"] = {"type": "json_object"}
                    r = client.chat.completions.create(**kwargs)
                    content = ""
                    fr = None
                    if r.choices:
                        content = _message_text_from_response_message(
                            r.choices[0].message
                        ).strip()
                        fr = getattr(r.choices[0], "finish_reason", None)
                    row = {
                        "model": mname,
                        "json_fmt": use_json_fmt,
                        "chars": len(content),
                        "finish_reason": fr,
                        "via": "openai_sdk",
                    }
                    meta["attempts"].append(row)
                    if content:
                        meta["ok_model"] = mname
                        meta["ok_json_fmt"] = use_json_fmt
                        return content, meta
                except Exception as e:
                    meta["attempts"].append(
                        {
                            "model": mname,
                            "json_fmt": use_json_fmt,
                            "err": type(e).__name__,
                            "msg": str(e)[:240],
                            "via": "openai_sdk",
                        }
                    )

    # —— httpx 直连（避免部分环境下 SDK/证书/连接池异常）——
    for mname in model_names:
        for use_json_fmt in json_flags:
            messages = _messages_payload(system_instruction, user_text)
            text, hd = _deepseek_post_chat_completions(
                mname,
                messages,
                temperature=temperature,
                max_tokens=max_tok(mname),
                timeout_sec=eff_timeout(mname),
                json_object=use_json_fmt,
            )
            row = {
                "via": "httpx",
                "model": mname,
                "json_fmt": use_json_fmt,
                "chars": len(text or ""),
                **{k: v for k, v in hd.items() if k != "via"},
            }
            meta["attempts"].append(row)
            if text:
                meta["ok_model"] = mname
                meta["ok_json_fmt"] = use_json_fmt
                return text.strip(), meta

    return None, meta


def _trace_suggests_insufficient_balance(attempts: list, detail: str) -> bool:
    blob = detail + json.dumps(attempts, ensure_ascii=False)
    if "Insufficient Balance" in blob or "insufficient balance" in blob.lower():
        return True
    return any(a.get("status_code") == 402 for a in attempts)


def summarize_api_trace(meta: dict) -> str:
    """给人看的简短失败原因（用于兜底文案与调试）。"""
    attempts = meta.get("attempts") or []
    if not attempts:
        out = "无任何 API 调用记录。"
        return out
    errs = [a for a in attempts if a.get("err")]
    if errs:
        last = errs[-1]
        out = f"{last.get('via', '?')}: {last.get('err')} — {last.get('msg', '')[:200]}"
    else:
        bad_http = [a for a in attempts if a.get("status_code") and a.get("status_code") != 200]
        if bad_http:
            b = bad_http[-1]
            out = f"HTTP {b.get('status_code')}: {(b.get('msg') or '')[:200]}"
        else:
            zeros = [a for a in attempts if a.get("chars") == 0]
            if zeros:
                z = zeros[-1]
                fr = z.get("finish_reason")
                rf = z.get("refusal")
                if rf:
                    out = f"模型拒绝回答（refusal 摘要）：{rf!r}"
                else:
                    out = f"多次返回空正文（finish_reason={fr!r}）。"
            else:
                out = "未知原因未取到正文。"
    if _trace_suggests_insufficient_balance(attempts, out):
        return (
            "【DeepSeek 账户余额不足】官方返回 HTTP 402 / Insufficient Balance。"
            "这与「几乎没调用过」不矛盾：余额为 0 时**任何**请求都会失败。"
            "请到 [DeepSeek 开放平台](https://platform.deepseek.com) 充值或更换有余额的 API Key。\n\n"
            f"技术摘要：{out}"
        )
    return out


def run_minimal_api_test() -> tuple[bool, str]:
    """
    单条极短 user 消息，不走 system、不长上下文。
    用于区分：Key/网络/端点 问题 vs. 长 prompt 问题。
    """
    configure_llm()
    messages = [{"role": "user", "content": "只回复两个汉字：好的"}]
    text, hd = _deepseek_post_chat_completions(
        "deepseek-chat",
        messages,
        temperature=0.0,
        max_tokens=32,
        timeout_sec=90.0,
        json_object=False,
    )
    if text and text.strip():
        return True, f"httpx 直连成功，模型返回：{text.strip()[:80]!r}"
    client = _get_deepseek_client()
    try:
        r = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            temperature=0.0,
            max_tokens=32,
            timeout=90.0,
        )
        c = ""
        fr = None
        if r.choices:
            c = _message_text_from_response_message(r.choices[0].message).strip()
            fr = getattr(r.choices[0], "finish_reason", None)
        if c:
            return True, f"SDK 成功，模型返回：{c[:80]!r}"
        return False, f"SDK 200 但正文为空（finish_reason={fr!r}）。"
    except Exception as e:
        sdk_err = f"{type(e).__name__}: {str(e)[:220]}"
    httpx_hint = summarize_api_trace({"attempts": [{**hd, "via": "httpx"}]})
    return False, f"极简请求仍失败。SDK：{sdk_err}；httpx：{httpx_hint}"


def llm_json(
    system: str,
    user: str,
    temperature: float = 0.1,
    retries: int = 4,
    timeout_sec: int = 180,
    models: list[str] | None = None,
) -> tuple[dict, dict]:
    full_system = system + "\n" + JSON_ONLY
    chain = models if models is not None else tutor_model_chain()
    text, meta = _generate_raw(
        chain,
        full_system,
        user,
        temperature,
        timeout_sec,
        retries_per_variant=max(1, retries),
        json_mode=True,
    )
    if not text:
        return {}, meta
    return _parse_json_object(text), meta


def llm_text_reply(
    system: str,
    user: str,
    *,
    temperature: float = 0.35,
    timeout_sec: int = 180,
    retries: int = 4,
    models: list[str] | None = None,
) -> tuple[str | None, dict]:
    chain = models if models is not None else tutor_model_chain()
    return _generate_raw(
        chain,
        system,
        user,
        temperature,
        timeout_sec,
        retries_per_variant=max(1, retries),
        json_mode=False,
    )
