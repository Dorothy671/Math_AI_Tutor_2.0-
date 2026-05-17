"""
DeepSeek API Key：平台托管（环境变量 / secrets）与当前会话（自备 Key）解析。
单独模块，避免 main 与 llm_api 循环依赖及导入名不一致问题。
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping

import streamlit as st


def _normalize_deepseek_key(key: str) -> str:
    k = (key or "").strip()
    if not k or k.startswith("sk-"):
        return k
    if re.fullmatch(r"[a-fA-F0-9]{24,64}", k):
        return "sk-" + k
    return k


def normalize_deepseek_api_key(key: str) -> str:
    """供 UI 校验/保存用户输入的 Key。"""
    return _normalize_deepseek_key(key)


def _touch_streamlit_secrets_parse() -> None:
    """
    触发 st.secrets 完成解析。解析后，TOML 顶层的字符串键会写入 os.environ（Streamlit 行为）。
    若先读环境变量再从未访问过 st.secrets，可能漏掉 Cloud 注入的密钥。
    """
    try:
        _ = list(st.secrets.keys())
    except Exception:
        pass


def _scalar_secret_value(v: object) -> str | None:
    if v is None:
        return None
    if isinstance(v, Mapping):
        return None
    s = str(v).strip()
    return s or None


def _streamlit_secret_first(*keys: str) -> str:
    """
    从 st.secrets 按顺序取第一个非空值。
    支持：① 顶层键；② 写在 TOML 表（[xxx]）下的键（常见误配导致读不到）。
    """
    try:
        sec = st.secrets
        for k in keys:
            try:
                v = sec[k]
            except Exception:
                v = None
            got = _scalar_secret_value(v)
            if got:
                return got

        try:
            for _section, sub in sec.items():
                if not isinstance(sub, Mapping):
                    continue
                for k in keys:
                    try:
                        inner = sub[k]
                    except Exception:
                        continue
                    got = _scalar_secret_value(inner)
                    if got:
                        return got
        except Exception:
            pass
    except Exception:
        pass
    return ""


def get_built_in_api_key() -> str:
    """
    部署方在环境变量或 Streamlit secrets 中配置的「平台托管」Key。
    未配置时返回空串。
    """
    _touch_streamlit_secrets_parse()
    key = (
        os.environ.get("DEEPSEEK_API_KEY", "")
        or os.environ.get("OPENAI_API_KEY", "")
    ).strip()
    if not key:
        key = _streamlit_secret_first(
            "DEEPSEEK_API_KEY",
            "OPENAI_API_KEY",
            "GOOGLE_API_KEY",
        )
    return _normalize_deepseek_key(key)


def get_session_effective_api_key() -> str:
    """
    tutor_api_mode == 'hosted' → 平台内置；'own' → tutor_own_api_key。
    未完成门闸时返回空串。
    """
    mode = st.session_state.get("tutor_api_mode")
    if mode == "own":
        return _normalize_deepseek_key(str(st.session_state.get("tutor_own_api_key") or ""))
    if mode == "hosted":
        return get_built_in_api_key()
    return ""
