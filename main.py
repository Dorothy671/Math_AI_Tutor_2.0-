"""
Streamlit 数学 Tutor：questions.xlsx + scaffolding.xlsx（含 behavior / scaffold / focus 等列）。
每轮按 trigger_condition 判定 behavior，苏格拉底式引导；侧栏进度与调试。
"""

from __future__ import annotations

import copy
from datetime import datetime
import json
import re
from pathlib import Path

import pandas as pd
import streamlit as st

from llm_api import configure_llm, run_minimal_api_test, summarize_api_trace
from tutor_api_keys import get_built_in_api_key, normalize_deepseek_api_key
from learning_summary import (
    FINAL_FORCE_LECTURE,
    build_summary_markdown,
    enrich_practice_history,
    is_incorrect_behavior,
    normalize_final_behavior,
)
from observer import observer_solution_autonomous_ok, observer_solution_restate_ok
from tutor_core import (
    BEHAVIOR_BEISHU_ENUM_PENDING,
    BEHAVIOR_CORRECT,
    classify_behavior,
    clear_solution_accum_marks,
    generate_completion_feedback_reply,
    generate_force_lecture_reply,
    generate_math_logic_wrapup_reply,
    generate_tutor_reply,
    accum_three_dims_all_correct,
    merge_observer_accum,
    reply_fallback,
    RESTATE_SOLUTION_INSTRUCTION,
    TONE_CONFIRMATORY_REPLY,
)

QUESTIONS_PATH = Path(__file__).resolve().parent / "questions.xlsx"
SCAFFOLDING_PATH = Path(__file__).resolve().parent / "scaffolding.xlsx"

def load_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(str(path))
    suf = path.suffix.lower()
    if suf in (".xlsx", ".xlsm"):
        df = pd.read_excel(path, engine="openpyxl")
    elif suf == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"不支持：{path}")
    df.columns = [str(c).strip() for c in df.columns]
    df = df.dropna(how="all").dropna(axis=1, how="all")
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)
    return df


def load_questions(path: Path) -> pd.DataFrame:
    df = load_table(path)
    for col in ("order", "title", "math_logic", "question_text", "answer", "strategy"):
        if col not in df.columns:
            raise ValueError(f"questions.xlsx 缺少列：{col}，当前列：{list(df.columns)}")
    df["order"] = df["order"].ffill()
    df["title"] = df["title"].ffill()
    # 与 order/title 相同：合并单元格时仅首行有值，需前向填充否则小结里多数题 math_logic 为空
    df["math_logic"] = df["math_logic"].ffill()
    return df


def load_scaffolding(path: Path) -> pd.DataFrame:
    df = load_table(path)
    for col in ("behavior", "scaffold"):
        if col not in df.columns:
            raise ValueError(
                f"scaffolding.xlsx 缺少列：{col}，当前列：{list(df.columns)}"
            )
    if "trigger_condition" not in df.columns:
        df["trigger_condition"] = ""
    if "focus" not in df.columns:
        df["focus"] = ""
    return df


def orders_a1_a7(df: pd.DataFrame) -> list[str]:
    seen = []
    for o in df["order"].dropna().unique():
        s = str(o).strip()
        if s and s not in seen:
            seen.append(s)
    want = [f"A{i}" for i in range(1, 8)]
    ordered = [x for x in want if x in seen]
    ordered += [x for x in seen if x not in ordered]
    return ordered


def row_for_order(df: pd.DataFrame, order: str) -> pd.Series:
    sub = df[df["order"].astype(str).str.strip() == str(order).strip()]
    if sub.empty:
        raise ValueError(f"无 order={order}")
    return sub.iloc[0]


def _excel_signature(path: Path) -> tuple[int, int]:
    stt = path.stat()
    return (int(getattr(stt, "st_mtime_ns", int(stt.st_mtime * 1e9))), stt.st_size)


def ensure_workbooks_loaded() -> None:
    if not QUESTIONS_PATH.exists():
        st.error(f"找不到题目文件：{QUESTIONS_PATH.resolve()}")
        st.stop()
    if not SCAFFOLDING_PATH.exists():
        st.error(f"找不到脚手架文件：{SCAFFOLDING_PATH.resolve()}")
        st.stop()
    q_sig = _excel_signature(QUESTIONS_PATH)
    s_sig = _excel_signature(SCAFFOLDING_PATH)
    if (
        st.session_state.get("tutor_q_sig") != q_sig
        or st.session_state.get("tutor_s_sig") != s_sig
        or "tutor_questions_df" not in st.session_state
    ):
        st.session_state.tutor_questions_df = load_questions(QUESTIONS_PATH)
        st.session_state.tutor_scaffolding_df = load_scaffolding(SCAFFOLDING_PATH)
        st.session_state.tutor_orders = orders_a1_a7(st.session_state.tutor_questions_df)
        st.session_state.tutor_q_sig = q_sig
        st.session_state.tutor_s_sig = s_sig


def enriched_practice_history() -> list[dict]:
    """生成小结用：补全 math_logic 等字段。"""
    return enrich_practice_history(
        list(st.session_state.get("tutor_practice_history") or []),
        st.session_state.get("tutor_questions_df"),
    )


def scaffolding_row_for_behavior(sdf: pd.DataFrame, behavior: str) -> tuple[str, str]:
    """返回 (scaffold, focus)；focus 来自表中 focus 列，供导师生成回复时内化。"""
    b = str(behavior or "").strip()
    m = sdf[sdf["behavior"].astype(str).str.strip() == b]
    if m.empty:
        m = sdf[sdf["behavior"].astype(str).str.contains(re.escape(b), na=False, regex=True)]
    if m.empty:
        return (
            "请结合本题引导学生自查：先后手判断与每轮拿取策略是否自洽。",
            "",
        )
    row = m.iloc[0]
    sc = str(row.get("scaffold", "") or "").strip()
    if not sc:
        sc = "请结合本题引导学生自查：先后手判断与每轮拿取策略是否自洽。"
    focus = str(row.get("focus", "") or "").strip()
    return sc, focus


def scaffold_for_behavior(sdf: pd.DataFrame, behavior: str) -> str:
    return scaffolding_row_for_behavior(sdf, behavior)[0]


def mastery_state_for_prompt(stats: dict) -> dict:
    return {
        "position": str(stats.get("accum_position") or "缺失"),
        "strategy": str(stats.get("accum_strategy") or "缺失"),
        "pattern": str(stats.get("accum_pattern") or "缺失"),
    }


def ensure_question_message_start(order: str) -> dict:
    stats = ensure_q_stats(order)
    if stats.get("message_start_idx") is None:
        stats["message_start_idx"] = len(st.session_state.get("tutor_messages") or [])
    return stats


def messages_for_question(stats: dict) -> list[dict]:
    msgs = list(st.session_state.get("tutor_messages") or [])
    start = stats.get("message_start_idx")
    if not isinstance(start, int) or start < 0 or start > len(msgs):
        start = 0
    return msgs[start:]


def build_chat_export_markdown() -> str:
    orders = list(st.session_state.get("tutor_orders") or [])
    df = st.session_state.get("tutor_questions_df")
    done = bool(st.session_state.get("tutor_all_done"))
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines: list[str] = [
        "# 数学 Tutor 本轮聊天记录",
        "",
        f"- 导出时间：{now}",
    ]
    if done:
        lines.append("- 当前状态：已完成全部题目")
    elif orders and df is not None:
        try:
            co = current_order(orders)
            meta = row_for_order(df, co)
            lines.append(f"- 当前题目：{co} {str(meta.get('title', '') or '')}")
        except Exception:
            lines.append("- 当前题目：读取失败")
    lines.append(f"- 已完成题数：{len(st.session_state.get('tutor_practice_history') or [])}")
    lines.extend(["", "## 聊天记录", ""])
    messages = list(st.session_state.get("tutor_messages") or [])
    if messages:
        for i, msg in enumerate(messages, start=1):
            role = str(msg.get("role", "") or "").strip()
            role_name = "学生" if role == "user" else "Tutor" if role == "assistant" else role or "未知"
            content = str(msg.get("content", "") or "").strip()
            lines.extend([f"### {i}. {role_name}", "", content or "（空）", ""])
    else:
        lines.extend(["暂无聊天记录。", ""])

    history = enriched_practice_history()
    lines.extend(["## 已完成题目", ""])
    if history:
        for rec in history:
            lines.append(
                f"- {rec.get('order', '')} {rec.get('title', '')}："
                f"{rec.get('final_behavior', '')}，轮次 {rec.get('total_rounds', 0)}，"
                f"错误次数 {rec.get('incorrect_count', 0)}"
            )
        lines.append("")
        lines.extend(["## 学习小结", "", build_summary_markdown(history), ""])
    else:
        lines.extend(["暂无已完成题目。", ""])
    return "\n".join(lines).strip() + "\n"


def _default_q_stats() -> dict:
    return {
        "behavior_sequence": [],
        "total_rounds": 0,
        "incorrect_count": 0,
        # 同一题跨轮累积（Observer 三维度；「缺失」表示尚未从模型得到有效判定）
        "accum_position": "缺失",
        "accum_strategy": "缺失",
        "accum_pattern": "缺失",
        # 有累积标记从 False→True 的轮次数（用于「多轮凑齐才需复述」）
        "accum_progress_turns": 0,
        # 各维度首次在 total_rounds 上变为「正确」的轮次（保留作调试/追踪）
        "accum_position_first_correct_round": None,
        "accum_strategy_first_correct_round": None,
        "accum_pattern_first_correct_round": None,
        # 0=未在枚举必败引导流；1=等待学生在「第一轮引导」后的作答（可再给一次机会）
        "beishu_enum_stage": 0,
        "beishu_enum_fail_count": 0,
        # None | {"kind":"tone"} | {"kind":"restate"} | {"kind":"beishu_restate"}
        "pending_confirmation": None,
        # 当前题在 tutor_messages 中的起点；只在题目真正进入时写入
        "message_start_idx": None,
    }


def _migrate_legacy_accum_marks(s: dict) -> None:
    """旧版 bool 累积字段 → 三态字符串。"""
    legacy_map = (
        ("accum_position_ok", "accum_position"),
        ("accum_strategy_ok", "accum_strategy"),
        ("accum_beishu_pattern_ok", "accum_pattern"),
    )
    touched = False
    for old_k, new_k in legacy_map:
        if old_k not in s:
            continue
        touched = True
        if new_k not in s:
            s[new_k] = "正确" if s.get(old_k) else "缺失"
        s.pop(old_k, None)
    # 旧会话曾三维度均「对」但无首次正确轮次记录：按当前 total_rounds 填成同值，避免误触发强制复述
    if touched and all(s.get(nk) == "正确" for _, nk in legacy_map):
        tr = int(s.get("total_rounds") or 1)
        for fk in (
            "accum_position_first_correct_round",
            "accum_strategy_first_correct_round",
            "accum_pattern_first_correct_round",
        ):
            if s.get(fk) is None:
                s[fk] = tr


def ensure_q_stats(order: str) -> dict:
    st.session_state.tutor_q_stats.setdefault(order, _default_q_stats())
    s = st.session_state.tutor_q_stats[order]
    _migrate_legacy_accum_marks(s)
    for k, v in _default_q_stats().items():
        s.setdefault(k, v)
    return s


def init_session(orders: list[str]) -> None:
    if "tutor_order_idx" not in st.session_state:
        st.session_state.tutor_order_idx = 0
    if "tutor_messages" not in st.session_state:
        st.session_state.tutor_messages = []
    if "tutor_started" not in st.session_state:
        st.session_state.tutor_started = False
    if "tutor_all_done" not in st.session_state:
        st.session_state.tutor_all_done = False
    if "tutor_q_stats" not in st.session_state:
        st.session_state.tutor_q_stats = {o: _default_q_stats() for o in orders}
    if "tutor_practice_history" not in st.session_state:
        st.session_state.tutor_practice_history = []
    if "tutor_undo_snapshots" not in st.session_state:
        st.session_state.tutor_undo_snapshots = []
    for o in orders:
        ensure_q_stats(o)


def current_order(orders: list[str]) -> str:
    return orders[st.session_state.tutor_order_idx]


def finish_question_record(order: str, meta: pd.Series, final_behavior: str) -> None:
    stats = st.session_state.tutor_q_stats.get(order, {})
    snaps = st.session_state.setdefault("tutor_undo_snapshots", [])
    snaps.append(
        {
            "order": order,
            "order_idx": st.session_state.tutor_order_idx,
            "messages_len": len(st.session_state.tutor_messages),
            "q_stats": copy.deepcopy(st.session_state.tutor_q_stats),
            "practice_history_len": len(st.session_state.tutor_practice_history),
        }
    )
    st.session_state.tutor_q_stats.pop(order, None)
    st.session_state.tutor_practice_history.append(
        {
            "order": order,
            "title": str(meta.get("title", order)),
            "math_logic": str(meta.get("math_logic", "") or ""),
            "total_rounds": int(stats.get("total_rounds", 0)),
            "behavior_sequence": list(stats.get("behavior_sequence", [])),
            "incorrect_count": int(stats.get("incorrect_count", 0)),
            "final_behavior": normalize_final_behavior(final_behavior),
        }
    )
    orders = st.session_state.tutor_orders
    if st.session_state.tutor_order_idx + 1 < len(orders):
        st.session_state.tutor_order_idx += 1
        no = orders[st.session_state.tutor_order_idx]
        next_stats = st.session_state.tutor_q_stats.setdefault(no, _default_q_stats())
        if next_stats.get("message_start_idx") is None:
            # 结题回复会同时包含上一题总结和下一题题干；下一题分析从其后的学生回复开始。
            next_stats["message_start_idx"] = len(st.session_state.tutor_messages) + 1
    else:
        st.session_state.tutor_all_done = True


def restore_previous_question() -> str | None:
    snaps: list = st.session_state.get("tutor_undo_snapshots") or []
    if not snaps:
        return None
    snap = snaps.pop()
    orders = st.session_state.tutor_orders
    df = st.session_state.tutor_questions_df
    st.session_state.tutor_order_idx = snap["order_idx"]
    st.session_state.tutor_q_stats = copy.deepcopy(snap["q_stats"])
    ml = snap["messages_len"]
    msgs = st.session_state.tutor_messages
    if isinstance(ml, int) and 0 <= ml <= len(msgs):
        st.session_state.tutor_messages = msgs[:ml]
    ph = snap["practice_history_len"]
    hist = st.session_state.tutor_practice_history
    if isinstance(ph, int) and ph <= len(hist):
        st.session_state.tutor_practice_history = hist[:ph]
    st.session_state.tutor_all_done = False
    order = orders[st.session_state.tutor_order_idx]
    meta = row_for_order(df, order)
    return (
        f"已撤回到 **{order}**（{meta.get('title', order)}）。接着聊即可。\n\n"
        f"### {meta['title']}\n{meta['question_text']}"
    )


def jump_to_order(orders: list[str], order: str) -> str:
    if order not in orders:
        return f"无效题号：{order}。有效：**{'、'.join(orders)}**"
    idx = orders.index(order)
    st.session_state.tutor_order_idx = idx
    st.session_state.tutor_all_done = False
    st.session_state.tutor_q_stats[order] = _default_q_stats()
    st.session_state.tutor_q_stats[order]["message_start_idx"] = (
        len(st.session_state.get("tutor_messages") or []) + 1
    )
    meta = row_for_order(st.session_state.tutor_questions_df, order)
    return f"已切换到 **{order}**（{meta.get('title', order)}）。\n\n### {meta['title']}\n{meta['question_text']}"


def parse_jump(user_text: str, orders: list[str]) -> str | None:
    raw = str(user_text or "").strip()
    for o in orders:
        if re.search(rf"\b{re.escape(o)}\b", raw, re.I):
            return o
    return None


def append_turn_debug(entry: dict) -> None:
    buf = st.session_state.setdefault("tutor_turn_debug", [])
    buf.append(entry)
    if len(buf) > 80:
        del buf[:-80]
    lines = [
        "========== Tutor 本轮调试 ==========",
        f"order: {entry.get('order', '')}",
        f"用户: {entry.get('user_text', '')!r}",
        f"分支: {entry.get('branch', '')}",
        f"behavior: {entry.get('behavior', '')}",
        f"判定说明: {entry.get('classify_brief_reason', '')}",
        f"分类模型有输出: {entry.get('classify_llm_ok', '')}",
        f"回复是否兜底: {entry.get('reply_used_fallback', '')}",
        "---- focus（Excel）----",
        (entry.get("focus") or "")[:1200],
        "---- scaffold（Excel）----",
        (entry.get("scaffold") or "")[:2000],
    ]
    tr = entry.get("reply_llm_trace")
    if tr:
        try:
            lines.append("reply attempts: " + json.dumps(tr, ensure_ascii=False)[:1200])
        except Exception:
            lines.append(f"reply attempts: {tr!r}"[:1200])
    try:
        print("\n".join(lines), flush=True)
    except BrokenPipeError:
        pass


def _completion_tail_reply(reply_core: str) -> str:
    if st.session_state.tutor_all_done:
        return (
            reply_core
            + "\n\n---\n\n"
            + build_summary_markdown(enriched_practice_history())
        )
    orders = st.session_state.tutor_orders
    df = st.session_state.tutor_questions_df
    no = current_order(orders)
    nm = row_for_order(df, no)
    return (
        reply_core
        + f"\n\n---\n\n下一题 **{no}**：\n\n### {nm['title']}\n{nm['question_text']}"
    )


def _run_full_completion(
    user_text: str,
    order: str,
    meta: pd.Series,
    sdf: pd.DataFrame,
    stats: dict,
    *,
    debug_branch: str,
    behavior_source: str = "",
    classify_brief: str = "",
    classify_llm_ok=None,
    classify_llm_trace=None,
) -> str:
    seq = list(stats.get("behavior_sequence") or [])
    if not seq or seq[-1] != BEHAVIOR_CORRECT:
        stats["behavior_sequence"].append(BEHAVIOR_CORRECT)
    question_messages = messages_for_question(stats)
    text, gen_meta = generate_completion_feedback_reply(meta, question_messages, stats)
    reply = (text or "").strip() or reply_fallback(
        BEHAVIOR_CORRECT,
        scaffold_for_behavior(sdf, BEHAVIOR_CORRECT),
        meta,
        api_hint=summarize_api_trace(gen_meta),
    )
    used_fb = not bool((text or "").strip())
    append_turn_debug(
        {
            "order": order,
            "user_text": user_text,
            "branch": debug_branch,
            "behavior": BEHAVIOR_CORRECT,
            "behavior_source": behavior_source,
            "classify_brief_reason": classify_brief,
            "classify_llm_ok": classify_llm_ok,
            "classify_llm_trace": classify_llm_trace,
            "scaffold": scaffold_for_behavior(sdf, BEHAVIOR_CORRECT),
            "focus": scaffolding_row_for_behavior(sdf, BEHAVIOR_CORRECT)[1],
            "reply_used_fallback": used_fb,
            "reply_llm_trace": gen_meta.get("attempts"),
        }
    )
    finish_question_record(order, meta, BEHAVIOR_CORRECT)
    return _completion_tail_reply(reply)


def _maybe_complete_accumulated_solution(
    user_text: str,
    order: str,
    meta: pd.Series,
    sdf: pd.DataFrame,
    stats: dict,
    msgs: list,
    *,
    behavior: str,
    cl_detail: dict,
    debug_branch: str,
) -> str | None:
    if not accum_three_dims_all_correct(stats):
        return None

    if bool(cl_detail.get("tone_uncertain")):
        stats["pending_confirmation"] = {"kind": "tone"}
        append_turn_debug(
            {
                "order": order,
                "user_text": user_text,
                "branch": f"{debug_branch}→三维度累计全对但语气不确定→暂缓收题",
                "behavior": behavior,
                "behavior_source": cl_detail.get("behavior_source", ""),
                "classify_brief_reason": cl_detail.get("brief_reason", ""),
                "classify_llm_ok": cl_detail.get("classify_llm_ok"),
                "classify_llm_trace": cl_detail.get("classify_llm_trace"),
                "scaffold": "",
                "reply_used_fallback": False,
                "reply_llm_trace": None,
            }
        )
        return TONE_CONFIRMATORY_REPLY

    question_messages = messages_for_question(stats)
    auto_ok, auto_detail = observer_solution_autonomous_ok(
        user_text, meta, question_messages
    )
    auto_reason = str(auto_detail.get("brief_reason", "") or "")
    classify_brief = str(cl_detail.get("brief_reason", "") or "")
    combined_brief = "；".join(x for x in (classify_brief, auto_reason) if x)
    if auto_ok:
        append_turn_debug(
            {
                "order": order,
                "user_text": user_text,
                "branch": f"{debug_branch}→三维度累计全对且自主推理充分→收题",
                "behavior": behavior,
                "behavior_source": cl_detail.get("behavior_source", ""),
                "classify_brief_reason": combined_brief,
                "classify_llm_ok": cl_detail.get("classify_llm_ok"),
                "classify_llm_trace": cl_detail.get("classify_llm_trace"),
                "scaffold": "",
                "reply_used_fallback": False,
                "reply_llm_trace": None,
            }
        )
        return _run_full_completion(
            user_text,
            order,
            meta,
            sdf,
            stats,
            debug_branch=f"{debug_branch}→三维度累计全对且自主推理充分→收题",
            behavior_source=cl_detail.get("behavior_source", ""),
            classify_brief=combined_brief,
            classify_llm_ok=cl_detail.get("classify_llm_ok"),
            classify_llm_trace=cl_detail.get("classify_llm_trace"),
        )

    stats["pending_confirmation"] = {"kind": "restate"}
    append_turn_debug(
        {
            "order": order,
            "user_text": user_text,
            "branch": f"{debug_branch}→三维度累计全对但由引导问答凑齐→进入复述验收",
            "behavior": behavior,
            "behavior_source": cl_detail.get("behavior_source", ""),
            "classify_brief_reason": combined_brief,
            "classify_llm_ok": cl_detail.get("classify_llm_ok"),
            "classify_llm_trace": cl_detail.get("classify_llm_trace"),
            "scaffold": "",
            "reply_used_fallback": False,
            "reply_llm_trace": None,
        }
    )
    return RESTATE_SOLUTION_INSTRUCTION


def process_user_message(user_text: str) -> str:
    ensure_workbooks_loaded()
    df = st.session_state.tutor_questions_df
    sdf = st.session_state.tutor_scaffolding_df
    orders = st.session_state.tutor_orders

    jo = parse_jump(user_text, orders)
    if jo is not None:
        return jump_to_order(orders, jo)

    if re.search(r"(回到|返回|退回).{0,8}(上一题|上道题|前一题)", str(user_text or "")):
        r = restore_previous_question()
        if r:
            return r
        return "目前没有可撤回的记录。"

    if st.session_state.tutor_all_done:
        return "你已完成全部题目。可点侧栏「重新开始」或「查看学习小结」。"

    order = current_order(orders)
    meta = row_for_order(df, order)
    stats = ensure_question_message_start(order)
    stats["total_rounds"] = stats.get("total_rounds", 0) + 1
    msgs = st.session_state.tutor_messages
    # —— 复述验收 / 枚举讲解后复述 ——（结题前复述失败：多轮收题不清除 Observer 累积）
    pc = stats.get("pending_confirmation")
    if isinstance(pc, dict) and pc.get("kind") in ("restate", "beishu_restate"):
        ok_re, re_detail = observer_solution_restate_ok(user_text, meta, msgs)
        k = str(pc.get("kind") or "")
        stats["pending_confirmation"] = None
        if ok_re:
            return _run_full_completion(
                user_text,
                order,
                meta,
                sdf,
                stats,
                debug_branch=(
                    "复述验收通过→收题"
                    if k == "restate"
                    else "枚举必败·讲解后复述通过→收题"
                ),
                behavior_source="restate_ok",
            )
        if k == "beishu_restate":
            clear_solution_accum_marks(stats)
            stats["accum_progress_turns"] = 0
        rd = re_detail if isinstance(re_detail, dict) else {}
        append_turn_debug(
            {
                "order": order,
                "user_text": user_text,
                "branch": (
                    "复述未通过→保留累积并继续引导（多轮收题）"
                    if k == "restate"
                    else "复述未通过→清空累积（枚举讲解后）"
                ),
                "behavior": "",
                "classify_brief_reason": str(rd.get("brief_reason", "")),
                "classify_llm_ok": None,
                "classify_llm_trace": None,
                "scaffold": "",
                "reply_used_fallback": False,
                "reply_llm_trace": None,
            }
        )
        # 继续走下方常规 classify（同一轮用户输入）

    # —— 语气确认跟进 ——
    pc = stats.get("pending_confirmation")
    if isinstance(pc, dict) and pc.get("kind") == "tone":
        behavior, cl_detail = classify_behavior(
            user_text,
            meta,
            sdf,
            recent_messages=msgs,
        )
        tu = bool(cl_detail.get("tone_uncertain"))
        stats["pending_confirmation"] = None
        merge_observer_accum(stats, cl_detail)
        done = _maybe_complete_accumulated_solution(
            user_text,
            order,
            meta,
            sdf,
            stats,
            msgs,
            behavior=behavior,
            cl_detail={**cl_detail, "tone_uncertain": tu},
            debug_branch="语气确认跟进",
        )
        if done is not None:
            return done
        stats["behavior_sequence"].append(behavior)
        if is_incorrect_behavior(behavior):
            stats["incorrect_count"] = int(stats.get("incorrect_count", 0)) + 1
        scaffold, focus = scaffolding_row_for_behavior(sdf, behavior)
        text, gen_meta = generate_tutor_reply(
            user_text,
            meta,
            behavior,
            scaffold,
            msgs,
            completion_mode=False,
            scaffold_focus=focus,
            mastery_state=mastery_state_for_prompt(stats),
        )
        reply = (text or "").strip() or reply_fallback(
            behavior, scaffold, meta, api_hint=summarize_api_trace(gen_meta)
        )
        used_fb = not bool((text or "").strip())
        append_turn_debug(
            {
                "order": order,
                "user_text": user_text,
                "branch": "语气确认未通过→按 Observer behavior 继续引导",
                "behavior": behavior,
                "behavior_source": cl_detail.get("behavior_source", ""),
                "classify_brief_reason": cl_detail.get("brief_reason", ""),
                "classify_llm_ok": cl_detail.get("classify_llm_ok"),
                "classify_llm_trace": cl_detail.get("classify_llm_trace"),
                "scaffold": scaffold,
                "focus": focus,
                "reply_used_fallback": used_fb,
                "reply_llm_trace": gen_meta.get("attempts"),
            }
        )
        return reply

    if int(stats.get("beishu_enum_stage") or 0) == 1:
        beh_e, det_e = classify_behavior(user_text, meta, sdf, recent_messages=msgs)
        merge_observer_accum(stats, det_e)
        if accum_three_dims_all_correct(stats):
            stats["beishu_enum_stage"] = 0
            stats["beishu_enum_fail_count"] = 0
            done = _maybe_complete_accumulated_solution(
                user_text,
                order,
                meta,
                sdf,
                stats,
                msgs,
                behavior=beh_e,
                cl_detail=det_e,
                debug_branch="枚举必败·第二轮三维度累计收题判定",
            )
            if done is not None:
                return done
        stats["behavior_sequence"].append(beh_e)
        fails = int(stats.get("beishu_enum_fail_count") or 0)
        if fails == 0:
            stats["beishu_enum_fail_count"] = 1
            sc_enum, focus_enum = scaffolding_row_for_behavior(
                sdf, BEHAVIOR_BEISHU_ENUM_PENDING
            )
            text, gen_meta = generate_tutor_reply(
                user_text,
                meta,
                BEHAVIOR_BEISHU_ENUM_PENDING,
                sc_enum,
                msgs,
                completion_mode=False,
                scaffold_focus=focus_enum,
                beishu_enum_phase1_guidance=True,
                mastery_state=mastery_state_for_prompt(stats),
            )
            reply = (text or "").strip() or reply_fallback(
                BEHAVIOR_BEISHU_ENUM_PENDING,
                sc_enum,
                meta,
                api_hint=summarize_api_trace(gen_meta),
            )
            used_fb = not bool((text or "").strip())
            append_turn_debug(
                {
                    "order": order,
                    "user_text": user_text,
                    "branch": "枚举必败·第一轮作答未通过→再给一轮引导",
                    "behavior": BEHAVIOR_BEISHU_ENUM_PENDING,
                    "behavior_source": "beishu_enum_retry",
                    "classify_brief_reason": det_e.get("brief_reason", ""),
                    "classify_llm_ok": det_e.get("classify_llm_ok"),
                    "classify_llm_trace": det_e.get("classify_llm_trace"),
                    "scaffold": sc_enum,
                    "focus": focus_enum,
                    "reply_used_fallback": used_fb,
                    "reply_llm_trace": gen_meta.get("attempts"),
                }
            )
            return reply
        stats["beishu_enum_stage"] = 0
        stats["beishu_enum_fail_count"] = 0
        stats["incorrect_count"] = int(stats.get("incorrect_count", 0)) + 1
        wrap, wm = generate_math_logic_wrapup_reply(user_text, meta, msgs)
        reply_wrap = (wrap or "").strip() or (
            "我暂时无法生成讲解，请对照题单的 math_logic 与 strategy 自行小结。"
        )
        stats["pending_confirmation"] = {"kind": "beishu_restate"}
        append_turn_debug(
            {
                "order": order,
                "user_text": user_text,
                "branch": "枚举必败·连错两轮→教师讲解后进入复述验收",
                "behavior": beh_e,
                "behavior_source": "beishu_enum_round2_wrapup",
                "classify_brief_reason": det_e.get("brief_reason", ""),
                "classify_llm_ok": det_e.get("classify_llm_ok"),
                "classify_llm_trace": det_e.get("classify_llm_trace"),
                "scaffold": "",
                "reply_used_fallback": False,
                "reply_llm_trace": wm.get("attempts"),
            }
        )
        return reply_wrap + "\n\n" + RESTATE_SOLUTION_INSTRUCTION

    behavior, cl_detail = classify_behavior(
        user_text,
        meta,
        sdf,
        recent_messages=msgs,
    )
    if behavior != BEHAVIOR_CORRECT:
        stats["behavior_sequence"].append(behavior)
    merge_observer_accum(stats, cl_detail)
    if is_incorrect_behavior(behavior):
        stats["incorrect_count"] = int(stats.get("incorrect_count", 0)) + 1

    done = _maybe_complete_accumulated_solution(
        user_text,
        order,
        meta,
        sdf,
        stats,
        msgs,
        behavior=behavior,
        cl_detail=cl_detail,
        debug_branch="三维度累计收题判定",
    )
    if done is not None:
        return done

    scaffold, focus = scaffolding_row_for_behavior(sdf, behavior)

    if behavior == BEHAVIOR_CORRECT:
        stats["behavior_sequence"].append(behavior)

    if behavior == BEHAVIOR_BEISHU_ENUM_PENDING:
        text, gen_meta = generate_tutor_reply(
            user_text,
            meta,
            behavior,
            scaffold,
            st.session_state.tutor_messages,
            completion_mode=False,
            scaffold_focus=focus,
            beishu_enum_phase1_guidance=True,
            mastery_state=mastery_state_for_prompt(stats),
        )
        reply = (text or "").strip() or reply_fallback(
            behavior, scaffold, meta, api_hint=summarize_api_trace(gen_meta)
        )
        used_fb = not bool((text or "").strip())
        stats["beishu_enum_stage"] = 1
        stats["beishu_enum_fail_count"] = 0
        append_turn_debug(
            {
                "order": order,
                "user_text": user_text,
                "branch": "枚举必败·第一轮引导（下一轮起比对规律；连错两轮则讲解后复述）",
                "behavior": behavior,
                "behavior_source": cl_detail.get("behavior_source", ""),
                "classify_brief_reason": cl_detail.get("brief_reason", ""),
                "classify_llm_ok": cl_detail.get("classify_llm_ok"),
                "classify_llm_trace": cl_detail.get("classify_llm_trace"),
                "scaffold": scaffold,
                "focus": focus,
                "reply_used_fallback": used_fb,
                "reply_llm_trace": gen_meta.get("attempts"),
            }
        )
        return reply

    text, gen_meta = generate_tutor_reply(
        user_text,
        meta,
        behavior,
        scaffold,
        st.session_state.tutor_messages,
        completion_mode=False,
        scaffold_focus=focus,
        mastery_state=mastery_state_for_prompt(stats),
    )
    reply = (text or "").strip() or reply_fallback(
        behavior, scaffold, meta, api_hint=summarize_api_trace(gen_meta)
    )
    used_fb = not bool((text or "").strip())
    append_turn_debug(
        {
            "order": order,
            "user_text": user_text,
            "branch": "苏格拉底引导",
            "behavior": behavior,
            "behavior_source": cl_detail.get("behavior_source", ""),
            "classify_brief_reason": cl_detail.get("brief_reason", ""),
            "classify_llm_ok": cl_detail.get("classify_llm_ok"),
            "classify_llm_trace": cl_detail.get("classify_llm_trace"),
            "scaffold": scaffold,
            "focus": focus,
            "reply_used_fallback": used_fb,
            "reply_llm_trace": gen_meta.get("attempts"),
        }
    )
    return reply


def ensure_api_connection_selected() -> None:
    """首次进入须选择平台托管 Key 或自备 Key；未完成则渲染门闸并 stop。"""
    if st.session_state.get("tutor_api_mode") in ("hosted", "own"):
        return

    builtin = get_built_in_api_key()
    st.subheader("第一步：选择 AI 连接方式")
    st.markdown(
        "对话、判题与讲解使用 **DeepSeek** 兼容接口（默认 `https://api.deepseek.com`）。\n\n"
        "你可以选择 **使用平台提供的连接**（由部署方配置 Key，你无需填写），"
        "或 **使用你自己的 API Key**（调用费用计入你的 DeepSeek 账户）。\n\n"
        "自备 Key **仅保存在当前浏览器会话**中，服务器不会写入题库或配置文件。"
    )
    if builtin:
        choice = st.radio(
            "请选择",
            options=["hosted", "own"],
            format_func=lambda v: (
                "使用平台提供的连接（无需填写 Key）"
                if v == "hosted"
                else "使用我自己的 DeepSeek API Key"
            ),
            key="tutor_api_gate_choice",
        )
    else:
        st.info("当前部署未配置平台托管 Key，请使用你自己的 DeepSeek API Key。")
        choice = "own"

    own_key = ""
    if choice == "own":
        own_key = st.text_input(
            "DeepSeek API Key",
            type="password",
            placeholder="sk-…",
            help="在 DeepSeek 开放平台创建；仅本次浏览器会话使用。",
            key="tutor_api_gate_key_input",
        )

    if st.button("确认并进入", type="primary", key="tutor_api_gate_go"):
        if choice == "hosted":
            if not get_built_in_api_key():
                st.error("平台 Key 不可用，请选用自备 Key 或联系部署方。")
            else:
                st.session_state["tutor_api_mode"] = "hosted"
                st.rerun()
        else:
            nk = normalize_deepseek_api_key(own_key)
            if not nk:
                st.warning("请先粘贴有效的 API Key。")
            else:
                st.session_state["tutor_api_mode"] = "own"
                st.session_state["tutor_own_api_key"] = nk
                st.rerun()

    st.caption("进入后可在侧栏「API 连接设置」中随时更换。")
    st.stop()


def main() -> None:
    st.set_page_config(page_title="数学 Tutor", page_icon="📐", layout="centered")
    st.title("数学 Tutor")

    ensure_workbooks_loaded()

    orders = st.session_state.tutor_orders
    df = st.session_state.tutor_questions_df
    if not orders:
        st.error("questions.xlsx 中没有有效 order。")
        st.stop()

    ensure_api_connection_selected()
    init_session(orders)
    configure_llm()

    with st.sidebar:
        st.caption("当前进度")
        if st.session_state.tutor_all_done:
            st.write("状态：已完成全部题目")
        else:
            co = current_order(orders)
            meta = row_for_order(df, co)
            st.write({"当前题": co, "标题": str(meta.get("title", ""))})
        export_md = build_chat_export_markdown()
        st.download_button(
            "下载本轮聊天记录（Markdown）",
            data=export_md,
            file_name=f"math_tutor_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown",
            use_container_width=True,
        )
        if st.button("重新开始"):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()
        with st.expander("API 连接设置", expanded=False):
            mode = st.session_state.get("tutor_api_mode")
            st.caption(
                "当前：**平台托管**"
                if mode == "hosted"
                else "当前：**自备 API Key**"
                if mode == "own"
                else "未配置"
            )
            if st.button("更换连接方式…", key="tutor_api_reset_mode"):
                for k in (
                    "tutor_api_mode",
                    "tutor_own_api_key",
                    "_deepseek_client",
                    "_deepseek_config_sig",
                ):
                    st.session_state.pop(k, None)
                st.rerun()
        if st.button("测试 API（极简请求）"):
            ok, ping_msg = run_minimal_api_test()
            if ok:
                st.success(ping_msg)
            else:
                st.error(ping_msg)
        if st.button("回到上一题"):
            back = restore_previous_question()
            st.session_state.tutor_messages.append(
                {
                    "role": "assistant",
                    "content": back or "暂无可撤回的上一题状态。",
                }
            )
            st.rerun()
        st.caption("题目切换（主线）")
        for o in orders:
            if st.button(f"{o}", key=f"jump_{o}"):
                msg = jump_to_order(orders, o)
                st.session_state.tutor_messages.append({"role": "assistant", "content": msg})
                st.rerun()
        if st.button("查看学习小结"):
            st.session_state.tutor_started = True
            summ = build_summary_markdown(enriched_practice_history())
            st.session_state.tutor_messages.append(
                {"role": "assistant", "content": summ}
            )
            st.rerun()
        if (
            st.session_state.tutor_started
            and not st.session_state.tutor_all_done
            and st.button("强制讲解并跳过本题", help="先简要讲解本题要点，再进入下一题")
        ):
            co = current_order(orders)
            fmeta = row_for_order(df, co)
            with st.spinner("正在根据本题要点生成简要讲解…"):
                lecture, _lecture_trace = generate_force_lecture_reply(fmeta)
            finish_question_record(co, fmeta, FINAL_FORCE_LECTURE)
            title_bit = str(fmeta.get("title", co) or co)
            head = (
                f"### 「{co}」{FINAL_FORCE_LECTURE}\n\n"
                f"**{title_bit}**\n\n"
                f"{lecture.strip()}\n\n"
                f"---\n\n"
            )
            if st.session_state.tutor_all_done:
                summ = build_summary_markdown(enriched_practice_history())
                st.session_state.tutor_messages.append(
                    {
                        "role": "assistant",
                        "content": head + f"本题已记入「{FINAL_FORCE_LECTURE}」。\n\n{summ}",
                    }
                )
            else:
                no = current_order(orders)
                nm = row_for_order(df, no)
                st.session_state.tutor_messages.append(
                    {
                        "role": "assistant",
                        "content": (
                            head
                            + f"下面进入下一题 **{no}**：\n\n"
                            f"### {nm['title']}\n{nm['question_text']}"
                        ),
                    }
                )
            st.rerun()
        with st.expander("本轮 behavior / focus / scaffold 调试", expanded=False):
            st.caption(
                "每轮用户发言后追加一条记录。修改并保存 Excel 后，任意一次交互会自动检测文件变更并重载。"
            )
            dbg = list(st.session_state.get("tutor_turn_debug") or [])
            if not dbg:
                st.write("暂无记录。")
            else:
                tail_start = max(0, len(dbg) - 12)
                for j in range(len(dbg) - 1, tail_start - 1, -1):
                    e = dbg[j]
                    rank = len(dbg) - j
                    st.markdown(
                        f"**最近第 {rank} 条** · `{e.get('order', '')}` · **{e.get('behavior') or '—'}**"
                    )
                    st.write(
                        f"分支：`{e.get('branch', '')}` · 回复兜底：`{e.get('reply_used_fallback', '')}` · "
                        f"分类走了 LLM：`{e.get('classify_used_llm', '—')}` · "
                        f"分类 JSON 有效：`{e.get('classify_llm_ok', '—')}`"
                    )
                    if e.get("classify_brief_reason"):
                        st.caption(f"判定说明：{e['classify_brief_reason']}")
                    if e.get("behavior_source"):
                        st.caption(f"判定来源：{e['behavior_source']}")
                    if e.get("classify_llm_trace"):
                        st.caption(f"分类 API 尝试：{e['classify_llm_trace']!s}"[:800])
                    fc_dbg = e.get("focus") or ""
                    st.text_area(
                        "选用的 focus（引导关切）",
                        value=fc_dbg,
                        height=min(160, 80 + len(fc_dbg) // 6),
                        key=f"tutor_dbg_focus_{j}",
                        disabled=True,
                    )
                    sc = e.get("scaffold") or ""
                    st.text_area(
                        "选用的 scaffold 全文",
                        value=sc,
                        height=min(280, 120 + len(sc) // 4),
                        key=f"tutor_dbg_scaffold_{j}",
                        disabled=True,
                    )

    if st.session_state.tutor_all_done and not st.session_state.tutor_messages:
        st.success("本轮已完成全部题目。")
        st.markdown(build_summary_markdown(enriched_practice_history()))
        return

    if not st.session_state.tutor_started:
        st.session_state.tutor_started = True
        o = current_order(orders)
        ensure_question_message_start(o)
        meta = row_for_order(df, o)
        st.session_state.tutor_messages.append(
            {
                "role": "assistant",
                "content": (
                    "你好，我是你的一对一AI Math Tutor。接下来我们通过练习题学习SG Theory这个板块。"
                    "你可以先试着回答，说出你的思路——哪怕不完全确定也没关系。"
                    "如果实在没想法，或者想先确认某个细节、问一句「是这样吗」「为什么」，随时告诉我。"
                    "我不会直接告诉你答案，但会引导你自己找到答案。\n\n"
                    f"### {meta['title']}\n{meta['question_text']}"
                ),
            }
        )

    for msg in st.session_state.tutor_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("输入你的想法、解答或疑问…"):
        st.session_state.tutor_messages.append({"role": "user", "content": prompt})
        with st.spinner("思考中…"):
            reply = process_user_message(prompt)
        st.session_state.tutor_messages.append({"role": "assistant", "content": reply})
        st.rerun()


if __name__ == "__main__":
    main()
