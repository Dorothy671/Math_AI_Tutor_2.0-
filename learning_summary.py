"""
学习小结：按题统计、按 math_logic 分组的表格与叙述、学习习惯判定与建议。
"""

from __future__ import annotations

import re
from collections import defaultdict

from tutor_core import (
    BEHAVIOR_CORRECT,
    BEHAVIOR_CORRECT_ALIAS,
    BEHAVIOR_CORRECT_LEGACY,
)

# —— 与产品文案一致的终态（写入 practice_history.final_behavior）——
FINAL_DISPLAY_CORRECT = "解题：全对"
FINAL_FORCE_LECTURE = "强制讲解"

# —— 计入「有效回答错误」的行为（与脚手架解题类判定一致）——
INCORRECT_BEHAVIORS = frozenset(
    {
        "解题：答案&策略都错误",
        "解题：答案正确&策略错误",
        "解题：答案错误&策略正确",
    }
)

BEHAVIOR_HELP = "求助"
BEHAVIOR_GUESS = "猜测"
BEHAVIOR_ASK = "提问"
BEHAVIOR_BOTH_WRONG = "解题：答案&策略都错误"
BEHAVIOR_STRAT_WRONG = "解题：答案正确&策略错误"

HABIT_TEXT: dict[str, tuple[str, str]] = {
    "独立推理型": (
        "你喜欢自己先琢磨、不轻易求助，能加深理解。",
        "偶尔卡住时主动求助会更高效。",
    ),
    "探索试错型": (
        "你敢于尝试，能发现规则边界。",
        "试错前先从小规模例子列规律，避免消耗太多时间。",
    ),
    "寻求确认型": (
        "你比较谨慎，能避免盲目自信。",
        "先给自己一个确定答案再验证，而不是等外部肯定。",
    ),
    "依赖引导型": (
        "你能快速获得方向。",
        "先自己试着推一小步再求助，效果会更好。",
    ),
}


def enrich_practice_history(history: list[dict], questions_df) -> list[dict]:
    """
    历史记录里若缺 math_logic（旧会话或未写入），按题号从 questions 表补全，
    便于小结按知识板块拆分（与 load_questions 的 math_logic ffill 一致）。
    """
    if not history:
        return []
    if questions_df is None:
        return [dict(h) for h in history]
    out: list[dict] = []
    for rec in history:
        r = dict(rec)
        if str(r.get("math_logic") or "").strip():
            out.append(r)
            continue
        oid = str(r.get("order", "")).strip()
        if oid:
            try:
                sub = questions_df[
                    questions_df["order"].astype(str).str.strip() == oid
                ]
                if len(sub) > 0:
                    r["math_logic"] = str(sub.iloc[0].get("math_logic", "") or "")
            except Exception:
                pass
        out.append(r)
    return out


def normalize_final_behavior(stored: str) -> str:
    """统一终态展示字符串。"""
    s = str(stored or "").strip()
    if s in (
        BEHAVIOR_CORRECT,
        BEHAVIOR_CORRECT_ALIAS,
        BEHAVIOR_CORRECT_LEGACY,
        FINAL_DISPLAY_CORRECT,
        "答案正确且策略正确",
        "全对",
    ):
        return FINAL_DISPLAY_CORRECT
    if s == FINAL_FORCE_LECTURE:
        return FINAL_FORCE_LECTURE
    return s or "（未知）"


def is_incorrect_behavior(behavior: str) -> bool:
    return str(behavior or "").strip() in INCORRECT_BEHAVIORS


def mastery_level_for_record(rec: dict) -> str:
    """单题掌握等级：优秀 / 良好 / 待加强。"""
    seq = list(rec.get("behavior_sequence") or [])
    inc = int(rec.get("incorrect_count") or 0)
    final = normalize_final_behavior(str(rec.get("final_behavior") or ""))
    has_correct = any(
        str(x).strip()
        in (BEHAVIOR_CORRECT, BEHAVIOR_CORRECT_ALIAS, BEHAVIOR_CORRECT_LEGACY)
        for x in seq
    )
    if final == FINAL_FORCE_LECTURE or not has_correct or inc > 3:
        return "待加强"
    if inc <= 1:
        return "优秀"
    return "良好"


def topic_display_name(math_logic: str) -> str:
    """题型展示名（在相同 math_logic 分组合并行内共用）。"""
    t = str(math_logic or "").strip()
    if not t:
        return "综合题型类"
    if any(k in t for k in ("复合", "多阶段", "多步")):
        return "复合题型类"
    if any(k in t for k in ("反巴", "反向")) or ("巴什" in t and "反" in t):
        return "反巴什博弈类"
    if any(k in t for k in ("非连续", "子集")):
        return "非连续子集类"
    if "巴什" in t or "Bash" in t.lower():
        return "巴什博弈类"
    if len(t) <= 20:
        return f"{t}类"
    return f"{t[:18]}…类"


def _order_sort_key(rec: dict) -> tuple:
    """题号排序：A1<A2<…，其余按字符串。"""
    o = str(rec.get("order", "")).strip()
    m = re.match(r"^A\s*(\d+)$", o, re.I)
    if m:
        return (0, int(m.group(1)), o)
    m2 = re.match(r"^(\d+)$", o)
    if m2:
        return (0, int(m2.group(1)), o)
    return (1, o, o)


def _clip_title(title: str, n: int = 36) -> str:
    t = str(title or "").strip()
    if not t:
        return "（无标题）"
    return t if len(t) <= n else t[: n - 1] + "…"


def _order_remark(rec: dict) -> str:
    """单题备注片段（如 A1有求助，掌握扎实）。"""
    o = str(rec.get("order", ""))
    seq = list(rec.get("behavior_sequence") or [])
    m = mastery_level_for_record(rec)
    final = normalize_final_behavior(str(rec.get("final_behavior") or ""))
    if final == FINAL_FORCE_LECTURE:
        return f"{o}强制讲解结束"
    segs: list[str] = []
    if BEHAVIOR_HELP in seq:
        segs.append("有求助")
    if m == "优秀":
        segs.append("掌握扎实" if BEHAVIOR_HELP in seq else "直接掌握")
    elif m == "良好":
        segs.append("经引导掌握")
    else:
        segs.append("掌握偏慢或未独立完成")
    return f"{o}{'，'.join(segs)}"


def _table_remark(rec: dict) -> str:
    """概览表备注：标题 + 互动与掌握摘要。"""
    return f"《{_clip_title(str(rec.get('title', '')), 40)}》{_order_remark(rec)}"


def _group_migration_note(records: list[dict]) -> str:
    """同 math_logic 组内知识迁移一句话。"""
    orders = [str(r.get("order", "")) for r in records]
    levels = [mastery_level_for_record(r) for r in records]
    any_force = any(
        normalize_final_behavior(str(r.get("final_behavior") or "")) == FINAL_FORCE_LECTURE
        for r in records
    )
    any_help = any(BEHAVIOR_HELP in (r.get("behavior_sequence") or []) for r in records)
    all_excellent = levels and all(x == "优秀" for x in levels)
    all_weak = levels and all(x == "待加强" for x in levels)
    if len(records) >= 2 and all_excellent and not any_help:
        return "同类题知识迁移良好，掌握稳定。"
    if len(records) >= 2 and all_excellent and any_help:
        return "同类题最终掌握扎实，前期略有求助，迁移尚可。"
    if len(records) >= 2 and levels[0] == "待加强" and levels[-1] in ("优秀", "良好"):
        return "后一题明显好于前一题，迁移在改善。"
    if any_force:
        return "部分题目未独立完成，建议复盘后再练同类题。"
    if all_weak:
        return "该类型整体偏薄弱，建议放慢节奏、补全关键概念。"
    if any_help:
        return "有主动求助，理解在加深；可尝试下一题减少提示依赖。"
    return "完成度尚可，建议对照策略要点自查一遍。"


def _flatten_behaviors(history: list[dict]) -> list[str]:
    out: list[str] = []
    for rec in history:
        out.extend(list(rec.get("behavior_sequence") or []))
    return out


def classify_learning_habit(history: list[dict]) -> str:
    """
    返回四种学习习惯之一。严格条件优先；否则在「寻求确认 / 探索试错 / 独立推理」中
    按行为频次取主导（「依赖引导型」仅当求助≥2，避免与规则矛盾）。
    """
    flat = _flatten_behaviors(history)
    if not flat:
        return "独立推理型"

    n_help = sum(1 for b in flat if b == BEHAVIOR_HELP)
    n_guess = sum(1 for b in flat if b == BEHAVIOR_GUESS)
    n_ask = sum(1 for b in flat if b == BEHAVIOR_ASK)
    n_both_wrong = sum(1 for b in flat if b == BEHAVIOR_BOTH_WRONG)
    n_strat_wrong = sum(1 for b in flat if b == BEHAVIOR_STRAT_WRONG)
    trial = n_both_wrong + n_strat_wrong
    confirm = n_guess + n_ask

    if n_help >= 2:
        return "依赖引导型"
    if confirm >= 3:
        return "寻求确认型"
    if trial >= 2:
        return "探索试错型"
    if n_help == 0 and n_guess <= 1:
        return "独立推理型"

    # 混合互动：在三类中择主导（不出现「依赖引导」除非已满足求助≥2）
    scores = {
        "寻求确认型": confirm * 4 + n_guess * 2 + n_ask,
        "探索试错型": trial * 5 + n_both_wrong * 2 + n_strat_wrong * 2,
        "独立推理型": (5 if n_help == 0 else 0)
        + max(0, 6 - 2 * confirm)
        + max(0, 4 - trial * 3),
    }
    # 单次求助仍偏「要脚手架」时略倾向寻求确认而非独立
    if n_help == 1:
        scores["寻求确认型"] += 2
        scores["独立推理型"] += 1
    best = max(scores, key=scores.get)
    if scores[best] <= 0:
        return "寻求确认型" if confirm > 0 else "探索试错型" if trial > 0 else "独立推理型"
    return best


def habit_paragraph(history: list[dict]) -> str:
    """学习习惯：固定四类之一 + 评价与建议，并附简要数据依据。"""
    flat = _flatten_behaviors(history)
    name = classify_learning_habit(history)
    ev, ad = HABIT_TEXT[name]
    n_help = sum(1 for b in flat if b == BEHAVIOR_HELP)
    n_guess = sum(1 for b in flat if b == BEHAVIOR_GUESS)
    n_ask = sum(1 for b in flat if b == BEHAVIOR_ASK)
    trial = sum(
        1
        for b in flat
        if b in (BEHAVIOR_BOTH_WRONG, BEHAVIOR_STRAT_WRONG)
    )
    data = (
        f"（依据：本轮行为共 {len(flat)} 条，其中求助 {n_help} 次、猜测 {n_guess} 次、"
        f"提问 {n_ask} 次，解题类明显失误 {trial} 次。）"
    )
    return f"**{name}** {ev}{ad}{data}"


def _narrative_line_for_record(rec: dict) -> str:
    """单题一行：等级、错误、终态、互动要点。"""
    o = str(rec.get("order", ""))
    lv = mastery_level_for_record(rec)
    inc = int(rec.get("incorrect_count") or 0)
    final = normalize_final_behavior(str(rec.get("final_behavior") or ""))
    seq = list(rec.get("behavior_sequence") or [])
    tags: list[str] = []
    if BEHAVIOR_HELP in seq:
        tags.append("有求助")
    if seq.count(BEHAVIOR_GUESS) >= 2:
        tags.append("多次猜测")
    if seq.count(BEHAVIOR_ASK) >= 2:
        tags.append("多次提问澄清")
    if final == FINAL_FORCE_LECTURE:
        tags.append("强制讲解结束")
    tag_s = f"；{'、'.join(tags)}" if tags else ""
    return (
        f"- **{o}**（{lv} · 有效错误 {inc} · 终态 {final}）"
        f"《{_clip_title(str(rec.get('title', '')), 32)}》"
        f"{tag_s}"
    )


def build_narrative_by_topic(history: list[dict]) -> str:
    """第二部分：按数学逻辑分块，块内逐题一行，再写板块小结。"""
    if not history:
        return ""
    by_ml: dict[str, list[dict]] = defaultdict(list)
    for rec in history:
        ml = str(rec.get("math_logic") or rec.get("_math_logic") or "").strip()
        key = ml or "（未标注数学要点）"
        by_ml[key].append(rec)

    paragraphs: list[str] = []
    for ml, group in sorted(by_ml.items(), key=lambda x: x[0]):
        group = sorted(group, key=_order_sort_key)
        label = topic_display_name(ml)
        levels = [mastery_level_for_record(r) for r in group]
        incs = [int(r.get("incorrect_count") or 0) for r in group]
        any_help = any(BEHAVIOR_HELP in (r.get("behavior_sequence") or []) for r in group)
        any_force = any(
            normalize_final_behavior(str(r.get("final_behavior") or "")) == FINAL_FORCE_LECTURE
            for r in group
        )
        n_ex = sum(1 for x in levels if x == "优秀")
        n_ok = sum(1 for x in levels if x == "良好")
        n_weak = sum(1 for x in levels if x == "待加强")
        max_inc = max(incs) if incs else 0

        bits: list[str] = []
        bits.append(f"**{label}**\n\n")
        if ml and ml != "（未标注数学要点）":
            ml_show = ml if len(ml) <= 200 else ml[:199] + "…"
            bits.append(f"> **知识要点摘录**：{ml_show}\n\n")
        for r in group:
            bits.append(_narrative_line_for_record(r) + "\n")
        bits.append("\n")

        if n_ex == len(group):
            bits.append("**本板块**：整体掌握扎实，错误控制较好。")
        elif n_weak == len(group):
            bits.append("**本板块**：目前最薄弱，纠偏轮次偏多，建议专项复盘。")
        elif n_ex + n_ok >= n_weak:
            bits.append("**本板块**：多数题到位，个别题仍需补洞。")
        else:
            bits.append("**本板块**：掌握程度参差，建议按题对照策略要点自查。")

        if "非连续" in label or "子集" in label:
            bits.append(
                " 这类题常需枚举或递推必败态，不能单靠简单取模，迁移往往比周期类慢。"
            )
        elif "反巴" in label:
            bits.append(" 反向规则易与常规巴什混淆，注意「可操作集合」与目标态的对应。")
        elif "巴什" in label and "反" not in label:
            bits.append(" 周期与凑整段思路若已建立，后续同类题可加快独立完成。")

        if any_help and not any_force:
            bits.append(" 过程中有适度求助，利于对齐关键概念。")
        if any_force:
            bits.append(" 有题目以讲解结束，建议课后独立重做以固化。")
        if max_inc > 2:
            bits.append(f" 单题有效错误峰值达 {max_inc}，中间曾反复调整，值得复盘。")

        mig = _group_migration_note(group)
        bits.append(f" {mig}")
        paragraphs.append("".join(bits))

    closing: list[str] = []
    if len(by_ml) >= 2:
        has_period = any(
            "巴什" in topic_display_name(k) and "反" not in topic_display_name(k)
            for k in by_ml
        )
        has_subset = any("非连续" in topic_display_name(k) for k in by_ml)
        if has_period and has_subset:
            closing.append(
                "**整体**：你对周期、凑整类（如常规巴什）相对更熟；"
                "对需逆向枚举、非连续结构的题目迁移仍偏弱，建议加练并对比建模差异。"
            )
        else:
            closing.append(
                "**整体**：建议以掌握最好的板块为模板，对照薄弱板块检查"
                "「状态定义—可操作—目标态」是否写清楚。"
            )
    return "\n\n".join(paragraphs + closing)


def build_summary_markdown(history: list[dict]) -> str:
    """完整学习小结 Markdown（表格 + 叙述 + 习惯）。"""
    if not history:
        return "## 学习小结\n\n暂无已完成题目记录。完成题目后会在此汇总掌握情况与习惯建议。"

    lines: list[str] = ["## 学习小结", ""]

    lines.append("### 一、做题概览")
    lines.append("")
    lines.append(
        "| 题号 | 题型 | 掌握等级 | 错误次数 | 备注 |\n"
        "| --- | --- | --- | --- | --- |"
    )

    flat_rows = sorted(history, key=_order_sort_key)
    for rec in flat_rows:
        o = str(rec.get("order", ""))
        ml = str(rec.get("math_logic") or "").strip()
        topic = topic_display_name(ml)
        lv = mastery_level_for_record(rec)
        inc = int(rec.get("incorrect_count") or 0)
        remark = _table_remark(rec)
        lines.append(f"| {o} | {topic} | {lv} | {inc} | {remark} |")

    lines.append("")
    lines.append("### 二、按知识板块回顾")
    lines.append("")
    lines.append(build_narrative_by_topic(history))
    lines.append("")
    lines.append("### 三、学习习惯与建议")
    lines.append("")
    lines.append(habit_paragraph(history))

    return "\n".join(lines)
