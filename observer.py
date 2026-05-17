"""
Observer：调用 DeepSeek，仅输出 scaffolding.xlsx 中 behavior 列的合法取值；
另提供「枚举必败第二轮」规律是否与标答 strategy 一致的判定。
"""

from __future__ import annotations

import pandas as pd

from llm_api import llm_json, tutor_model_chain

OBSERVER_SYSTEM = """
你是「行为观察」助手。下面是一张表，每行包含 behavior 名称与 trigger_condition（何时应判定为该行为）。

【任务】结合【最近对话】与【学生本轮发言】及【本题材料】，选出**唯一**最匹配的 behavior 名称。

【硬性规则】
1. trigger_condition 若要求对照标答或策略，必须严格依据本题给出的【标答 answer】【策略 strategy】做语义判断。
2. behavior 字符串必须与表中的名称**完全一致**（含全角标点），一字不差。
3. 仅陈述先后手选择、无概念追问时，不要判为「提问」类行为（以表中 trigger 为准）。
4. 若多条部分符合，选更具体的一条；仍无法归类则选表中「其他」对应的 behavior（须与表中名称完全一致）。

【策略正确判定】（凡 behavior 或 trigger 涉及「策略正确」「策略错误」「策略缺失」等，必须遵守）
1. 学生须**主动**陈述可执行的拿法或数学规律，**不得**仅凭「回答导师刚提的一个引导性小问题」就认定其已理解**整道题**的策略。
2. 须对照【最近约两轮对话】：若 assistant 刚问了**非常具体**的问题（如单个总数、一步拿法、「这一轮一共拿了几颗」「对方拿 1 你拿几」等），而学生**本轮只**给出该问题的直接答案（如「3 颗」「拿 2」「1+2=3」这类短答），**没有**在同轮或后续补充完整配对/通式/不变量 → **不算**学生已主动说出整题策略，不得因此选用含「策略正确」或与事实不符的「策略已掌握」类判定。
3. 「策略正确」的积极标准至少满足其一：能说明 **「对方拿 X 时我拿 Y」** 的完整配对或控和规则；或能概括与题干一致的**数学规律**（如「每轮两人一共拿 3」「凑 3」「模几余几」等），且与 **strategy** 语义一致。

【策略正确—正反例】
- 导师问：「这一轮一共拿了几颗？」学生答：「3 颗」→ **否**（仅回答引导问句，不算已掌握策略）。
- 导师问：「对方拿 1 你拿几？」学生答：「拿 2」→ **否**（同上；除非学生同句或紧接主动补全所有分支与规律）。
- 学生主动说：「对方拿 1 我拿 2，对方拿 2 我拿 1，这样每轮凑 3」→ **是**（主动 + 完整规则，可结合 strategy 判定策略类 behavior）。

5. **tone_uncertain**（必填布尔，**禁止省略该键**）：只根据【学生本轮发言】**原文**判断是否「语气不笃定、在求确认、或猜测式作答」。
   - **必须为 true** 的典型信号（出现即重点检查，常与解题结论同句）：句末或短语中的 **「吧、呢、啊、吗、么」**（如「后手吧」「应该是吧」）、**「？」「??」**、以及 **「可能、也许、大概、好像、估计、试试、猜、觉得、不确定、说不清、对不对、对吗、是不是、靠谱吗」** 等。
   - **应为 false**：语气果断、有清晰推理或计算步骤，无上述模糊/求确认标记，且明显在陈述自己的结论而非单纯提问。
   - 若本轮**没有**在给出解题结论（例如只在问规则、闲聊），tone_uncertain 一般为 false。
6. **一致性铁律**：`behavior`、`reason`、`position`、`strategy`、`pattern` 五者必须自洽——**以 JSON 三键为准**，`reason` 不得与三键矛盾（禁止 reason 写「三维度均正确」却将 `pattern` 标为「缺失」等）。若理由中认定学生**策略与 strategy 不符**，则 `behavior` **不得**为「解题：全对」，且 `strategy` 不得为「正确」；若认定**先后手与 answer 不符**，不得选「解题：全对」，且 `position` 不得为「正确」。**禁止**在 reason 里写「策略错误」却输出「解题：全对」或 `strategy`:「正确」。
7. **「解题：全对」与三键（硬性）**：仅当 `position`、`strategy`、`pattern` 三个 JSON 字段的值**字面均为**「正确」时，才可选表中 **「解题：全对」**（或与表中收题类名称完全一致者）。**任一**键为「缺失」或「错误」→ **禁止**选「解题：全对」；若 position、strategy 已为「正确」而 pattern 非「正确」→ 须选 **「解题：枚举必败点待抽象规律」**（若表中有该名称）。判定须对照 **answer、strategy、math_logic** 的语义，不得仅凭语气肯定就升格 pattern。

【三维度判定】（每个字段必须是 **「正确」「错误」「缺失」** 之一，禁止其它写法）
- **position**（先后手/结论立场）：对照 **answer** 与题干，学生是否给出与标答一致的先后手或胜负侧结论；若本轮完全未涉及先后手结论 →「缺失」；若说了但与标答不符 →「错误」。
- **strategy**（可操作策略）：对照 **strategy**，是否掌握「对方拿几我拿几」「每轮凑几」等完整可执行规则；须遵守上文【策略正确判定】；若只答了导师引导性小问、未主动给出整题策略 →「缺失」或「错误」，**不得**因「1+2=3」式短答判「正确」。
- **pattern**（数学规律层）：**每一道题**都必须对照 **math_logic** 与 **strategy** 做语义判断（**含巴什/Nim**）。**pattern 正确** = 学生用**自己的话**说出了 math_logic 所要求的**规律层**要点（如凑 (m+1)、模 (n+1)、不变量、必败递推/通式、周期等），且与材料一致。**不算 pattern**：仅复述 **strategy 列里已经写明的**操作句（如「每轮合拿 3」「对方 1 我拿 2」）而**未**额外说出 math_logic 中的抽象规律或「为何能推出先后手」的推理链 → 仍标 **「缺失」**。**禁止**把「操作复述」误标为「规律已掌握」；**禁止**在 `pattern` 为「缺失」或「错误」时选「解题：全对」。

输出 JSON：{{"behavior":"…","position":"缺失","strategy":"缺失","pattern":"缺失","tone_uncertain":false,"reason":"…"}}（若收题类 behavior 在表中为「解题：全对」，则 behavior 字段须写该**完整字符串**。）
"""

RESTATE_SYSTEM = """
你是「复述验收」助手。学生需要用**自己的话**完整说明本题解法要点。

【任务】阅读【学生本轮发言】与【最近对话节选】，对照【标答 answer】【策略 strategy】【题干】，判断学生复述是否**正确且足够完整**。

【必须覆盖的要点】（若本题某条明显不适用可忽略，但其余须到位）
1. 先后手结论（谁应先手/后手或让对方先）；
2. 每轮怎么拿（可操作层面的配对/凑数/对方拿几我拿几等，若题干有）；
3. 核心数学规律（不变量、必败点规律/周期/模运算等，与 strategy 一致）。

【规则】
1. 明显遗漏任一关键要点、或与标答矛盾、或只是复述零散数字而无清晰策略链，判为不完整/错误。
2. 表述可与标答用词不同，但数学结论须一致。
3. 输出 JSON：{{"restate_ok": true 或 false, "brief_reason":"内用简短理由"}}
"""

AUTONOMY_SYSTEM = """
你是「收题前自主推理判定」助手。系统已确认当前题内 position、strategy、pattern 三个维度累计均为正确；你的任务不是重新判答案对错，而是判断这些正确维度是学生**自主推理串起来的完整解法**，还是主要靠回答导师一步步引导小问凑齐的。

【判为 autonomous_ok=true 的标准】
- 学生在当前题对话中，主动用自己的话把解法链条说清楚：先后手/胜负侧、可执行拿法、数学规律或为什么这样能赢；
- 可以分布在多轮里，但学生需要有主动串联、总结、解释的表达，而不是只回答导师刚问的单点问题；
- 表述不必和标答一致逐字相同，但语义需完整。

【判为 autonomous_ok=false 的标准】
- 学生主要是在回答导师非常具体的小问，例如「拿几」「一共几颗」「能不能赢」「余数是多少」；
- 学生虽然每个局部都答对了，但没有主动复述或串联完整策略与数学规律；
- 导师在最近回复中替学生补全了关键策略、通式、除法/取模结论，而学生只是确认或短答。

输出 JSON：{{"autonomous_ok": true 或 false, "brief_reason":"简短说明为什么可直接收题或为什么应先复述"}}
"""


def _valid_behaviors(sdf: pd.DataFrame) -> set[str]:
    return {str(x).strip() for x in sdf["behavior"].dropna().unique() if str(x).strip()}


def _scaffolding_trigger_table(sdf: pd.DataFrame) -> str:
    lines: list[str] = []
    for _, row in sdf.iterrows():
        b = str(row.get("behavior", "") or "").strip()
        tc = str(row.get("trigger_condition", "") or "").strip()
        lines.append(f"- **{b}**\n  触发条件：{tc}\n")
    return "\n".join(lines) if lines else "（空表）"


def _format_recent_turns(messages: list | None, *, max_messages: int = 4) -> str:
    """最近约 2 轮：取末尾若干条 role 消息（不含本轮 user 时由调用方先切片）。"""
    msgs = [m for m in (messages or []) if isinstance(m, dict)]
    tail = msgs[-max_messages:] if msgs else []
    if not tail:
        return "（无）"
    out: list[str] = []
    for m in tail:
        role = str(m.get("role", "") or "").strip()
        c = str(m.get("content", "") or "").strip()
        if len(c) > 1200:
            c = c[:1199] + "…"
        out.append(f"{role}: {c}")
    return "\n".join(out)


ACCUM_DIM_VALUES = frozenset({"正确", "错误", "缺失"})


def normalize_observer_dim(v: object) -> str | None:
    """将模型输出规范为「正确」「错误」「缺失」；无法识别则返回 None（合并时视为不更新累积）。"""
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None
    if s in ACCUM_DIM_VALUES:
        return s
    if s in ("对", "是", "对的", "正确。"):
        return "正确"
    if s in ("错", "不对", "错误。", "错的"):
        return "错误"
    if s in ("无", "没有", "未涉及", "未给出", "不知道", "不清楚"):
        return "缺失"
    return None


def _parse_bool(v: object) -> bool:
    if v is True:
        return True
    if v is False:
        return False
    if isinstance(v, str):
        return v.strip().lower() in ("true", "yes", "1", "是", "对", "正确")
    return False


def observer_judge(
    user_text: str,
    meta: pd.Series,
    sdf: pd.DataFrame,
    *,
    recent_messages: list | None = None,
) -> tuple[str, dict]:
    """
    调用 DeepSeek，返回表中合法的 behavior。
    recent_messages：完整会话列表；若最后一项为本轮 user，会先去掉以免与 user_text 重复。
    """
    if not str(user_text or "").strip():
        return "其他", {
            "brief_reason": "空输入",
            "observer_raw": {},
            "position": None,
            "strategy": None,
            "pattern": None,
            "tone_uncertain": False,
            "tone_uncertain_omitted": False,
            "classify_used_llm": False,
            "classify_llm_ok": None,
            "classify_source": "observer",
            "classify_llm_trace": None,
        }
    valid = _valid_behaviors(sdf)
    if not valid:
        return "其他", {
            "brief_reason": "scaffolding 表无有效 behavior 列",
            "observer_raw": {},
            "position": None,
            "strategy": None,
            "pattern": None,
            "tone_uncertain": False,
            "tone_uncertain_omitted": False,
            "classify_used_llm": False,
            "classify_llm_ok": None,
            "classify_source": "observer",
            "classify_llm_trace": None,
        }
    rubric = _scaffolding_trigger_table(sdf)
    msgs = list(recent_messages or [])
    if msgs and str(msgs[-1].get("role", "")).strip() == "user":
        last_c = str(msgs[-1].get("content", "") or "").strip()
        if last_c == str(user_text or "").strip():
            msgs = msgs[:-1]
    recent_block = _format_recent_turns(msgs, max_messages=4)
    payload = (
        f"{rubric}\n\n"
        "【本题题干 question_text】\n"
        f"{meta.get('question_text', '')}\n\n"
        "【标答 answer】\n"
        f"{meta.get('answer', '')}\n\n"
        "【策略 strategy】\n"
        f"{meta.get('strategy', '')}\n\n"
        "【最近约两轮对话】\n"
        f"{recent_block}\n\n"
        "【学生本轮发言】\n"
        f"{str(user_text or '').strip()}\n"
    )
    data, raw_meta = llm_json(
        OBSERVER_SYSTEM.strip(),
        payload,
        temperature=0.05,
        retries=4,
        timeout_sec=180,
        models=tutor_model_chain(),
    )
    b_raw = str(data.get("behavior", "") or "").strip()
    if b_raw in valid:
        b = b_raw
    else:
        b = ""
    fallback = (
        "其他"
        if "其他" in valid
        else (sorted(valid)[0] if valid else "其他")
    )
    if not b:
        b = fallback
        extra = f"（模型输出的 behavior 不在表中，退回「{fallback}」）"
    else:
        extra = ""
    tu_raw = data.get("tone_uncertain")
    if tu_raw is None or (isinstance(tu_raw, str) and not str(tu_raw).strip()):
        # 模型漏字段：不推断 true/false，由产品流程按「不确定」发追问（与明确 false 区分见 tone_uncertain_omitted）
        tone_uncertain = True
        tu_omitted = True
        tu_suffix = "（tone_uncertain 未输出，按不确定处理）"
    else:
        tone_uncertain = _parse_bool(tu_raw)
        tu_omitted = False
        tu_suffix = ""
    br = str(data.get("reason") or data.get("brief_reason", "") or "").strip()
    if extra:
        br = (br + extra).strip() if br else extra.strip()
    if tu_suffix:
        br = (br + tu_suffix).strip() if br else tu_suffix.strip()
    pos = normalize_observer_dim(data.get("position"))
    strat = normalize_observer_dim(data.get("strategy"))
    pat = normalize_observer_dim(data.get("pattern"))
    detail: dict = {
        "brief_reason": br,
        "observer_raw": data,
        "position": pos,
        "strategy": strat,
        "pattern": pat,
        "tone_uncertain": tone_uncertain,
        "tone_uncertain_omitted": tu_omitted,
        "classify_used_llm": True,
        "classify_llm_ok": bool(data),
        "classify_source": "observer",
        "classify_llm_trace": raw_meta.get("attempts"),
    }
    return b, detail


def observer_solution_restate_ok(
    user_text: str,
    meta: pd.Series,
    recent_messages: list | None = None,
) -> tuple[bool, dict]:
    """结题前复述验收：先后手 + 每轮拿法 + 数学规律是否自述正确完整。"""
    msgs = list(recent_messages or [])
    if msgs and str(msgs[-1].get("role", "")).strip() == "user":
        last_c = str(msgs[-1].get("content", "") or "").strip()
        if last_c == str(user_text or "").strip():
            msgs = msgs[:-1]
    recent_block = _format_recent_turns(msgs, max_messages=8)
    payload = (
        f"【本题题干 question_text】\n{meta.get('question_text', '')}\n\n"
        f"【标答 answer】\n{meta.get('answer', '')}\n\n"
        f"【策略 strategy】\n{meta.get('strategy', '')}\n\n"
        f"【数学要点 math_logic（辅助）】\n{str(meta.get('math_logic', '') or '')[:4000]}\n\n"
        f"【最近对话节选】\n{recent_block}\n\n"
        f"【学生本轮复述】\n{str(user_text or '').strip()}\n"
    )
    data, raw_meta = llm_json(
        RESTATE_SYSTEM.strip(),
        payload,
        temperature=0.05,
        retries=4,
        timeout_sec=180,
        models=tutor_model_chain(),
    )
    detail = {
        "restate_observer_raw": data,
        "restate_trace": raw_meta.get("attempts"),
    }
    if not data:
        return False, {**detail, "brief_reason": "Observer 无有效 JSON，判复述未通过"}
    ok = _parse_bool(data.get("restate_ok", data.get("ok", data.get("correct"))))
    detail["brief_reason"] = str(data.get("brief_reason", "") or "")
    return ok, detail


def observer_solution_autonomous_ok(
    user_text: str,
    meta: pd.Series,
    recent_messages: list | None = None,
) -> tuple[bool, dict]:
    """三维度累计全对后：判断是否可直接收题，或需要先完整复述。"""
    msgs = [m for m in (recent_messages or []) if isinstance(m, dict)]
    recent_block = _format_recent_turns(msgs, max_messages=14)
    payload = (
        f"【本题题干 question_text】\n{meta.get('question_text', '')}\n\n"
        f"【标答 answer】\n{meta.get('answer', '')}\n\n"
        f"【策略 strategy】\n{meta.get('strategy', '')}\n\n"
        f"【数学要点 math_logic（辅助）】\n{str(meta.get('math_logic', '') or '')[:4000]}\n\n"
        f"【当前题对话节选】\n{recent_block}\n\n"
        f"【学生本轮】\n{str(user_text or '').strip()}\n"
    )
    data, raw_meta = llm_json(
        AUTONOMY_SYSTEM.strip(),
        payload,
        temperature=0.05,
        retries=4,
        timeout_sec=180,
        models=tutor_model_chain(),
    )
    detail = {
        "autonomy_observer_raw": data,
        "autonomy_trace": raw_meta.get("attempts"),
    }
    if not data:
        return False, {
            **detail,
            "brief_reason": "自主推理判定无有效 JSON，保守要求复述",
        }
    ok = _parse_bool(data.get("autonomous_ok", data.get("ok", data.get("direct_ok"))))
    detail["brief_reason"] = str(data.get("brief_reason", "") or "")
    return ok, detail
