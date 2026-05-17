"""
先后手等仍用轻量规则做答案侧推断（标答解析、拿法一致性等）。
学生行为标签由 Observer（DeepSeek）对照 scaffolding 表判定；回复默认 deepseek-chat。
"""

from __future__ import annotations

import re
import unicodedata

import pandas as pd

from llm_api import _coerce_message_any, llm_json, llm_text_reply, tutor_model_chain
from observer import observer_judge

# 须与 scaffolding.xlsx 中「收题/全对」行为名称**逐字一致**（否则维度收敛等逻辑不会触发）
BEHAVIOR_CORRECT = "解题：全对"
# 若表中曾用短名「全对」，Observer 输出短名时归一成上式
BEHAVIOR_CORRECT_ALIAS = "全对"
# 旧版 scaffolding 行为名（历史会话兼容）
BEHAVIOR_CORRECT_LEGACY = "解题：答案正确&策略正确"
# 仅枚举/倒推必败点、或只谈避开必败态，或尚未写出通式；须追问规律后才算策略完全掌握
BEHAVIOR_BEISHU_ENUM_PENDING = "解题：枚举必败点待抽象规律"

# —— 复述 / 语气确认（固定追问，避免依赖生成模型漏字）——
TONE_CONFIRMATORY_REPLY = "你确定吗？能说说为什么这么判断吗？"
RESTATE_SOLUTION_INSTRUCTION = (
    "请你用自己的话，完整复述一遍这道题的解法，包括：先后手、每轮怎么拿、数学规律是什么。"
)

TUTOR_REPLY_SYSTEM = """
你是苏格拉底式的数学游戏导师。

【铁律】
1. 所有引导必须基于本题 question_text、math_logic、strategy；禁止编造题干未出现的规则、数字或胜负条件。
2. **最高优先级**（**例外见下方「本轮模式标记」**）：用自然对话、追问和简短提示，引导学生**自己**逐步推出先后手结论与具体拿法；不要一次性背诵完整标答。若 user 以「[模式:全对收题]」或「[模式:枚举二轮教师代讲结题]」开头，则**不适用**本条，须**完全按该模式**执行（陈述式小结、禁止追问）。
3. **每轮先验算再回应**：在肯定学生前，必须用 question_text、math_logic、strategy、answer 和最近对话中你刚提出的具体问题，校验学生本轮说法是否数学正确。尤其当学生只回答「能/不能/是/不是/先手/后手/拿几颗」这类短答时，先在内部推演该小局面，确认无误后才可说「对」「你说得对」或顺着总结；若学生答错或与题目材料矛盾，要温和指出矛盾，并带他重新算一个关键状态。若你不确定，宁可说「我们先验算一下」并展开核对，禁止未经校验直接肯定。
4. **禁止替学生补齐缺失维度**：你会收到「当前题内累计掌握状态」，包含 position（先后手/结论）、strategy（可操作策略）、pattern（数学规律层）。若本轮不是「全对收题」或「枚举二轮教师代讲结题」，且任一维度仍为「缺失」或「错误」，只能围绕缺失/错误维度做局部肯定、追问或纠错；禁止直接替学生说出完整策略、完整数学规律、除法/取模结论、必败态通式、最终收束语，也禁止说“完整策略就是…”“现在你已经都答对了”等结题式话语。
5. **辅助参考**：你会收到「本轮判定的 behavior」、**「引导焦点 focus」**（若有）与「脚手架参考 scaffold」。focus 标明本题该状态下导师应优先关切的方向；scaffold 提供话术与策略意图。三者只供你**内化**学生状态与引导重心——用你自己的话自然表达。**禁止**向学生朗读、罗列或照搬 focus/scaffold 原文，**禁止**输出「引导策略：」「降维：」等模板式条目。
6. 语气亲切、像真人聊天；一至两段为宜。不要输出 JSON、不要用 Markdown 大标题堆砌。

【本轮模式标记】
- 若 user 消息中以「[模式:全对收题]」开头：学生已同时答对答案与策略。请真诚肯定；用**简短、口语化**的话概括 math_logic 的核心思想；**不要**提及下一题或新题干。
- 若 user 消息中以「[模式:枚举二轮教师代讲结题]」开头：学生在「枚举必败待抽象」的**第二轮**仍未说到位，须由你**代为收束讲解**（与全对收题后的小结**同一风格**：陈述、直接、不讲谜语）。要求：① 先用 **1～2 句口语**承接【最近对话】里学生**上一轮及本轮**的具体说法（可点名他提到的必败数等），自然过渡；② 随后**通篇用陈述句**依据 math_logic、strategy、answer 讲清**完整**规律与先后手/操作要点；③ **严禁**句末问号、严禁「试试」「看看」「你发现」「对吗」「能否」「请接着」「补全序列」等引导语；④ 不要布置新题、不要提下一题题干。一至二段，约 450 字内。
- 否则为常规引导模式：不要承诺「下一题」或切换题号。
"""

FACTUAL_QA_SYSTEM = """
学生问的是知识性问题（为什么/什么是/怎么理解）。请根据题目、answer、strategy、math_logic 直接简短回答。
然后自然地把话题拉回本题思考。输出 JSON：{{"message":"正文"}}
"""


def _normalize_student_text(text: str) -> str:
    t = unicodedata.normalize("NFKC", str(text or "")).strip()
    return re.sub(r"[\u200b\ufeff\u00a0]", "", t)


_ULTRA_SHORT_SECOND = frozenset(
    {
        "后拿吧",
        "后拿呢",
        "后拿啊",
        "后拿呀",
        "后拿哦",
        "后拿",
        "后手吧",
        "后手呢",
        "后手啊",
        "后手呀",
        "后手哦",
        "后手",
        "后上吧",
        "后上",
        "让对方先",
        "我先不拿",
        "後拿吧",
        "後拿",
        "後手吧",
        "後手",
    }
)
_ULTRA_SHORT_FIRST = frozenset(
    {
        "先拿吧",
        "先拿呢",
        "先拿啊",
        "先拿呀",
        "先拿哦",
        "先拿",
        "先手吧",
        "先手呢",
        "先手啊",
        "先手呀",
        "先手哦",
        "先手",
        "先取吧",
        "先取",
        "我先拿",
        "我要先",
    }
)


def _ultra_short_position_bucket(text: str) -> str | None:
    t = _normalize_student_text(text)
    c = re.sub(r"\s+", "", t).strip("。！？!?…,，、")
    if c in _ULTRA_SHORT_SECOND:
        return "second"
    if c in _ULTRA_SHORT_FIRST:
        return "first"
    return None


def _user_position_hint(text: str) -> str | None:
    t = _normalize_student_text(text)
    if re.search(
        r"(后拿|後拿|后手|後手|后上|後上|让对方先|讓對方先|讓对方先|我先不拿|我后|我後|后取|後取)",
        t,
    ):
        return "second"
    if re.search(r"(先手|先拿|我先拿|我要先|先取)", t):
        return "first"
    return None


def _meta_answer_bundle(meta: pd.Series) -> str:
    return (
        f"{meta.get('answer', '')}\n{meta.get('strategy', '')}\n"
        f"{meta.get('math_logic', '')}\n{meta.get('question_text', '')}"
    )


def _answer_position_from_meta(meta: pd.Series) -> str | None:
    """
    推断标答中的先后手结论。
    不能用整段 math_logic：里面常有「若…则先手…若…则后手」通式，会误判本题结论。
    只综合 answer / strategy / title；并优先读「选择/选…先手|后手」类短句。
    """
    parts: list[str] = []
    for k in ("answer", "strategy", "title"):
        s = str(meta.get(k, "") or "").strip()
        if s:
            parts.append(s)
    if not parts:
        return None
    combined = "\n".join(parts)
    for line in combined.splitlines():
        ln = line.strip()
        if not ln:
            continue
        if re.search(
            r"(选择|选|要|应该|应当|需).{0,14}?(后手|后拿|后取|让对方先|讓對方先)",
            ln,
        ):
            return "second"
        if re.search(
            r"(选择|选|要|应该|应当|需).{0,14}?(先手|先拿|先取)",
            ln,
        ) and not re.search(r"(后手|后拿|后取|让对方先)", ln):
            return "first"
    if re.search(r"先手(?:首轮|第一[轮局]?|先).{0,14}(?:取|拿|走)", combined):
        if not re.search(
            r"(选择|选|要).{0,10}?(后手|后拿|让对方先)",
            combined[: min(400, len(combined))],
        ):
            return "first"
    if re.search(r"(后手|後手|后拿|後拿|后取|後取|让对方先|讓對方先)", combined) and not re.search(
        r"(先手|先拿|先取)",
        combined,
    ):
        return "second"
    if re.search(r"(先手|先拿|先取)", combined) and not re.search(r"(后手|後手)", combined):
        return "first"
    return None


def _answer_position_hint(bundle: str) -> str | None:
    """整段文本粗判（仅用于无 meta 时的兼容）；有 meta 请用 _answer_position_from_meta。"""
    a = str(bundle or "")
    if re.search(r"后手|後手|后拿|後拿|后取|後取|让对方先|讓對方先", a):
        return "second"
    if re.search(r"先手|先拿|先取", a) and not re.search(r"后手|後手", a):
        return "first"
    return None


def _user_stated_numbers(text: str) -> set[str]:
    t = _normalize_student_text(text)
    s = set(re.findall(r"\d+", t))
    if re.search(r"[三叁]", t):
        s.add("3")
    if re.search(r"两|二|贰", t):
        s.add("2")
    if re.search(r"[四肆]", t):
        s.add("4")
    return s


_ZH_TAKE_COUNT = {
    "一": "1",
    "二": "2",
    "两": "2",
    "三": "3",
    "四": "4",
    "五": "5",
    "六": "6",
    "七": "7",
    "八": "8",
    "九": "9",
    "十": "10",
}


def _normalize_take_count_token(tok: str) -> str | None:
    t = (tok or "").strip()
    if not t:
        return None
    if t.isdigit():
        return t
    return _ZH_TAKE_COUNT.get(t)


def _bundle_answer_strategy(meta: pd.Series) -> str:
    return "\n".join(str(meta.get(k, "") or "") for k in ("answer", "strategy"))


def _meta_first_player_opening_take_digit(meta: pd.Series) -> str | None:
    """answer/strategy 里「先手第一步取几颗」的明确数字（仅当标答为先手侧时使用）。"""
    b = _bundle_answer_strategy(meta)
    pats = [
        r"先手[^。\n]{0,30}?拿走\s*(\d+)\s*[颗个粒枚]?",
        r"先手[^。\n]{0,30}?(?:拿|取|走)\s*(\d+)\s*[颗个粒枚]?",
        r"选择先手[^。\n]{0,30}?(?:拿|取|走)\s*(\d+)",
        r"先手[^。\n]{0,30}?拿走\s*(两|二|三|四|五|六|七|八|九|一)\s*[颗个粒枚]?",
        r"先手[^。\n]{0,30}?(?:拿|取|走)\s*(两|二|三|四|五|六|七|八|九|一)\s*[颗个粒枚]?",
    ]
    for pat in pats:
        m = re.search(pat, b)
        if m:
            return _normalize_take_count_token(m.group(1))
    return None


def _user_first_player_opening_take_digit(text: str) -> str | None:
    """用户话里写明的「先手拿几颗」（阿拉伯或常见中文）。"""
    t = _normalize_student_text(text)
    m = re.search(
        r"先手拿\s*(\d+)\s*[颗个粒枚]?|"
        r"先手\s*拿\s*(\d+)\s*[颗个粒枚]?|"
        r"先手[^。\n]{0,8}?拿\s*(\d+)\s*[颗个粒枚]?|"
        r"先手拿\s*(两|二|三|四|五|六|七|八|九|一)\s*[颗个粒枚]?|"
        r"先手\s*拿\s*(两|二|三|四|五|六|七|八|九|一)\s*[颗个粒枚]?",
        t,
    )
    if not m:
        return None
    for g in m.groups():
        if g:
            return _normalize_take_count_token(g)
    return None


def _opening_take_conflicts_meta_first_player(
    user_text: str, meta: pd.Series
) -> bool:
    """用户声明的先手第一步颗数与标答先手第一步不一致（用于收紧 strat_ok）。"""
    if _answer_position_from_meta(meta) != "first":
        return False
    ud = _user_first_player_opening_take_digit(user_text)
    md = _meta_first_player_opening_take_digit(meta)
    return bool(ud is not None and md is not None and ud != md)


def _user_gives_take_strategy(text: str) -> bool:
    t = _normalize_student_text(text)
    if re.search(
        r"(对方|他|她|对手|對方).{0,8}(拿|取).{0,6}\d|我拿\s*\d+|我取\s*\d+|"
        r"你拿\s*\d+|你取\s*\d+|凑\s*[到成為为]?\s*\d+|"
        r"和\s*为\s*\d+|合\s*(?:计|起来)?\s*\d+|"
        r"每轮.{0,4}?\d+|"
        # 「每轮…拿/凑…3颗」类：中间可隔较多字，如「每轮都控制两个人一起拿3颗糖」
        r"每轮[\s\S]{0,50}?(?:拿|取|凑|和|共|合|控制|补齐).{0,20}?\d+\s*[颗个粒枚]|"
        r"每轮[\s\S]{0,40}?\d+\s*[颗个粒枚]|"
        r"(?:两人|两个人|俩人|一起|二人).{0,18}?(?:拿|取).{0,10}?\d+|"
        r"一起\s*拿\s*\d+|"
        # 除法余数 + 首轮取子（巴什类口答）
        r"商\s*\d+\s*余\s*\d+|"
        r"除(?:以|完)?[\s\S]{0,8}?\d+[\s\S]{0,10}?余\s*\d+|"
        r"先手.{0,8}拿\s*\d+\s*[颗个粒枚]|先手拿\s*\d+|"
        r"先[手]?.{0,4}拿\s*(?:两|二|三|四|五|六|七|八|九|十|\d+)\s*[颗个粒枚]?|"
        r"先取\s*\d+\s*[颗个粒枚]|"
        r"核心\s*(?:数|周期)?\s*\d+|锁定\s*(?:为)?\s*\d+|周期\s*\d+|模\s*\d+",
        t,
    ):
        return True
    if _user_stated_numbers(t) and re.search(
        r"凑|和为|合起来|核心|锁定|周期|模|每轮|一起拿|控制|互补|补齐|商|余|除|先手拿|先手.*拿",
        t,
    ):
        return True
    return False


def _strategy_consistent_with_field(
    user_text: str, strategy_field: str, meta: pd.Series | None = None
) -> bool:
    if not _user_gives_take_strategy(user_text):
        return False
    if meta is not None and _opening_take_conflicts_meta_first_player(user_text, meta):
        return False
    st = str(strategy_field or "").strip()
    ut = _normalize_student_text(str(user_text or ""))
    if not st:
        return True
    sn = set(re.findall(r"\d+", st))
    un = _user_stated_numbers(ut)
    if sn and un and not (sn & un):
        return False
    if re.search(r"凑\s*\d+|和\s*为\s*\d+|余\s*\d+|=\s*\d+", st):
        for m in re.finditer(
            r"凑\s*(\d+)|和\s*为\s*(\d+)|余\s*(\d+)|=\s*(\d+)|\+\s*(\d+)\s*=\s*(\d+)",
            st,
        ):
            for g in m.groups():
                if g and g in un:
                    return True
    if re.search(r"凑|和为|核心|周期|锁定", ut) and (sn & un):
        return True
    return bool(sn & un) if sn else bool(un)


def _user_states_executable_take_plan(text: str) -> bool:
    """已说出可执行的轮次拿法（对方拿几我拿几、先/后手拿几颗、第一步拿等），非仅枚举必败点。"""
    t = _normalize_student_text(text)
    return bool(
        re.search(
            r"(对方|对手|他|她|對方).{0,12}(?:拿|取).{0,10}\d|"
            r"先[手]?.{0,6}拿\s*(?:两|二|三|四|五|六|七|八|九|十|\d+)\s*[颗个粒枚]?|"
            r"后[手]?.{0,6}拿\s*(?:两|二|三|四|五|六|七|八|九|十|\d+)\s*[颗个粒枚]?|"
            r"第\s*\d+\s*步.{0,8}拿\s*\d+|"
            r"第\s*一\s*步.{0,8}拿\s*\d+|"
            r"第一步.{0,10}拿\s*\d+|"
            r"首轮.{0,8}拿\s*\d+|"
            r"每轮[\s\S]{0,45}?(?:凑|和|补齐|共拿|合取).{0,14}?\d+",
            t,
        )
    )


def _user_cites_beishu_or_enumeration_only(text: str) -> bool:
    """
    像在倒推/枚举必败（必胜）点，且尚未写出通式/周期（见 _user_expresses_beishu_pattern_abstraction）。
    可与「所以应该后手/先手」等同句并存——仍视为枚举阶段，需追问规律。
    例：「2,7,9是必败！」「3,6,9是必败所以后手」
    """
    t = _normalize_student_text(text)
    if _user_states_executable_take_plan(text):
        return False
    if _user_expresses_beishu_pattern_abstraction(text):
        return False
    if not re.search(
        r"(必败|必胜|输面|赢面|必败态|必胜态|P态|N态|倒推|逆推|枚举|试.?算|"
        r"周期|公差|递推|验证|推导)",
        t,
    ):
        return False
    nums = re.findall(r"\d{1,3}", t)
    if len(nums) >= 2:
        return True
    if re.search(r"\d{1,3}\s*[,，、]\s*\d{1,3}", t):
        return True
    return bool(re.search(r"\d{1,3}\s*(?:和|与|跟)\s*\d{1,3}", t))


def _user_expresses_beishu_pattern_abstraction(text: str) -> bool:
    """已用通式/周期/k 的式子等概括必败（或必胜）点规律，而非仅列几个数。"""
    t = _normalize_student_text(text)
    return bool(
        re.search(
            r"(?:7k|4k|5k|nk|\d+\s*k\s*[+\＋加]|k\s*为|非负整数|"
            r"通式|通项|一般形式|形如|写成|表示为|等价于|"
            r"周期|每隔|间距|公差|等差|递推式|"
            r"模\s*\d+\s*余|除以\s*\d+.{0,10}余.{0,6}规律|"
            r"(?:每次|下一轮).{0,8}(?:加|多)\s*\d+.{0,15}?(?:再|然后).{0,8}(?:加|多)\s*\d+)",
            t,
            re.I,
        )
    )


def _user_rhetoric_beishu_surface_only(text: str) -> bool:
    """只说到避开/留给对方必败态等，无数值枚举也无通式。"""
    t = _normalize_student_text(text)
    if _user_position_hint(t) or _ultra_short_position_bucket(t):
        return False
    if _user_states_executable_take_plan(text):
        return False
    if _user_cites_beishu_or_enumeration_only(text):
        return False
    if _user_expresses_beishu_pattern_abstraction(text):
        return False
    return bool(
        re.search(
            r"(避开|躲开|绕开|不进|别落到|离开|不要进).{0,18}(必败|输面|P态|输态)",
            t,
        )
        or re.search(
            r"(把|甩|丢|留|送给|搁|留给).{0,14}(对方|对手|他|她).{0,18}(必败|输面)",
            t,
        )
        or re.search(
            r"(必败|输面).{0,12}(丢|给|让|留给)(对方|他|她|对手)",
            t,
        )
        or re.search(r"(拖入|陷入|保持|弄到).{0,10}(必败|输面)", t)
    )


BEISHU_ENUM_PHASE1_DIRECTIVE = """
【枚举必败点·第一轮：仅引导，不讲通式、不收题】
系统已将本题标为「枚举必败点待抽象规律」的**第一轮**回复：学生在等下一轮用自己的话概括数学规律。
请：
1）先依据 question_text、math_logic、strategy、answer **校验**学生已有发现（先后手、若干必败数等）；正确处可简短肯定，错误处要温和指出并带他重算；
2）用 **1～2 个具体问题**引导他**自己**把规律说清楚（例如：这些必败数间隔有何共性？能否写成含 k 的式子或「模几余几」？）；
3）本轮**不要**直接写出完整通式、**不要**展开从 1 颗糖逐步倒推的长教程（除非学生明确追问某一步）；
4）**不要**表示本题已结束、**不要**提「下一题」「做完」或类似收题语；
5）可内化【脚手架参考】的意图，**勿整段照念**脚手架原文。
"""


def _clip(s: object, n: int) -> str:
    t = str(s or "").strip()
    if len(t) <= n:
        return t
    return t[: n - 1] + "…"


ACCUM_STATE_VALUES = frozenset({"正确", "错误", "缺失"})

_OBSERVER_TO_STAT_KEY = (
    ("position", "accum_position"),
    ("strategy", "accum_strategy"),
    ("pattern", "accum_pattern"),
)

_ACCUM_DIM_TO_FIRST_ROUND = {
    "accum_position": "accum_position_first_correct_round",
    "accum_strategy": "accum_strategy_first_correct_round",
    "accum_pattern": "accum_pattern_first_correct_round",
}


def _accum_dims_snapshot(stats: dict) -> dict:
    """供 register_accum_progress_turn 比对；缺省或非法值视为「缺失」。"""
    out = {}
    for _, sk in _OBSERVER_TO_STAT_KEY:
        v = stats.get(sk)
        out[sk] = v if v in ACCUM_STATE_VALUES else "缺失"
    return out


def clear_solution_accum_marks(stats: dict) -> None:
    stats["accum_position"] = "缺失"
    stats["accum_strategy"] = "缺失"
    stats["accum_pattern"] = "缺失"
    for fk in _ACCUM_DIM_TO_FIRST_ROUND.values():
        stats[fk] = None


def register_accum_progress_turn(stats: dict, before: dict, after: dict) -> None:
    """任一轮中若有某维度从非「正确」→「正确」，计为一次「累积进展」。"""
    keys = ("accum_position", "accum_strategy", "accum_pattern")
    flipped = any(
        before.get(k) != "正确" and after.get(k) == "正确" for k in keys
    )
    if flipped:
        stats["accum_progress_turns"] = int(stats.get("accum_progress_turns") or 0) + 1


def merge_observer_accum(stats: dict, detail: dict | None) -> None:
    """
    将 Observer 单轮三维度合并进跨轮累积：「正确」「错误」覆盖；「缺失」或缺字段则保持累积不变。
    detail 中 position/strategy/pattern 已为「正确」「错误」「缺失」或 None（不更新）。
    """
    if not isinstance(detail, dict):
        return
    before = _accum_dims_snapshot(stats)
    for obs_k, st_k in _OBSERVER_TO_STAT_KEY:
        v = detail.get(obs_k)
        if v in ("正确", "错误"):
            stats[st_k] = v
    after = _accum_dims_snapshot(stats)
    cur_round = int(stats.get("total_rounds") or 0)
    for sk in ("accum_position", "accum_strategy", "accum_pattern"):
        fk = _ACCUM_DIM_TO_FIRST_ROUND[sk]
        if before.get(sk) != "正确" and after.get(sk) == "正确":
            if stats.get(fk) is None:
                stats[fk] = cur_round
    register_accum_progress_turn(stats, before, after)


def accum_three_dims_all_correct(stats: dict) -> bool:
    """累积三维度（先后手/答案侧、策略、规律）均为「正确」时，才视为可收题的「全对」。"""
    snap = _accum_dims_snapshot(stats)
    return all(snap[k] == "正确" for k in ("accum_position", "accum_strategy", "accum_pattern"))


def _scaffolding_behavior_names(sdf: pd.DataFrame) -> set[str]:
    return {str(x).strip() for x in sdf["behavior"].dropna().unique() if str(x).strip()}


def classify_behavior(
    user_text: str,
    meta: pd.Series,
    sdf: pd.DataFrame,
    *,
    recent_messages: list | None = None,
) -> tuple[str, dict]:
    """
    behavior 由 Observer 初选；若 position、strategy 已为「正确」而 pattern 尚未「正确」，
    则与「解题：全对」矛盾，收敛为「解题：枚举必败点待抽象规律」（表中须存在该 behavior）。
    """
    b, d = observer_judge(user_text, meta, sdf, recent_messages=recent_messages)
    valid = _scaffolding_behavior_names(sdf)
    if b == BEHAVIOR_CORRECT_LEGACY and BEHAVIOR_CORRECT in valid:
        d = {**d, "behavior_normalized_from_legacy_name": True}
        b = BEHAVIOR_CORRECT
    if (
        b == BEHAVIOR_CORRECT_ALIAS
        and BEHAVIOR_CORRECT in valid
        and BEHAVIOR_CORRECT_ALIAS != BEHAVIOR_CORRECT
    ):
        d = {**d, "behavior_normalized_from_short_name": True}
        b = BEHAVIOR_CORRECT
    pos, strat, pat = d.get("position"), d.get("strategy"), d.get("pattern")
    if (
        b == BEHAVIOR_CORRECT
        and pos == "正确"
        and strat == "正确"
        and pat != "正确"
        and BEHAVIOR_BEISHU_ENUM_PENDING in valid
    ):
        d = {
            **d,
            "behavior_dim_adjust": "position+strategy 正确且 pattern 非正确 → 待抽象规律",
            "observer_behavior_before_dim_adjust": b,
        }
        b = BEHAVIOR_BEISHU_ENUM_PENDING
    return b, {**d, "behavior_source": "Observer（DeepSeek）"}


FORCE_LECTURE_SYSTEM = """
你是数学导师。学生在本题选择了「强制讲解并跳过」，需要你**一次性把本题讲清楚**，他再看下一题。
要求：
- 严格依据提供的 math_logic、strategy，并结合题干；可参考 answer **直接写出结论**（如先后手、关键操作或不变量），不要卖关子。
- 用**一段连贯说明**即可，总字数控制在 **350 字以内**；不要过多小节标题，不要「你来试试」等互动语。
- 不要逐句照抄材料原文，用**自己的话概括**核心思路即可。
"""


def _force_lecture_fallback(meta: pd.Series) -> str:
    qt = _clip(meta.get("question_text", ""), 500)
    ml = _clip(meta.get("math_logic", ""), 1500)
    st = _clip(meta.get("strategy", ""), 1500)
    an = _clip(meta.get("answer", ""), 500)
    parts = [
        "（模型暂不可用，以下为根据题库整理的要点摘要。）\n",
        f"**题干节选**：{qt}\n\n",
        f"**数学要点**：{ml}\n\n",
        f"**策略要点**：{st}\n\n",
    ]
    if an.strip():
        parts.append(f"**参考结论**：{an}\n")
    return "".join(parts)


def _math_logic_wrapup_fallback(meta: pd.Series) -> str:
    ml = _clip(meta.get("math_logic", ""), 2000)
    st = _clip(meta.get("strategy", ""), 1500)
    an = _clip(meta.get("answer", ""), 600)
    return (
        "（模型暂不可用，以下为数学要点与策略摘要。）\n\n"
        f"**数学要点**：{ml}\n\n**策略要点**：{st}\n\n**结论参考**：{an}\n"
    )


def generate_math_logic_wrapup_reply(
    user_text: str,
    meta: pd.Series,
    messages: list | None = None,
) -> tuple[str, dict]:
    """
    第二轮仍未概括对：用与「全对收题」相同的主系统人设 + 专用模式标记，
    强制陈述句代讲（避免独立短 system 被模型当成「继续引导」）。
    """
    hist_lines: list[str] = []
    for m in (messages or [])[-8:]:
        role = m.get("role", "")
        c = str(m.get("content", ""))[:700]
        hist_lines.append(f"{role}: {c}")
    history = "\n".join(hist_lines) if hist_lines else "（无）"
    payload = (
        "[模式:枚举二轮教师代讲结题]\n\n"
        f"【最近对话节选】\n{history}\n\n"
        f"【学生本轮】\n{str(user_text or '').strip()[:2000]}\n\n"
        f"【本题题干】\n{meta.get('question_text', '')}\n\n"
        f"【数学要点 math_logic】\n{_clip(meta.get('math_logic', ''), 6000)}\n\n"
        f"【策略说明 strategy】\n{_clip(meta.get('strategy', ''), 6000)}\n\n"
        f"【标答 answer — 讲解时可内化，勿机械照念】\n{_clip(meta.get('answer', ''), 2000)}\n"
    )
    text, raw = llm_text_reply(
        TUTOR_REPLY_SYSTEM.strip(),
        payload,
        temperature=0.3,
        timeout_sec=180,
        retries=3,
        models=tutor_model_chain(),
    )
    if text and text.strip():
        return text.strip(), raw
    return _math_logic_wrapup_fallback(meta), raw


def generate_force_lecture_reply(meta: pd.Series) -> tuple[str, dict]:
    """强制跳过本题前：据 math_logic + strategy（及题干、answer）生成一段简短讲解。"""
    payload = (
        f"【题干】\n{_clip(meta.get('question_text', ''), 3500)}\n\n"
        f"【math_logic】\n{_clip(meta.get('math_logic', ''), 6000)}\n\n"
        f"【strategy】\n{_clip(meta.get('strategy', ''), 6000)}\n\n"
        f"【answer 参考】\n{_clip(meta.get('answer', ''), 800)}\n"
    )
    text, raw = llm_text_reply(
        FORCE_LECTURE_SYSTEM.strip(),
        payload,
        temperature=0.25,
        timeout_sec=120,
        retries=3,
        models=tutor_model_chain(),
    )
    if text and text.strip():
        return text.strip(), raw
    return _force_lecture_fallback(meta), raw


def generate_factual_reply(user_text: str, meta: pd.Series) -> tuple[str, dict]:
    payload = (
        f"题目：\n{meta.get('question_text', '')}\n\n"
        f"answer：\n{meta.get('answer', '')}\n\n"
        f"strategy：\n{meta.get('strategy', '')}\n\n"
        f"math_logic：\n{meta.get('math_logic', '')}\n\n"
        f"问：\n{user_text}"
    )
    data, raw_meta = llm_json(
        FACTUAL_QA_SYSTEM,
        payload,
        temperature=0.15,
        retries=3,
        timeout_sec=180,
        models=tutor_model_chain(),
    )
    msg = _coerce_message_any(data)
    trace = {"ok": bool(msg), "attempts": raw_meta.get("attempts")}
    if msg:
        return msg, trace
    return (
        "我这边暂时连不上模型。你可以先对照题干里的总数和「每次可取几颗」，想想每一轮两人一共拿走几颗最利于控局面；想通后再说说你的先后手判断。",
        trace,
    )


def generate_tutor_reply(
    user_text: str,
    meta: pd.Series,
    behavior: str,
    scaffold: str,
    messages: list,
    *,
    completion_mode: bool,
    scaffold_focus: str = "",
    beishu_enum_phase1_guidance: bool = False,
    mastery_state: dict | None = None,
) -> tuple[str | None, dict]:
    hist_lines: list[str] = []
    for m in messages[-10:]:
        role = m.get("role", "")
        c = str(m.get("content", ""))[:800]
        hist_lines.append(f"{role}: {c}")
    history = "\n".join(hist_lines) if hist_lines else "（无）"

    mode_line = "[模式:全对收题]\n\n" if completion_mode else "[模式:常规引导]\n\n"
    beishu_directive = ""
    if behavior == BEHAVIOR_BEISHU_ENUM_PENDING and beishu_enum_phase1_guidance:
        beishu_directive = BEISHU_ENUM_PHASE1_DIRECTIVE.strip() + "\n\n"
    fc = str(scaffold_focus or "").strip()
    ms = mastery_state if isinstance(mastery_state, dict) else {}
    position_state = str(ms.get("position") or ms.get("accum_position") or "缺失").strip() or "缺失"
    strategy_state = str(ms.get("strategy") or ms.get("accum_strategy") or "缺失").strip() or "缺失"
    pattern_state = str(ms.get("pattern") or ms.get("accum_pattern") or "缺失").strip() or "缺失"
    mastery_block = (
        "【当前题内累计掌握状态 — 必须遵守】\n"
        f"position（先后手/结论）：{position_state}\n"
        f"strategy（可操作策略）：{strategy_state}\n"
        f"pattern（数学规律层）：{pattern_state}\n"
        "说明：这是当前题目内跨轮累积后的状态；之前学生已说对的维度会保持正确。"
        "常规引导模式下，不要替学生补齐仍为「缺失」或「错误」的维度，只能引导他自己说出。\n\n"
    )
    focus_block = (
        f"【本轮引导焦点 focus — 内化关切优先级，勿照念】\n{_clip(fc, 1200)}\n\n"
        if fc
        else "【本轮引导焦点 focus】\n（表中未配置，按 behavior 与 scaffold 自行把握。）\n\n"
    )
    user_payload = (
        f"{mode_line}"
        f"{beishu_directive}"
        f"【本题题干】\n{meta.get('question_text', '')}\n\n"
        f"【数学要点 math_logic】\n{_clip(meta.get('math_logic', ''), 6000)}\n\n"
        f"【策略说明 strategy】\n{_clip(meta.get('strategy', ''), 6000)}\n\n"
        f"【内部标答 answer — 勿照抄给学生】\n{_clip(meta.get('answer', ''), 2000)}\n\n"
        f"【本轮判定 behavior】\n{behavior}\n\n"
        f"{mastery_block}"
        f"{focus_block}"
        f"【脚手架参考 — 内化意图，勿照念】\n{_clip(scaffold, 2500)}\n\n"
        f"【最近对话】\n{history}\n\n"
        f"【学生本轮】\n{user_text}"
    )
    chains = (tutor_model_chain(), ["deepseek-chat"])

    def _run(sys_p: str | None, user_p: str) -> tuple[str | None, dict]:
        all_attempts: list = []
        last_meta: dict = {}
        for chain in chains:
            text, m = llm_text_reply(
                sys_p or "",
                user_p,
                temperature=0.35,
                timeout_sec=180,
                retries=3,
                models=chain,
            )
            all_attempts.extend(m.get("attempts") or [])
            last_meta = m
            if text and text.strip():
                last_meta = {**last_meta, "attempts": all_attempts}
                return text.strip(), last_meta
        return None, {**last_meta, "attempts": all_attempts}

    text, meta_tr = _run(TUTOR_REPLY_SYSTEM, user_payload)
    if text:
        return text, meta_tr

    merged = (
        "以下是导师人设与任务要求，请严格按此回复学生：\n"
        f"{TUTOR_REPLY_SYSTEM}\n\n---\n\n"
        f"{user_payload}"
    )
    text2, meta2 = _run(None, merged)
    if text2:
        return text2, meta2

    short_user = (
        f"{mode_line}"
        f"题干：{_clip(meta.get('question_text', ''), 1200)}\n\n"
        f"math_logic：{_clip(meta.get('math_logic', ''), 2000)}\n\n"
        f"strategy：{_clip(meta.get('strategy', ''), 2000)}\n\n"
        f"behavior：{behavior}\n"
        f"当前题内累计掌握状态：position={position_state}；strategy={strategy_state}；pattern={pattern_state}。"
        "常规引导时不要替学生补齐缺失/错误维度。\n"
        f"focus：{_clip(fc, 800)}\n"
        f"脚手架要点：{_clip(scaffold, 1500)}\n\n"
        f"学生本轮：{user_text}"
    )
    return _run(TUTOR_REPLY_SYSTEM, short_user)


def reply_fallback(
    behavior: str, scaffold: str, meta: pd.Series, *, api_hint: str = ""
) -> str:
    ml = _clip(meta.get("math_logic", ""), 280)
    sc = _clip(scaffold, 400)
    hint = f"\n\n【接口诊断】{api_hint}" if api_hint else ""
    proxy_tip = (
        "\n\n若你在公司网/校园网，可设环境变量 `HTTPS_PROXY` 或 `DEEPSEEK_HTTP_PROXY` 再试。"
    )
    return (
        "（模型未返回正文，已用备用提示。）"
        f"{hint}{proxy_tip}\n\n"
        f"先别急着要答案，结合题干想一想：{ml}\n\n"
        f"若仍卡住，可参考这些方向（用你自己的话说出来）：{sc}"
    )
