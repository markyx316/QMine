#!/usr/bin/env python
"""One intent frame for comparing surfaces whose taxonomies were built separately.

WHY A SHARED FRAME AND NOT A JOIN. The five search runs and the assistant run
share zero class codes (112 codes, 112 in the union). They are semantically
parallel and structurally unjoinable, and they differ in GRANULARITY: `med-pool`
splits a health question into 症状原因 / 用法用量 / 判读个人结果 / 严重程度, while
`ai04` has one `MEDICAL_CONSULTATION_AND_DOSAGE`, and `ai04` has no bare-entity
class at all. Mapping both onto a shared frame is therefore a JUDGEMENT at every
boundary, and a share difference between surfaces can be produced by the mapping
rather than by users.

So the frame is used two ways, and they are not equally trustworthy:

  PRIMARY    both surfaces labelled by ONE instrument (same prompt, same model,
             blind to source), checked by a second model from another lab
  SECONDARY  each run's own labels mapped through `CROSSWALK` below — full data,
             useful for description and examples, and read against PRIMARY

Every mapping decision is in this file so it can be argued with.
"""
from __future__ import annotations

FRAME: dict[str, tuple[str, str]] = {
    "U01": ("导航直达", "想打开某个网站、App、官方入口、账号主页、频道直播、论坛或下载页；仅输入工具名（如“翻译”“在线翻译”）也算。例：芒果TV、cctv5在线直播、工商银行app下载、海康威视股吧"),
    "U02": ("裸实体", "只输入一个名称、代码或词，没有任何动作、问题或修饰。例：蔡奇、600643、布洛芬、延禧攻略、合肥工业大学"),
    "U03": ("事实与数值", "要一个具体事实值：价格行情、分数线、日期时间、天气、数量、年龄籍贯、电话号码、读音、换算或计算结果。例：今日金价、华南理工录取分数线、蔡奇哪里人、10000韩元是多少人民币"),
    "U04": ("解释与介绍", "要对概念、事物、人物、作品、专业、症状表现等的解释或介绍。例：集合竞价什么意思、五味子的功效与作用、藏海传简介、计算机专业学什么"),
    "U05": ("核实与动态", "要判断一个说法、传闻、平台、来电是否属实可靠，或要某件事、某人的最新进展与状态。例：信用飞贷款可靠吗、赵本山死了吗、今天黄金为什么大跌、银耳隔夜能吃吗"),
    "U06": ("办事与操作", "要做成一件事的步骤、流程、办法或去处（非个案判断）。例：怎样关闭花呗、中药怎么熬、腋下异味挂什么科、这个蓝牙耳机怎样开关"),
    "U07": ("个案判断与建议", "针对用户自己或一个具体个案要判断、建议或预测：该不该、能不能治好、吃什么药、我的结果是否正常、怎么选、谁会赢。例：301630值得申购吗、我血压89/65正常吗、感冒头疼吃什么药好、志愿怎么填"),
    "U08": ("清单与推荐", "要一组条目、排行、名单或推荐片单。例：中国四大行是哪几个、刘宇宁主演的电视剧、适合二人看的电影、2026最火的歌"),
    "U09": ("获取现成内容", "要拿到资源本身：在线观看、下载、网盘、原文全文、图片、电子课本、答题答案、按线索找歌找书找片。例：甄嬛传免费观看、岳阳楼记原文、股癣图片、蚂蚁庄园今日答案"),
    "U10": ("生成与编辑", "要系统新写、画、翻译给定内容、续写、改写，或编辑用户上传的图片视频。例：帮我写一篇家长寄语、画一只猫、把这句翻译成英文、去掉图片背景"),
    "U11": ("会话与系统指令", "依赖上文或作用于系统本身：修改上一轮输出、确认寒暄、评价或抱怨助手、控制推荐流。例：再短一点、换一个、你说错了、少推这种视频"),
    "U12": ("违规或灰色内容", "索取成人色情内容、黑料爆料，或普通人的非公开隐私信息。"),
    "U13": ("无法判定", "脱离上下文无法判断用户要什么的碎片。"),
}

#: (run, class code) -> frame code. Comments mark the boundary calls.
CROSSWALK: dict[tuple[str, str], str] = {
    # ---- fin-pool ----
    ("fin-pool", "LOOKUP_MARKET_QUOTE"): "U03",
    ("fin-pool", "BARE_CODE_ENTITY_LOOKUP"): "U02",
    ("fin-pool", "VIEW_PRICE_CHART"): "U03",
    ("fin-pool", "GET_MARKET_NEWS"): "U05",
    ("fin-pool", "LOOKUP_MARKET_CALENDAR_AND_STATUS"): "U03",
    ("fin-pool", "NAVIGATE_STOCK_FORUM"): "U01",
    ("fin-pool", "NAVIGATE_OFFICIAL_ENTRY"): "U01",
    ("fin-pool", "LOOKUP_SERVICE_CONTACT"): "U03",        # a number is a fact
    ("fin-pool", "IDENTIFY_UNKNOWN_CALLER"): "U05",       # is this caller legitimate
    ("fin-pool", "VERIFY_PLATFORM_LEGITIMACY"): "U05",
    ("fin-pool", "EXPLAIN_FINANCE_CONCEPT"): "U04",
    ("fin-pool", "HOWTO_ACCOUNT_SUPPORT"): "U06",
    ("fin-pool", "COMPUTE_FINANCIAL_AMOUNT"): "U03",
    ("fin-pool", "ENUMERATE_LIST"): "U08",
    ("fin-pool", "GET_INVESTMENT_OPINION_FORECAST"): "U07",
    ("fin-pool", "GET_DAILY_QUIZ_ANSWER"): "U09",
    ("fin-pool", "OUT_OF_DOMAIN_OR_UNINTELLIGIBLE"): "U13",
    # ---- med-pool ----
    ("med-pool", "LOOKUP_EFFICACY"): "U04",
    ("med-pool", "CLAIM_VERIFICATION"): "U05",
    ("med-pool", "CONTRAINDICATION_SAFETY_LOOKUP"): "U04",
    ("med-pool", "SYMPTOM_CAUSE"): "U04",                 # boundary: general cause, not a personal verdict
    ("med-pool", "SYMPTOM_CHECKLIST"): "U04",
    ("med-pool", "TREATMENT_REQUEST"): "U07",             # boundary: asks what to take
    ("med-pool", "QUICK_CURE_DEMAND"): "U07",
    ("med-pool", "DOSAGE_REQUEST"): "U03",                # boundary: a label parameter
    ("med-pool", "NORMAL_VALUE_CHECK"): "U03",
    ("med-pool", "PERSONAL_RESULT_INTERPRETATION"): "U07",
    ("med-pool", "TERM_DEFINITION"): "U04",
    ("med-pool", "SEVERITY_PROGNOSIS"): "U07",
    ("med-pool", "ROUTE_TO_CARE"): "U06",
    ("med-pool", "PROCEDURE_LOGISTICS"): "U03",
    ("med-pool", "FOOD_SUITABILITY"): "U05",              # a yes/no verdict on a general claim
    ("med-pool", "PREGNANCY_FERTILITY_DECISION"): "U07",
    ("med-pool", "IMAGE_SELF_MATCH"): "U09",
    ("med-pool", "SELF_CARE_PROCEDURE"): "U06",
    ("med-pool", "AMBIGUOUS_HEALTH_ENTITY"): "U02",
    ("med-pool", "OUT_OF_DOMAIN_NON_HEALTH"): "U13",
    # ---- edu-pool ----
    ("edu-pool", "INSTITUTION_LOOKUP"): "U02",
    ("edu-pool", "SCHOOL_PLATFORM_NAVIGATION"): "U01",
    ("edu-pool", "OFFICIAL_EDU_PORTAL_NAVIGATION"): "U01",
    ("edu-pool", "SCHOOL_TIER_VERDICT"): "U03",
    ("edu-pool", "SCORE_LINE_LOOKUP"): "U03",
    ("edu-pool", "EXAM_SCHEDULE_LOOKUP"): "U03",
    ("edu-pool", "EXAM_RESULT_LOOKUP"): "U03",
    ("edu-pool", "APPLICATION_REQUIREMENT_LOOKUP"): "U04",
    ("edu-pool", "CERTIFICATION_EXAM_LOOKUP"): "U04",
    ("edu-pool", "CREDENTIAL_SERVICE_TRANSACTION"): "U06",
    ("edu-pool", "MAJOR_CAREER_CONSULTING"): "U04",
    ("edu-pool", "ADMISSION_DECISION_ADVISORY"): "U07",
    ("edu-pool", "MEANING_EXPLANATION"): "U04",
    ("edu-pool", "TRANSLATION_REQUEST"): "U10",           # boundary: many are tool launches (U01)
    ("edu-pool", "CHAR_PRONUNCIATION_LOOKUP"): "U03",
    ("edu-pool", "CHAR_WRITING_USAGE_HELP"): "U04",
    ("edu-pool", "CLASSICAL_TEXT_ACCESS"): "U09",
    ("edu-pool", "TEXTBOOK_AND_LEARNING_MATERIAL"): "U09",
    ("edu-pool", "ZODIAC_RIDDLE_LOOKUP"): "U09",
    ("edu-pool", "OTHER_OR_AMBIGUOUS"): "U13",
    # ---- film-pool ----
    ("film-pool", "WATCH_TITLE_ONLINE"): "U09",
    ("film-pool", "BARE_TITLE_HUB"): "U02",
    ("film-pool", "RESOLVE_DESCRIPTION_TO_TITLE"): "U09",
    ("film-pool", "NAVIGATE_TO_PLATFORM"): "U01",
    ("film-pool", "INSTALL_APP"): "U01",
    ("film-pool", "SEEK_ADULT_SEXUAL_CONTENT"): "U12",
    ("film-pool", "ACQUIRE_VIA_NETDISK"): "U09",
    ("film-pool", "WATCH_LIVE_TV"): "U01",
    ("film-pool", "LOOKUP_CAST_CREW"): "U04",
    ("film-pool", "LOOKUP_PLOT_SYNOPSIS"): "U04",
    ("film-pool", "LIST_WORKS_BY_PERSON"): "U08",
    ("film-pool", "BROWSE_BY_CRITERIA"): "U08",
    ("film-pool", "ASK_AVAILABILITY"): "U03",
    ("film-pool", "LOCATE_EPISODE"): "U09",
    ("film-pool", "OTHER_NON_FILM_TV"): "U13",
    # ---- ppl-pool ----
    ("ppl-pool", "BARE_NAME_IDENTITY_LOOKUP"): "U02",
    ("ppl-pool", "BIO_PROFILE_LOOKUP"): "U04",
    ("ppl-pool", "PERSON_STATUS_NEWS_CHECK"): "U05",
    ("ppl-pool", "OFFICIAL_APPOINTMENT_VERIFY"): "U05",
    ("ppl-pool", "DEATH_HEALTH_RUMOR_CHECK"): "U05",
    ("ppl-pool", "SLOT_TO_PERSON_RESOLUTION"): "U03",
    ("ppl-pool", "PERSON_ATTRIBUTE_FACTOID"): "U03",
    ("ppl-pool", "PERSON_IMAGE_SEARCH"): "U09",
    ("ppl-pool", "FAMILY_RELATION_LOOKUP"): "U04",
    ("ppl-pool", "FAN_SUPPORT_PLATFORM_ACTION"): "U01",
    ("ppl-pool", "PERSON_NAVIGATION_ACCOUNT_LOOKUP"): "U01",
    ("ppl-pool", "LIST_COUNT_RANK_LOOKUP"): "U08",
    ("ppl-pool", "EXPLAIN_SPEECH_THEORY_REFERENCE"): "U04",
    ("ppl-pool", "ADULT_ADJACENT_PERSON_CONTENT"): "U12",
    ("ppl-pool", "GOSSIP_BLACK_MATERIAL_NAVIGATION"): "U12",
    ("ppl-pool", "NON_PERSON_CATEGORY_BROWSE_OR_UNSUPPORTED"): "U13",
    # ---- ai04 (coarser; several classes span two frame codes — see PRIMARY) ----
    ("ai04", "NON_INTENT_SOCIAL_OR_EMOTIONAL_TURN"): "U11",
    ("ai04", "NON_INTENT_FRAGMENT_UNDERSPECIFIED_INPUT"): "U13",
    ("ai04", "FEED_TUNING_COMMAND"): "U11",
    ("ai04", "TOOL_PANEL_SINGLE_STEP_COMMAND"): "U10",
    ("ai04", "GENERATE_OR_EDIT_SAFE_IMAGE"): "U10",
    ("ai04", "GENERATE_OR_REVISE_TEXT"): "U10",
    ("ai04", "CORRECT_OR_ADJUST_PREVIOUS_OUTPUT"): "U11",
    ("ai04", "FICTION_CONTINUATION_AND_ROLEPLAY"): "U10",
    ("ai04", "ADULT_OR_VIOLENT_CONTENT_REQUEST"): "U12",
    ("ai04", "FACT_VERIFICATION_AND_RUMOR_OR_STATUS_CHECK"): "U05",
    ("ai04", "NUMERIC_FACT_AND_POLICY_LOOKUP"): "U03",
    ("ai04", "EXPLAIN_OR_IDENTIFY"): "U04",
    ("ai04", "LIST_AND_RECOMMENDATION_REQUEST"): "U08",
    ("ai04", "PRICE_COST_INQUIRY"): "U03",
    ("ai04", "ENTITY_BACKGROUND_LOOKUP"): "U04",
    ("ai04", "PRIVATE_PERSON_INFO_LOOKUP"): "U12",
    ("ai04", "NAVIGATE_OFFICIAL_ENTRY_OR_TOPIC_FEED"): "U01",
    ("ai04", "GENERAL_HOWTO_GUIDE"): "U06",
    ("ai04", "MEDIA_CONTENT_SEARCH_OR_ACCESS"): "U09",
    ("ai04", "LEGAL_CONSULTATION_AND_OUTCOME_PREDICTION"): "U07",   # spans U04/U06/U07
    ("ai04", "MEDICAL_CONSULTATION_AND_DOSAGE"): "U07",             # spans U03/U04/U07
    ("ai04", "FINANCIAL_AND_VENTURE_ADVICE"): "U07",
    ("ai04", "FORTUNE_DREAM_DIVINATION"): "U07",
    ("ai04", "SPORTS_MATCH_OUTCOME_PREDICTION"): "U07",
    ("ai04", "UNLABELED"): "U13",
}


def frame_code(run: str, code: str) -> str:
    return CROSSWALK.get((run, str(code)), "U13")
