"""What may be quoted, and the seven layers that decide it.

REAL ROWS ARE THE POINT OF THE EXAMPLES, AND THAT IS EXACTLY WHY THIS EXISTS.
A class card with an invented example is useless; a class card with a real
example that should never have been reproduced is worse than useless. Five guard
layers, all of them lexicon-based, still let roughly 38 forbidden strings through
into shipped documents, because a word list can only catch the phrasings someone
thought of. So the layers are stacked deepest-first and the last two are learned
from what actually got printed.

None of this touches a statistic. Every count, share and interval is computed
over the whole corpus; the guard decides only whether a row's TEXT may appear.
A class whose rows are all blocked ships as a count with a stated reason — never
as a blank, because a blank reads as "no data".

    L1  risk layer      — rows the run's own risk screen flagged
    L2  hard rules      — vertical-independent regex (minors+sex, explicit, PII)
    L3  co-occurrence   — two lexicons anywhere in one string, not adjacent
    L4  named strings   — strings the run's own risk machinery quoted by name
    L5  never-quote     — whole classes where reproducing the text IS the harm
    L6  declared regex  — the domain profile's risk patterns, plus config
    L7  screened list   — strings a reader looked at one by one

REGEXES ARE EVALUATED THROUGH PYTHON'S `re`, NEVER `Series.str.contains`.
pandas 3 routes `str.contains` to RE2, which rejects a `一-鿿` class outright and
silently changes `\\w`/`\\s` to ASCII-only. A Unicode class read every Chinese
query as empty once already. `_mask` is the only evaluation path here.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd

# --------------------------------------------------------------- L2 / L3 core

_AGE = r"(?:[0-9]{1,2}|[一二三四五六七八九十两]{1,3})\s*岁|未成年|小学生|初中生|学生妹|中学生|女童|男童"
_LOOK = r"苗条|身材|胸|腿|足|丝袜|泳装|比基尼|性感|裸|色情|写真|私房|艳|诱惑|白虎|萝莉|玉足"
_PERSON = r"女孩|男孩|小女孩|小男孩|少女|少男|女生|男生|女童|男童|学生"

#: Two lexicons anywhere in the same string. An adjacency rule missed the long
#: persona-generation prompts, where the two kinds of word sit dozens of
#: characters apart.
QUOTE_COOC = (
    re.compile(r"少女|女孩|小女孩|女童|学生妹|萝莉|女生|JK|girl|teen", re.I),
    re.compile(r"身材|三围|胸围|裸|丝袜|泳装|比基尼|性感|情色|色气|诱惑|白丝|黑丝"
               r"|lingerie|bikini|nude", re.I),
)

QUOTE_BLOCK = re.compile(
    r"(?:做爱|性交|自慰|裸体|裸照|色情|黄片|毛片|情色|av[0-9]|无码|三级片|强奸|迷奸|约炮|援交|卖淫|嫖|"
    r"萝莉|幼女|童女|未成年.{0,4}(?:性|裸|色)|小学生.{0,4}(?:性|裸)|"
    # age/schooling marker near an appearance or sexual word, either word order
    rf"(?:{_AGE}).{{0,8}}(?:{_PERSON}).{{0,8}}(?:{_LOOK})|(?:{_AGE}).{{0,6}}(?:{_LOOK})|"
    rf"(?:{_LOOK}).{{0,8}}(?:{_AGE})|(?:少女|女生|女童|小女孩).{{0,6}}(?:{_LOOK})|"
    rf"(?:{_LOOK}).{{0,6}}(?:少女|女童|小女孩|女生|学生妹)|"
    # soft-porn image retrieval
    r"(?:大?胸|丝袜|泳装|比基尼|性感|玉足|艳星|艳女|私房照|艳照|裸照|写真)(?:美女|图|照|视频)|"
    r"美女(?:图|照|视频|秀)|"
    # body part + image noun. Dropping 胸/腿 as standalone triggers to fix a
    # false positive on 摇粒绒卫衣 made 女生的胸 / 胸部照片 quotable again; the
    # PAIR carries no such ambiguity. Deliberately NOT blocked: 胸像 (a bust),
    # 胸透 (a chest X-ray), 妆前乳 (a cosmetic) — blocking those empties cards.
    r"(?:胸|胸部|乳房|乳沟|臀|臀部|屁股|奶子|三点)(?:的)?(?:图片|照片|图$|写真|大图)|"
    r"(?:女生|美女|少女|女人|女性|妹子)(?:的)?(?:胸|胸部|乳房|臀|臀部|屁股)|"
    r"束胸|束腰|充气娃娃|哺乳照|乳房图|胸部图|翘臀|美臀|蜜臀|"
    # identifiable contact details and government id numbers
    r"(?:私人)?(?:联系方式|电话号码|手机号|家庭住址|身份证号)|私人电话|"
    r"1[3-9][0-9]{9}|[0-9]{15,18}|"
    # A NAMED CLINICIAN is a private individual's identity, and blocking it must
    # not depend on which config a `compare` happened to load — a run's resolved
    # config freezes its domain profile, so a profile-only rule protects future
    # runs and not a re-comparison of an existing one. Measured cost across eight
    # delivered corpora: at most 0.11% of rows, zero on five of them. The errors
    # it does make (a drama title like 善良的医生) are on the cheap side; missing
    # a real named doctor is not.
    r"(?:科|院)[一-鿿]{2,3}(?:医生|大夫)$|"
    r"(?:医院|保健院|卫生院|中心|诊所).{0,10}"
    r"(?:主任医师|副主任医师|主治医师|住院医师|医师|教授|专家).{0,8}医生$|"
    # reputation harm and non-consensual intimate imagery
    r"包养|艳照门|不雅照|不雅视频|乱伦|"
    # rewrite-style undressing / de-censoring instructions. Found by a risk
    # sentinel, absent from every profile: these evade a word list by changing
    # the PHRASING, not the characters. An answered request produces the image.
    r"p掉.{0,8}(?:衣|服)|(?:衣服|外衣|内衣|上衣|衣物).{0,8}(?:p掉|去掉|删掉|变透明|透视)|"
    r"透视装|透明度.{0,10}(?:100|全透)|肚脐以下|"
    r"(?:去|删|消)(?:除)?.{0,4}(?:打码|马赛克)|马赛克.{0,6}(?:去除|删除|消除|工具)|去码|"
    r"打码.{0,8}(?:去掉|删掉|去除|删除|消除)|"
    r"里番|本子下载|工口|绅士向|肉番|18禁|R18|催眠性|调教(?:少女|美少女|人妻|女仆)|凌辱|痴汉|"
    # the same universal categories in Latin script, for non-Chinese corpora
    r"\bporn|\bnsfw\b|\bhentai\b|\bnudes?\b|\bsex\s?tape\b|\brape\b|\bchild\s?porn)",
    re.I)

#: The hard-rule scan: narrower than the guard, and aimed at what got PRINTED.
#: Each is a category where reproducing the string is itself the harm, so a hit
#: is reviewed by a person rather than being silently dropped.
#: `(?<![0-9])(?:[1-9]|1[0-7])岁` and not a looser age run: `47岁` contains
#: `7岁`, and a naive pattern reads every 47-year-old as a minor.
_MINOR = (r"(?:(?<![0-9])(?:[1-9]|1[0-7])岁|[一二三四五六七八九十]{1,3}岁|未成年|初中|小学|中学|高中|学生妹|"
          r"幼儿|儿童|小孩|女孩|男孩|少女|少年|萝莉|正太)")
_SEXUAL = (r"(?:性行为|性交|做爱|口交|肛交|自慰|手淫|裸体|裸照|脱衣|内裤|内衣|胸部|乳房|阴道|阴茎|私处|下体|"
           r"发生关系|开房|援交|卖淫|嫖|色情|情色|黄片|三级片|成人片|涩涩|福利姬|擦边)")

HARD_RULES: dict[str, re.Pattern[str]] = {
    "MINOR_SEX": re.compile(rf"(?=.*{_MINOR})(?=.*{_SEXUAL})"),
    "EXPLICIT": re.compile(
        r"三级片|色情片|黄片|成人电影|av号|无码|里番|工口|情色电影|福利姬|擦边视频|"
        r"做爱|口交|肛交|自慰|手淫|裸聊|约炮|一夜情细节|性爱视频|偷拍(?:裙底|厕所|浴室)"),
    #: A PRIVATE individual's identifiers. A public figure's public identity —
    #: role, works, birthplace — is not this, and blocking it would gut a
    #: people or film corpus, which is the subject matter itself.
    "PRIVATE_PII": re.compile(
        r"(?<![0-9])1[3-9][0-9]{9}(?![0-9])|"
        r"(?<![0-9A-Za-z])[1-9][0-9]{5}(?:19|20)[0-9]{2}(?:0[1-9]|1[0-2])"
        r"(?:[0-2][0-9]|3[01])[0-9]{3}[0-9Xx](?![0-9A-Za-z])|"
        r"[一-鿿]{2,}(?:小区|花园|公寓|村)[0-9]{1,3}(?:栋|号楼|单元|室)|"
        r"(?:身份证号|手机号码是|电话号码是|家庭住址|工号是|学号是|车牌号)"),
}

#: Rules a corpus can switch on by name, rather than re-deriving the regex.
#: `NAMED_DOCTOR` identifies an individual clinician — a medical corpus's
#: equivalent of a private person's contact details. Generic titles (儿科医生,
#: 家庭医生) are excluded, because those name a speciality and not a person.
#: Titles that name a SPECIALITY or a description, not a person. Measured
#: additions from four corpora: 乳腺医生 / 产科医生 / 年轻医生 / 善良的医生 /
#: 漂亮女医生 / 什么是医生 — a drama title or a common noun, never an individual.
_GENERIC_DOC = (r"(?:中医|西医|牙科|儿科|眼科|皮肤科|妇科|男科|外科|内科|骨科|口腔|家庭|社区|私人|心理|值班|主治|实习|"
                r"全科|急诊|门诊|住院|主任|副主任|专家|乳腺|产科|骨伤|中西医|好|名|老|小|男|女|"
                r"年轻|善良的|漂亮女|有名的|什么是|做|当)")
PRESET_HARD_RULES: dict[str, str] = {
    "NAMED_DOCTOR": (rf"(?:科|院)[一-鿿]{{2,3}}(?:医生|大夫)$|"
                     rf"^(?!{_GENERIC_DOC}(?:医生|大夫)$)[一-鿿]{{2,3}}(?:医生|大夫)$|"
                     r"(?:医院|保健院|卫生院|中心|诊所).{0,10}"
                     r"(?:主任医师|副主任医师|主治医师|住院医师|医师|教授|专家).{0,8}医生$"),
}

#: Risk-category keywords for which QUOTING the row reproduces the harm, as
#: opposed to the far commoner case where the risk is in how a system would
#: ANSWER. Whole-class blocking on every risk-flagged class costs 11%–23% of
#: quotable rows and buys nothing: quoting "请假条模板" harms no one.
#: NO SINGLE-CHARACTER TERMS. Every entry is a compound, because one Chinese
#: character carries meanings this rule must not act on. Measured: a bare `裸`
#: matched `裸名称` ("bare name") inside the definition of 查询物品或活动的功效与作用
#: and blocked **28.6% of a medical corpus** — 12,602 rows whose hazard is how a
#: system answers, not whether the query may be repeated. Quoting 阿司匹林的功效
#: harms nobody.
_QUOTE_IS_THE_HARM = re.compile(
    r"色情|性化|露骨|情色|裸体|裸照|裸露|全裸|半裸|裸聊|淫秽|猥亵|性行为|性暗示|"
    r"未成年|幼女|萝莉|儿童.{0,4}性|性.{0,4}儿童|"
    r"自杀|自残|自伤|轻生|危机干预|"
    r"个人信息|身份证号|联系方式|可识别到个人|隐私影像|"
    r"sexual|explicit|minor|self.?harm|suicide|adult content|nsfw|porn|"
    r"personal.?data|personally identifiab",
    re.I)

#: THIS LEXICON COVERS THE UNIVERSAL CATEGORIES ONLY, and deliberately so.
#: Whether a vertical's own sensitive topic is one where quoting reproduces the
#: harm is an editorial judgment about that corpus, and inferring it from a word
#: list would be guessing with a guard. A finance corpus whose owner decides
#: that debt-collection and credit-repair rows must never be quoted says so in
#: `pooled.never_quote_classes` / `pooled.extra_quote_patterns`, where the
#: decision is visible, attributable and reversible.

_QUOTED = re.compile(r"[“”\"「」]([^“”\"「」]{2,40})[“”\"「」]")

#: Spans a reader would recognise as a quotation, used to find what a document
#: actually printed.
SPAN = re.compile(r"[「『“]([^」』”\n]+)[」』”]")


def _mask(rx: re.Pattern[str] | list[re.Pattern[str]] | None, q: pd.Series) -> pd.Series:
    """Regex (or a list of them) over a text column, via Python's `re`.

    See the module docstring: `Series.str.contains` routes to RE2 in pandas 3,
    which rejects a `一-鿿` class outright.
    """
    if rx is None:
        return pd.Series(False, index=q.index)
    pats = rx if isinstance(rx, list) else [rx]
    if not pats:
        return pd.Series(False, index=q.index)
    text = q.astype(str)
    out = pd.Series(False, index=q.index)
    for p in pats:
        out |= text.map(lambda s, _p=p: bool(_p.search(s))).astype(bool)
    return out


def compile_extra(patterns: Sequence[str]) -> list[re.Pattern[str]]:
    """Each declared pattern compiled SEPARATELY. Never joined into one alternation.

    Joining them looks tidier and is wrong: a pattern carrying an inline global
    flag — `(?i)…`, which domain profiles legitimately write — is only valid at
    the START of an expression, so wrapping it as the second branch of an
    alternation raises `global flags not at the start of the expression`. The
    whole layer then fails to compile and the guard fails OPEN, which is the
    worst outcome available to it.

    Compiling separately also means a malformed pattern names itself.
    """
    out: list[re.Pattern[str]] = []
    for p in patterns:
        if not str(p).strip():
            continue
        try:
            out.append(re.compile(str(p), re.I))
        except re.error as exc:
            raise ValueError(f"quote-block pattern {p!r} does not compile: {exc}") from exc
    return out


#: A risk category declares which of the two hazards it is about. The profiles
#: already say so in their own policy text — `never quote`, `do not surface`,
#: `never relay` — and where they do not, the category's NAME does.
_NEVER_QUOTE_POLICY = re.compile(
    r"never\s+quote|do\s+not\s+quote|never\s+surface|never\s+reproduce|"
    r"never\s+relay|不得引用|不可引用", re.I)


def quote_patterns_from_profile(profile: Any) -> list[str]:
    """The domain profile's patterns that belong in the QUOTE guard — not all of them.

    Most risk categories are about how a system should ANSWER: never state a
    dose, never assert a combination is safe, route a crisis first. Quoting such
    a row harms no one, and blocking all of them empties the class cards —
    measured at 60%-90% of quotable rows lost on two verticals when a study
    tried blocking by risk-flagged class.

    The ones that belong here are the categories where reproducing the text IS
    the harm, and the profiles say which those are: their policy line reads
    "never quote", or their name is one of the universal categories. On a
    medical profile this selects seven patterns out of fifty-one.
    """
    out: list[str] = []
    for cat in (getattr(profile, "risk_categories", None) or []):
        blob = " ".join(str(getattr(cat, k, "") or "")
                        for k in ("name", "rationale", "policy"))
        if str(getattr(cat, "policy", "")).strip().lower() == "drop" \
                or _NEVER_QUOTE_POLICY.search(blob) \
                or _QUOTE_IS_THE_HARM.search(blob):
            out += [str(p) for p in (getattr(cat, "patterns", None) or []) if str(p).strip()]
    return out


def risk_row_indices(risk_screen: dict[str, Any] | None, n: int) -> set[int]:
    if not risk_screen:
        return set()
    return {int(i) for i in (risk_screen.get("flag_mask_indices") or []) if 0 <= int(i) < n}


def named_risk_strings(tree_naming: dict[str, Any] | None,
                       risk_screen: dict[str, Any] | None) -> set[str]:
    """Strings the run's own risk machinery singled out by name.

    Far narrower than blocking the classes those strings live in: a risk flag
    usually says "answering this badly is the hazard", not "this sentence may
    not be repeated". Blocking by class costs 60%–90% of quotable rows.
    """
    out: set[str] = set()
    tn = tree_naming or {}
    for f in ((tn.get("risk_report") or {}).get("findings") or []):
        out.update(str(x).strip() for x in (f.get("evidence") or []))
        out.update(m.strip() for m in _QUOTED.findall(str(f.get("rationale") or "")))
    for nmg in (tn.get("namings") or []):
        out.update(m.strip() for m in _QUOTED.findall(str(nmg.get("risk_reason") or "")))
    for fam in (tn.get("families_final") or []):
        out.update(m.strip() for m in _QUOTED.findall(str(fam.get("risk") or "")))
    for c in ((risk_screen or {}).get("categories") or []):
        if c.get("exemplar"):
            out.add(str(c["exemplar"]).strip())
        out.update(str(x).strip() for x in (c.get("samples") or []))
    return {x for x in out if x}


def never_quote_classes(taxonomy: dict[str, Any] | None,
                        declared: Iterable[str] = ()) -> set[str]:
    """L1 codes where reproducing a row's text is itself the harm.

    Derived from the run rather than from a per-corpus list: a class the
    architect marked risky, whose name / definition / risk note reads as one of
    the "quoting it reproduces it" categories. A software corpus's explicit
    image-editing class was found by exactly this route — 51 rows, of which a
    keyword regex caught 12, because each row changed a garment noun or a verb.
    """
    out = {str(x) for x in declared if str(x).strip()}
    tx = (taxonomy or {}).get("taxonomy", taxonomy) or {}
    for n in (tx.get("nodes") or []):
        if n.get("level") not in (1, None) or not n.get("code"):
            continue
        if not n.get("risk"):
            continue
        blob = " ".join(str(n.get(k) or "") for k in
                        ("name", "name_zh", "definition", "risk_note", "user_need"))
        if _QUOTE_IS_THE_HARM.search(blob):
            out.add(str(n["code"]))
    return out


def load_screened(path: Path | str | None) -> set[str]:
    """L7: strings a reader read one at a time. Absent file == empty layer."""
    if not path:
        return set()
    p = Path(path)
    if not p.exists():
        return set()
    data = json.loads(p.read_text(encoding="utf-8"))
    raw = data.get("strings", data) if isinstance(data, dict) else data
    return {str(x) for x in raw}


class QuoteGuard:
    """The seven layers, assembled once and applied to a frame."""

    def __init__(self, *, risk_screen: dict[str, Any] | None = None,
                 tree_naming: dict[str, Any] | None = None,
                 taxonomy: dict[str, Any] | None = None,
                 extra_patterns: Sequence[str] = (),
                 declared_never_quote: Iterable[str] = (),
                 screened_path: Path | str | None = None,
                 extra_hard_rules: dict[str, str] | None = None) -> None:
        self.risk_screen = risk_screen or {}
        self.named = named_risk_strings(tree_naming, risk_screen)
        self.never = never_quote_classes(taxonomy, declared_never_quote)
        self.extra = compile_extra(list(extra_patterns))
        self.screened = load_screened(screened_path)
        self.screened_path = Path(screened_path) if screened_path else None
        self.hard_rules = dict(HARD_RULES)
        for name, pat in (extra_hard_rules or {}).items():
            # A bare preset NAME switches the preset on; anything else is a
            # regex the caller supplied and owns.
            src = PRESET_HARD_RULES.get(str(pat).strip(), str(pat))
            self.hard_rules[str(name)] = re.compile(src)

    #: The bare `<2-3 chars>医生` form cannot live in `QUOTE_BLOCK` because it
    #: needs the generic-title exclusion, which is defined after it. It is
    #: always on for the same reason the rest of L2 is.
    #: `科?` because a speciality is written both ways — 儿科医生 and 急诊科医生
    #: are the same kind of phrase, and without it the second read as a name.
    _NAMED_PERSON = re.compile(
        rf"^(?!{_GENERIC_DOC}科?(?:医生|大夫)$)[一-鿿]{{2,3}}(?:医生|大夫)$")

    def layers(self, d: pd.DataFrame) -> dict[str, pd.Series]:
        """Each layer's mask, separately — so the report can say what blocked what."""
        q = d["query"].astype(str)
        idx = risk_row_indices(self.risk_screen, len(d))
        l1_col = d["td_l1"].astype(str) if "td_l1" in d.columns else pd.Series("", index=d.index)
        return {
            "L1_风控图层": pd.Series(d.index.isin(idx), index=d.index),
            "L2_硬规则": _mask(QUOTE_BLOCK, q) | _mask(self._NAMED_PERSON, q),
            "L3_共现": _mask(QUOTE_COOC[0], q) & _mask(QUOTE_COOC[1], q),
            "L4_风控点名串": q.isin(self.named),
            "L5_整类不引": l1_col.isin(self.never),
            "L6_声明正则": _mask(self.extra, q),
            "L7_逐串筛查": q.isin(self.screened),
        }

    def quotable(self, d: pd.DataFrame) -> pd.Series:
        blocked = pd.Series(False, index=d.index)
        for m in self.layers(d).values():
            blocked |= m
        return ~blocked

    def counts(self, d: pd.DataFrame) -> dict[str, int]:
        lay = self.layers(d)
        blocked = pd.Series(False, index=d.index)
        for m in lay.values():
            blocked |= m
        out = {k: int(v.sum()) for k, v in lay.items()}
        out["任一层拦下"] = int(blocked.sum())
        out["可引行数"] = int((~blocked).sum())
        out["总行数"] = int(len(d))
        return out

    # ------------------------------------------------------------ hard scan
    def hard_rule_scan(self, d: pd.DataFrame, printed: set[str],
                       snapshot_col: str = "snapshot") -> pd.DataFrame:
        """Independent pass: the three hard categories, against what was PRINTED.

        Its value is entirely in `printed` being the real thing. A version that
        globbed a directory that did not exist matched nothing and reported
        "0 printed" — silence read as a pass. The caller asserts `printed` is
        non-empty before this is trusted.
        """
        q = d["query"].astype(str)
        quotable = self.quotable(d).to_numpy()
        rows = []
        for name, rx in self.hard_rules.items():
            hit = _mask(rx, q).to_numpy()
            for i in d.index[hit]:
                s = q.loc[i]
                rows.append({"规则": name, "query": s,
                             "快照": d.at[i, snapshot_col] if snapshot_col in d.columns else "",
                             "已被护栏拦下": not bool(quotable[d.index.get_loc(i)]),
                             "出现在交付文档里": s in printed})
        if not rows:
            return pd.DataFrame(columns=["规则", "query", "快照", "已被护栏拦下", "出现在交付文档里"])
        return pd.DataFrame(rows).drop_duplicates(["规则", "query", "快照"])


def printed_strings(paths: Iterable[Path], corpus: set[str] | None = None, *,
                    tables: Iterable[Path] = ()) -> set[str]:
    """Every corpus row a document actually shows, including truncated ones.

    Two ways this under-counted, and both made the guarantee weaker than it read:

    * it looked only at quoted spans in the markdown, while the example rows
      also ship in `tables/examples_*.csv` and in the workbook built from them;
    * an exact-match lookup misses a row that was printed in truncated form, so
      a blocked string shown as its first 26 characters counted as "0 printed".

    So table cells are read too, and a corpus row counts as printed when any
    span or cell is a PREFIX of it.
    """
    spans: set[str] = set()
    for p in paths:
        try:
            spans |= set(SPAN.findall(Path(p).read_text(encoding="utf-8")))
        except (OSError, UnicodeDecodeError):
            continue
    for t in tables:
        try:
            df = pd.read_csv(t, encoding="utf-8-sig", keep_default_na=False)
        except Exception:  # noqa: BLE001
            continue
        for col in df.columns:
            if df[col].dtype == object or str(df[col].dtype).startswith("str"):
                spans |= {str(v) for v in df[col] if str(v).strip()}
    if corpus is None:
        return spans
    out = {s for s in spans if s in corpus}
    # Prefix match for the truncated case. Bounded to spans long enough to be a
    # real fragment, so a one-character cell does not sweep in the whole corpus.
    frags = [s for s in spans if s not in corpus and len(s) >= 8]
    if frags:
        for row in corpus:
            for f in frags:
                if row.startswith(f):
                    out.add(row)
                    break
    return out
