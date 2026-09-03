<!-- Moved out of README.md for the same reason. The README keeps the part that
     tells a reader what to DO with the output; the survey of why intent
     taxonomies matter lives here. -->

## What it is for

An intent taxonomy is not a report you file. It is a control surface, an
analytics substrate, and — increasingly — a compliance artifact.

**It decides which retrieval path runs.** This was the point from the beginning:
Broder introduced the navigational/informational/transactional split because "each
type is best satisfied by very different results" ([SIGIR Forum 2002][broder]),
and Rose & Levinson spelled out the mechanism — advice-seeking queries lean on
usage- and connectivity-based relevance, open-ended research on term-frequency
signals ([WWW 2004][rose]). It is still how it works. Google's rater guidelines
make intent **step one** of relevance measurement, before any rating is given
([SQRG][sqrg]). And LinkedIn's production query-understanding model routes every
query into four types — extract structured constraints, rewrite with member
context, fall back to high-recall lexical retrieval when intent is unclear, or
block — cutting request failures 17.13% in an online A/B, with relevance up 57% on
navigational and 75% on exploratory queries ([KDD '26][linkedin]). Note the shape:
four classes, one of them a **risk** class, one an explicit **unclear-intent
fallback**.

**A borrowed taxonomy collapses, measurably.** Three studies of the *same* three
Broder classes report navigational at 20%, at 11.7–15.3%, and at ~10%, and
informational at 48%, 61–63%, and >80% ([Broder][broder]; [Rose &
Levinson][rose]; [Jansen et al. 2008][jansen]) — a factor of four on identical
labels. Worse, when Rose & Levinson's taxonomy was applied to real Bing question
queries, **86.7% of them fell into a single category**; a purpose-built 16-class
taxonomy spread them out *and* achieved higher inter-assessor agreement
([Cambazoglu et al., CHIIR 2021][cambazoglu]). Finer was not harder to agree on.
That is the case for mining the taxonomy from the corpus you actually have, and
it is the same lesson this repo learned the expensive way when K12 thresholds were
imported into gates meant to be corpus-independent.

**Once labelled, the log becomes an instrument.** Microsoft Research applied a
product-intent classifier to Bing logs and could then report, per class, success
rate (navigational 77.28%, comparison 61.31%), volume share, relative user effort
(navigational costs 2.63× comparison), and session position — 56% of transactional
queries are preceded by a comparison query ([Rao et al.][rao]). Applying the same
taxonomy to corpora a decade apart measured a demand shift: 15–17% of web queries
are product-related, against 5–7% before. Two of their five classes — Comparison
and Support — appear in no prior taxonomy and came from the data, which is the
argument for this pipeline's bottom-up route in one sentence.

Every query in a QMine run carries both labels, so intent volume is a `group by`
and the demand/supply gap is a diff against your catalogue or content inventory
([Goswami et al., WWW '19][goswami] — abstract-level only; the method detail is
unverified).

**It is the guardrail on query rewriting.** eBay's null/low-result rewriter treats
the shopping-category distribution as the check that a rewrite has not destroyed
intent: "a sufficiently strong signal that the alternative query distorts user's
search intent if the inferred level-2 categories are changed", with top-1 rewrites
preserving category 71.5% of the time ([SIGIR eCom 2017][ebay] — offline on 3,000
sampled queries; no online A/B, despite what secondary sources say). Zero-results
rate is a first-class tracked KPI at least at [Wikimedia][wmf].

**The risk class is not optional, and for this corpus it is regulatory.** NAVER
ships a 12-class sensitive-query taxonomy as a runtime classifier in front of
generation, grouped Legal / Ethical / Service-sensitive, with **Age-restricted
contents** as a first-class category ([arXiv 2404.08672][naver]). Their data also
shows why a taxonomy is re-run rather than written once: self-harm queries average
1.6% of sensitive traffic and hit **17.2% in a single day** against a news event.
For a Chinese K12 corpus this is law, not practice — the
《[未成年人网络保护条例][minors]》 took effect 1 January 2024, and the CAC's
《[移动互联网未成年人模式建设指南][cac]》 (Nov 2024) prescribes five age bands
(<3, 3–8, 8–12, 12–16, 16–18) each with its own content prescriptions.

QMine's taxonomy carries an explicit risk class and the screen runs on **every
row**, not the sample. On `live42` it hit 1,499 rows across 30 leaves, and the leaf
catalogue reports each leaf's own count — because the blind namer's risk flag sees
a *sample* and the screen sees *everything*, and conflating them hides most of the
exposure. What a mined risk class is **not** is a shipped safety system: NAVER's
live harm-class precision was 74.7% with ~50 queries reviewed by humans daily.

**It replaces the most expensive part of an annotation programme.** The published
cost of building one taxonomy by hand: 1,000 queries, five assessors, **50–70
hours each** — roughly 250–350 person-hours of labelling alone — through a nine-
phase loop of draft, pilot, feedback, guideline revision, relabel, adjudicate
([Cambazoglu et al.][cambazoglu]). At product scale it never ends: Google works
with ~16,000 raters against a ~180-page guideline revised roughly twice a year
([SQRG][sqrg]), NIST budgets six contractors for two to four weeks per track
([Soboroff][nist]), and LinkedIn labelled ~14K production queries while noting that
"query understanding labels are expensive: correctness depends not only on semantic
plausibility but also on product policy" ([KDD '26][linkedin]). And the labels do
not amortise — "whenever a new vertical is introduced, a costly new set of
editorial data must be gathered" ([Arguello et al., SIGIR 2010][arguello]).

The most reusable thing QMine emits is therefore not the labels but
`标注规范与裁定规则.md`: the labeling guide plus 139 adjudication rules, each with
its trigger, target class, rationale and worked examples, marked by whether the
architect predicted it or the referee earned it against a real disagreement. That
is the artifact those 250–350 hours produce.

**Migrating to another domain.** Nothing above is K12-specific. Point it at a
different corpus and override the defaults; domain profiles carry the phrasing
seeds and risk vocabulary, and the pipeline gates whether a corpus has the
reference labels it would otherwise silently do without.

[amazon]: https://www.amazon.science/blog/from-structured-search-to-learning-to-rank-and-retrieve
[broder]: https://sigir.org/files/forum/F2002/broder.pdf
[rose]: https://www.ambuehler.ethz.ch/CDstore/www2004/docs/1p13.pdf
[jansen]: https://eprints.qut.edu.au/30951/
[cambazoglu]: https://marksanderson.org/files/papers/CHIIR21b.pdf
[sqrg]: https://services.google.com/fh/files/misc/hsw-sqrg.pdf
[linkedin]: https://arxiv.org/abs/2605.27441
[rao]: https://arxiv.org/pdf/2005.08591
[goswami]: https://dl.acm.org/doi/10.1145/3308560.3316605
[ebay]: https://ceur-ws.org/Vol-2311/paper_8.pdf
[wmf]: https://www.mediawiki.org/wiki/Wikimedia_Discovery/FAQ
[naver]: https://arxiv.org/abs/2404.08672
[minors]: https://www.moj.gov.cn/pub/sfbgw/zcjd/202310/t20231024_488321.html
[cac]: https://www.cac.gov.cn/2024-11/15/c_1733364304749288.htm
[nist]: https://arxiv.org/abs/2409.15133
[arguello]: https://ils.unc.edu/~jarguell/ArguelloSIGIR10.pdf

---
