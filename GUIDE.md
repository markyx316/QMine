# QMine — what it does, and which path to take

A companion to `CLAUDE.md` (which is for whoever is *changing* the program).
This one is for whoever is *using* it, and for the moment someone arrives with
a pile of exports and asks what to do with them.

---

## 1. Four layers. Each works on its own.

| layer | command | costs | produces |
|---|---|---|---|
| **Prepare** | `qmine prepare A,B,C --out DIR` | nothing | one pooled corpus + every removed row, with the tier that removed it |
| **Mine** | `qmine run --input CORPUS --run-id ID` | **money and hours** | two independent label systems over every row, and the evidence behind them |
| **Compare** | `qmine compare ID` | nothing | per-class cross-snapshot tables, figures, a report, two workbooks |
| **Ask** | `qmine chat` · `qmine mcp` | nothing (a model reads, it does not mine) | answers about a finished study, with citations |

Only **Mine** spends. The other three are free and re-runnable, which is why the
recommended path front-loads them.

**Mine is the irreversible one in a second sense**: one run derives ONE taxonomy.
Two separate runs on two exports of the same vertical produced 20 and 19 classes
sharing **zero** codes — and the controlled version is worse: the *same* 20,000
rows, byte-identical, run twice, delivered 12 leaves and then 34. So everything
you want to compare has to go into the *same* run.

---

## 2. Someone arrives with datasets. Do this.

### Step 0 — is this one corpus or several?

If the files are snapshots of the same vertical that you want to **compare**
(two years, two interfaces, head vs tail), they go into ONE run. If they are
different verticals, they are separate runs and their labels will not be
comparable — that is not a limitation to work around, it is what the labels mean.

### Step 1 — look, before deciding anything

```bash
.venv/bin/qmine prepare data/a.xlsx,data/b.xlsx,data/c.xlsx --dry-run
```

Free, writes nothing. It measures each file and prints the plan it would run:
which column is the query, which is traffic, what each snapshot will be called,
and which cleaning operations apply. Read the **`⚠` concerns** — they are the
questions the data cannot answer for you.

### Step 2 — settle the comparison axis. This is the one that bites.

`time` means the snapshots are different periods of the same thing. `stratum`
means they are different samples of one period — different interfaces, different
sampling depths. The computation is identical; **the prose inverts**. Calling a
sampling difference a change over time is the more damaging error, so an
un-declared axis defaults to `stratum` and is reported as an assumption.

Cross-file overlap will not settle it for you. Measured: two random 1w samples
of the *same* surface a year apart share **0.02%** of their strings; two head
exports of that surface share **63.3%**; a *different* surface shares **0.00%**.
Overlap separates head from tail, not time from interface.

```bash
.venv/bin/qmine prepare data/a.xlsx,data/b.xlsx --out prepared \
    --axis stratum \
    --label '20250701=2025搜索' --label '20260914=健康管家' \
    --group '20250701=搜索'   --group '20260914=AI助手'
```

`--label` is what a reader sees. `--group` collapses snapshots into sides
(interface, product, sampling method) for the two-sided tables — **either every
snapshot gets a group or none does**, because a partial grouping drops the
unassigned ones out of an `n` that still gets reported.

### Step 3 — pick a domain profile, or accept that you have none

`--domain` selects `configs/domains/*.yaml`: the phrasing seeds, the risk
categories, the expected family range. With none, the run uses the generic
profile and a scout agent guesses the vertical — which is honest but weaker.
Writing a profile for a new vertical is a real piece of work and it pays for
itself across every later run in that vertical.

### Step 4 — see what it will cost

```bash
.venv/bin/qmine models                 # routing plan + estimate. Spends nothing.
```

### Step 5 — mine

```bash
.venv/bin/qmine run --input prepared/prepared_corpus.parquet \
    --run-id med-2026a --domain medical_zh_v2 --config configs/pool5_med.yaml --fast
```

`--fast` keeps the analysis at full size and drops only the second-opinion
layer, so **kappa is absent, not 1.0**. It is the right default for a corpus you
are exploring; use full when the run is the deliverable.

Or do steps 1–5 in one command — the prepared corpus lands inside the run
directory as evidence:

```bash
.venv/bin/qmine run --prepare --input data/a.xlsx,data/b.xlsx --run-id med-2026a --fast
```

### Step 6 — the comparison is already there

`p10c` builds it at the end of any pooled run. To rebuild it — after renaming a
snapshot, adding a screened quote list, or fixing anything — re-run it freely:

```bash
.venv/bin/qmine compare med-2026a --axis stratum
```

### Step 7 — check the run, against a control

```bash
.venv/bin/python tools/verify_run.py runs/med-2026a/gen01 runs/OLD-BROKEN/gen01
```

**Always pass the control.** A harness that passes on one run proves nothing
about the harness.

### Step 8 — ask it things

```bash
.venv/bin/qmine chat     # terminal
make chat-setup          # once: install the chat web app and wire QMine into it
make chat                # thereafter: open it
```

---

## 3. What `qmine chat` actually does — and what it does not

It is a **router with a confirmation gate**, not a second brain.

It turns what you say into a choice from a fixed list of twelve actions, fills
in typed parameters, prints the equivalent command line, and runs it. Free and
reversible steps happen immediately; anything that spends money or writes a file
stops and asks first. A model understands the sentence; a program decides what
runs.

**It can**: measure your files and tell you what they are; propose a pooling and
cleaning plan and explain each step; estimate a run's cost; show a run's status
and which gates did not pass; build or rebuild a comparison; list runs; explain
a concept.

**It will not**: start a paid run because you showed it some files. Escalation
needs you to ask *and then* approve at the prompt. That is pinned by a test.

**It does not**: read your results and interpret them. That is what `qmine mcp`
adds — see below. `chat` drives the pipeline; `mcp` also answers questions about
what came out of it.

**Without a model** (`--keywords`) it still works, on keyword routing. That path
is not a dead fallback: it is what the tests exercise, and its answer is handed
to the model as a baseline so a disagreement has to be earned.

---

## 4. `qmine mcp` — the program as tools for a chat app

`qmine mcp` speaks the Model Context Protocol on stdin/stdout. Any harness that
bridges MCP tools gets the whole program — Claude Desktop, Cursor, VS Code, or
DeepSeek Harness, whose config is in `integrations/dsh/` (verified end to end
against dsh 0.1.5-rc.2). **Nothing is forked**: the harness is installed
unmodified and supplies the chat app, the model adapter, the session log and the
agent loop.

**Two model layers that never meet.** The chat model — one model, chosen in the
harness, and not restricted to DeepSeek — reads your sentence and picks a tool.
QMine's own router is untouched: when a run starts it is a subprocess reading
`QMine/.env` and matching each AGENT ROLE (architect, annotator_a, referee,
namer, risk_sentinel…) to a model exactly as `qmine run` always has. The chat
model never annotates a row; the annotator never learns there is a conversation.
Set the chat model for conversation quality and `.env` for mining quality.

The half that only exists after mining: **ask questions about a finished study**.
The report is ~190KB; `qmine_findings` returns its measured headline in about
3,700 characters, every distance beside its noise ceiling. `qmine_class` answers
"why is that missing from the assistant" with an expected count, a P(0) and a
verdict instead of a bare zero. `qmine_examples` returns real rows that have been
through the quote guard, and reports a count — never a blank — when a class has
rows that may not be reproduced.

Tools are tiered **read / write / spend**, and `spend` is refused unless
`QMINE_MCP_ALLOW_SPEND=1` was set outside the conversation.

**The assistant already knows this project.** It does not discover QMine by
listing your directories — `make chat` installs an agent preset carrying a
persona (what QMine is, which tool answers which question, and the ten ways to
state a correct number so that the reader draws the wrong conclusion) plus five
loadable skills: read a study, compare snapshots, prepare datasets, start a run,
explain the method. Edit them in `integrations/dsh/presets/qmine/persona.md` and
`integrations/dsh/skills/`; `make chat` reinstalls them every launch. `qmine
doctor` has a `dsh preset` row that says what is installed.

Two channels that look usable and are not: an MCP server's `instructions` field
never reaches the model in dsh (`dsh-mcp-client` bridges Tools only), and the web
profile disables workspace instructions and skills at the host level, re-enabling
them per preset. If you are wiring a different MCP client, `qmine mcp` still
advertises `instructions` — Claude Desktop and Cursor do read it.

**Watching a run from the chat.** `qmine_start_run` launches detached and
returns a run id at once, because a run takes hours and a tool call cannot.
After that, `qmine_status` reports the current phase, every gate so far, model
calls and tokens spent so far, which artifacts exist, open findings, and the
path to `runs/<id>/dashboard.html` — a live page you can keep open in a browser
tab. `qmine_partial` reads one intermediate artifact mid-run (the corpus audit
after p1, the intent classes after p2a, the families and leaves after p7), so
"what has it decided so far?" has a real answer instead of "wait".

Two limits, stated because they change how you read it. The chat model learns
the state **only when it polls** — MCP has a progress-notification facility but
`dsh-mcp-client` does not consume it, so nothing is pushed. And every reading is
a snapshot stamped `as_of`; a run with no `run_summary.json` has not finished,
and the tools refuse to describe its results.

---

## 5. The blind product-layer audit — what it is and when you need it

**The problem.** An AI-assistant export is not a list of queries. Mixed in with
what people typed is what the *product* printed: triage-form answer chips,
feature buttons, pushed question cards, doctor cards, empty acknowledgements.
Mine those and you are measuring the interface's own vocabulary and calling it
user intent — and it is not a rounding error. On one health-assistant export,
**86% of rows carried a product wrapper**, and the product layer was a
double-digit share of what remained.

**Why rules are not enough.** A word list catches the phrasings someone thought
of. The hard case is an answer chip that *describes a symptom* — because a
triage form's options describe symptoms by design, and "white discharge,
yellowish, fishy odour" is a form option, not a sentence anyone typed.

**What the audit is.** Three independent readings of each ambiguous string, by
the same model under three deliberately different framings, against a written
codebook:

| lens | the question it asks |
|---|---|
| `origin` | who most plausibly produced this string — a person at a keyboard, or the interface? |
| `standalone` | lift it out of context: does it still stand on its own? |
| `form` | describe only its surface — length, grammar, person, question markers — and infer nothing |

Six codes: `A` answer option, `F` feature or facet button, `P` pushed question,
`D` doctor card, `C` no content, `U` user-typed. **Ties go to `U`**: keeping a
product string adds noise, deleting a real need removes it from the analysis
where nobody can see that it is gone.

**How it is used.** Only unanimity removes a row. A split reading keeps it. The
rules alone may remove only three narrow categories.

**What it cost and bought, measured.** On the 健康管家 corpus: Fleiss κ **0.838**,
three votes identical on **90.5%** of strings. The codebook had to be
*calibrated*, not trusted — re-labelling 300 previously-audited strings agreed
with the original instrument on only **78.7%** of keep/remove decisions, and the
error was entirely one-directional and concentrated in code `A`. Fixing the
*definition* (naming the answer-chip signature: fallback chips like 都没有/不清楚,
comma-enumerated findings with no connective grammar, bare test values) took it
to **91.3%**.

**Is it in the program?** No, and deliberately. It needs a human-written codebook
and a calibration set, and a mechanical version that looked similar would be a
*looser instrument* — which, compared against another snapshot, shows up as a
difference between snapshots rather than between instruments. `qmine prepare`
does the mechanical part (wrapper stripping, content-free rules, collision
merging) and is honest about the gap: on one corpus it retained 98.5% where the
audited pipeline retained 98.2%, and **those 38 rows are exactly this layer**.

**When you need it.** When an assistant/chatbot export is one of your snapshots
and the product layer is more than a few percent — `qmine prepare --dry-run`
tells you, because a high-coverage prefix and a pile of short repeated strings
are what it looks like from outside.

---

## 6. The five things people get wrong

1. **Diffing two runs' labels.** They share no codes. Pool into one run.
2. **Reading a `stratum` comparison as change over time.** It is not a timeline.
3. **Reading a bare zero as absence.** Zero of 950 rows is consistent with a
   share up to 0.39%. Quote the verdict column.
4. **Reading a `fast` run's missing kappa as a perfect one.** It is absent.
   `fast_skipped` lists exactly what was not checked.
5. **Reading a `PASS` count from `verify_run.py` alone.** SKIP, N/A and ERROR are
   not passes, and a count across modes compares different things.
