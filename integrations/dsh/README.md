# QMine inside DeepSeek Harness

For whoever is wiring the chat front door up, or debugging one that came up
empty. The one-command path is `make chat-setup && make chat`
([README](../../README.md#the-chat-front-door)); this file is everything that
command hides.

**Nothing here is a fork.** dsh is installed unmodified from npm; QMine adds a
Python MCP server (`qmine mcp`) and one config entry. dsh supplies the chat web
app, the model adapter, the session log, context compaction and the agent loop.
QMine supplies tools.

Verified end to end against **dsh 0.1.5-rc.2**: the patch composes, the harness
boots, the QMine server runs as a child process of `dsh`, and `mcp-qmine` shows
**Enabled** under Settings → Plugins → Plugin list.

---

## The thing to understand first: there are TWO model layers, and they do not meet

```
   you ─── chat ──▶ ┌───────────────────── dsh (web app, port 3080) ───────────────────┐
                    │  CHAT MODEL — one model, picked in Settings → Models             │
                    │  anthropic · openai · deepseek · moonshotai(Kimi) · zai(GLM) ·   │
                    │  bedrock · azure · any OpenAI-compatible endpoint                │
                    │  Its whole job: read your sentence, choose a tool, write the     │
                    │  answer from what the tool returned.                             │
                    └───────────────┬─────────────────────────────────────────────────┘
                                    │  MCP over stdio  (mcp__qmine__*)
                    ┌───────────────▼─────────────────────────────────────────────────┐
                    │  qmine mcp — 18 tools                                            │
                    │  reads finished runs · plans corpora · proposes runs             │
                    └───────────────┬─────────────────────────────────────────────────┘
                                    │  subprocess, only when you ask for a run
                    ┌───────────────▼─────────────────────────────────────────────────┐
                    │  THE MINING RUN — unchanged, exactly as it has always worked     │
                    │  QMine's OWN router, per AGENT ROLE, from QMine/.env:            │
                    │  architect · annotator_a · annotator_b · referee · namer ·       │
                    │  tree_auditor · risk_sentinel · adversary · observer · …         │
                    │  each matched to a model by `llm/requirements.py` across         │
                    │  DeepSeek · Zhipu · Qwen · Kimi · Anthropic · …                  │
                    └─────────────────────────────────────────────────────────────────┘
```

**dsh does not route QMine's agents, and QMine's router does not serve the
chat.** The chat model never sees an annotator prompt; the annotator never
learns there is a conversation. A run launched from the chat is a subprocess
that reads `QMine/.env` and routes exactly as `qmine run` always has — same
`qmine models` estimate, same per-role matching, same provider failover.

So: **yes, your program still runs as before.** The chat app is a new front
door, not a new engine. Two consequences worth stating:

- Your multi-provider routing is untouched, and dsh's model choice has no effect
  on which model annotates a row. Set the chat model for conversation quality;
  set `QMine/.env` and `llm/requirements.py` for mining quality.
- The chat model is **not** DeepSeek-only. dsh ships adapters for `anthropic`,
  `openai`, `deepseek`, `moonshotai`, `zai`, Bedrock and Azure, plus any
  OpenAI-compatible endpoint via a custom provider. Keys live in
  `$DSH_HOME/.credentials.yaml`, not in QMine.

---

## Setup

```bash
# 1. dsh (Node 20+). Nothing is forked; this is upstream npm.
mkdir -p ~/dsh && cd ~/dsh
npm install @deepseek-ai/dsh

# 2. the QMine entry, with this checkout's absolute paths filled in
cd /path/to/QMine
.venv/bin/pip install -e '.[mcp]'            # the MCP SDK
.venv/bin/qmine mcp --print-dsh-config > ~/dsh/qmine.patch.yml

# 3. the agent preset — what the assistant knows before it is asked anything
.venv/bin/qmine mcp --install-preset ~/dsh/home/.agent-presets

# 4. boot with it. DSH_HOME is where step 3 just wrote the preset; without it
#    dsh looks in ~/.dsh, finds nothing, and silently serves the coding agent.
cd ~/dsh && DSH_HOME=~/dsh/home ./node_modules/.bin/dsh web --patch ./qmine.patch.yml
```

`make chat-setup` then `make chat`, from the QMine checkout, does all four and
rewrites steps 2 and 3 on every launch so neither can go stale. `make chat-stop`
stops it. **A preset mounts once per process**: an edited persona or skill reaches
the model only after a restart, and a second `make chat` on a busy port is refused
with one line instead of node's `EADDRINUSE` trace.

To make it permanent instead of passing `--patch` each time, put the same
content in `$DSH_HOME/profiles/web/cordis.patch.yml` (it ships as `[]`).

### Check it took — do not assume

```bash
dsh --profile web --patch ./qmine.patch.yml --dump-config | grep mcp-qmine
```

and in the UI: **Settings → Plugins → Plugin list → search `qmine`** → it should
read `mcp-client / mcp-qmine  ● Enabled` under *Global plugins*.

### The one syntax that bites

`cordis.patch.yml` is a list of **loader patches**, not a list of plugins. A bare
`- id: … name: … config: …` is read as an *override of an entry that already
exists*, and dsh answers `patch: entry "mcp-qmine" not found`, warns, skips —
and starts a harness with **no QMine tools in it**. Adding a plugin needs a
patch whose `insert` holds the entry and which carries no `id`:

```yaml
- insert:
    - id: mcp-qmine
      name: '@deepseek-ai/dsh-mcp-client'
      config: { ... }
```

`qmine mcp --print-dsh-config` emits this correctly; a test pins it.

### Two settings that are not the defaults

**`failOnStartupError: true`** — the default `false` leaves the harness up with
zero QMine tools and an error only in the log, so the assistant answers from
memory about a program it cannot reach.

**`toolCallTimeoutMs: 900000`** — the default 60s is shorter than a comparison
rebuild on a large corpus.

---

## What the assistant knows before you ask it anything

`make chat` also installs QMine's **agent preset** into
`$DSH_HOME/.agent-presets/qmine/` and points `agent-presets.default` at it, so a
new session opens as **QMine 研究助手** rather than the shipped coding agent. By
hand:

```bash
.venv/bin/qmine mcp --install-preset "$DSH_HOME/.agent-presets"
```

It is authored here, not generated from nothing:

| file | what it is |
|---|---|
| `presets/qmine/persona.md` | the always-on brief — what QMine is, the four layers, tool routing, and the ten ways to state a correct number wrongly |
| `presets/qmine/agent.cordis.yml` | the composition; derived from the shipped `standard` preset by swapping exactly two rows, the persona and the skill roots |
| `presets/qmine/preset.yml` | what the session picker shows |
| `skills/*/SKILL.md` | five loadable procedures; a one-line summary is always in context, the body only when the model loads it |

### Why a preset, and not a patch

Three channels look like they would carry a project brief. Two of them do not.

1. **The MCP server's `instructions` field.** `dsh-mcp-client` bridges the
   **Tools** capability and nothing else — the string "instructions" does not
   appear anywhere in the package. Anything written there is dead prose in dsh.
   (Claude Desktop and Cursor do read it, which is why `qmine mcp` still sends
   it.)
2. **A host-level patch** re-enabling `agent-instructions`, `skill-filesystem` or
   `tool-skill`. The web profile ships those — and `persona`'s host-plane twin —
   `disabled: true` and re-enables them per agent preset. A host patch would
   apply to the coding preset too.
3. **An agent preset.** This one. It is the only per-agent seam.

### Three things that bite when authoring one

- **A config override REPLACES; it does not merge.** `- id: system-prompt` with a
  single key silently drops every other key that entry had. Restate the whole
  config.
- **`customSkillDirs` resolves against the harness process's cwd**, not your
  checkout, so a relative root finds nothing — and an absent skill root is valid
  empty state that warns nobody. `--install-preset` writes absolute paths and
  asserts no token survived.
- **Skill discovery is one level deep.** `<root>/<name>/SKILL.md` only; a nested
  bundle is not found and not reported.

Check it took, the same way you check the tools:

```bash
dsh --profile web --patch ./qmine.patch.yml --dump-config | grep -A3 'id: agent-presets'
```

## What the model may set in motion

**`QMINE_MCP_ALLOW_SPEND` has three postures.** Unset or `0` refuses and hands
back the command. `ask` — what `make chat` sets — lets a run start, but only
after `qmine_preflight` passes AND an approval gate was consulted for it: the
preset installs a `PreToolUse` hook that runs the preflight and answers `ask`,
and dsh then holds the call at its own dialog showing the cost, which only a
person can answer. `1` allows it outright, for unattended use.

**A missing gate fails CLOSED.** A hook that fails emits no decision, and a
decisionless hook does not block anything — dsh falls through to allow. So the
hook issues a one-shot ticket and the server refuses in `ask` mode without one.
That is a guard against the gate being ABSENT; it is not a credential, and the
consent itself is the dialog, which no process here can answer.

| tier | tools | behaviour |
|---|---|---|
| read | `findings` `class` `examples` `table` `tables` `document` `glossary` `overview` `status` `partial` `list_runs` `inspect_inputs` `plan_corpus` `estimate_cost` `capabilities` | run freely |
| write | `prepare_corpus` `build_comparison` | allowed, only inside permitted roots; no model calls, so re-running is safe |
| spend | `start_run` | **refused** — returns the exact command for a person |

`QMINE_MCP_ALLOW_SPEND=1` in the launching environment turns spending on. That is
deliberately something you do **outside** the conversation: a token returned
through a tool result is a token the model can read and repeat.

**dsh has its own permission layer too** (Settings → General → Permission, and
`dsh-user-approval` / `dsh-permission-presets`), which can require approval per
tool call. The two are complementary — QMine's gate holds whatever harness is in
front of it, dsh's gate holds whatever server is behind it. Use both.

A started run is **detached**: `qmine_start_run` returns immediately with a run
id and a pid, because a run takes hours and a tool call cannot. Poll it with
`qmine_status`, which reports the phase, the log tail, and refuses to describe
results for a run that has not written a summary.

---

## Will this keep working?

dsh is **0.1.5-rc.2** and says so on first launch: *"DeepSeek Harness 0.1 remains
in testing for Harness developers… core plugins and foundational APIs will
continue to evolve rapidly over the coming months."* Treat the config as
something that may need re-generating after an upgrade — which is why it is
generated rather than hand-written.

**The insurance is that the MCP server is harness-agnostic.** `qmine mcp` speaks
plain MCP over stdio and works unchanged with any MCP client — Claude Desktop,
Cursor, VS Code, Codex, a custom app. If dsh changes or you prefer another
front end, you change one config file, not any QMine code. The durable asset is
the tool surface; the harness is swappable.

## Known limits

- `dsh-mcp-client` bridges the **Tools** capability only. Resources and Prompts
  are unsupported, so everything here is a tool, including the glossary.
- Tool definitions cost tokens in every model request — 18 tools is a real but
  modest standing cost.
- A run started from the chat outlives the conversation. That is correct, and it
  means "did it finish?" is always a `qmine_status` call, never an assumption.
