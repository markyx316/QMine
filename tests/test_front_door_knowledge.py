"""The chat front door's standing knowledge: the agent preset, its persona, its skills.

WRITTEN AFTER THE FRONT DOOR SHIPPED WITHOUT ANY. The first live session against
the harness spent its opening turns listing directories to work out what the
project was, because the only channel carrying project knowledge — the MCP
server's `instructions` field — is never read: `dsh-mcp-client` bridges Tools and
nothing else, and the word "instructions" does not occur in it. The web profile
additionally ships `persona`'s host-plane twin, `agent-instructions`,
`skill-filesystem` and `tool-skill` all `disabled: true`, re-enabling them per
agent PRESET. So a preset is the only place this deployment can put what the
assistant should know before anyone asks, and these tests pin that it arrives
intact.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from qmine.cli import DSH_PRESET_DIR, DSH_SKILLS_DIR, _dsh_config, _dsh_preset


class _JsTolerantLoader(yaml.SafeLoader):
    """`agent.cordis.yml` carries `!!js` platform gates that dsh evaluates."""


_JsTolerantLoader.add_constructor(
    "tag:yaml.org,2002:js", lambda loader, node: "<js-expression>")


def _composition() -> list[dict]:
    return yaml.load(_dsh_preset()["agent.cordis.yml"], Loader=_JsTolerantLoader)


def _row(ident: str) -> dict:
    return next(r for r in _composition() if r.get("id") == ident)


def _skill_files() -> list[Path]:
    return sorted(DSH_SKILLS_DIR.glob("*/SKILL.md"))


# ------------------------------------------------------------- the preset

def test_the_persona_survives_the_yaml_block_scalar_intact():
    """The authored prose must reach the model byte for byte.

    The persona is embedded as a `prefix: |-` block scalar, so every line has to
    be indented to the block's level. A line left at column 0 ENDS the scalar and
    is read as a new mapping key: the assistant then starts with whatever
    paragraph happened to come first and silently loses the rest — including the
    rules that stop it quoting a corpus row. Nothing downstream would complain.
    """
    authored = (DSH_PRESET_DIR / "persona.md").read_text(encoding="utf-8").rstrip()
    assert _row("persona")["config"]["prefix"] == authored


def test_no_substitution_token_reaches_the_installed_preset():
    """A surviving token is a path the harness resolves against its own cwd."""
    rendered = _dsh_preset()
    for name, body in rendered.items():
        assert "__QMINE_" not in body, f"{name} still carries a template token"


def test_the_skills_root_is_absolute_and_exists():
    """`customSkillDirs` is resolved against the HARNESS process's cwd.

    That is the harness directory, never this checkout, so a relative root finds
    nothing and the catalogue comes up empty — with no error, because an absent
    skill root is valid empty state.
    """
    roots = _row("skill-filesystem")["config"]["customSkillDirs"]
    assert roots, "the preset declares no skill root"
    for root in roots:
        assert Path(root).is_absolute(), f"{root} is relative"
        assert Path(root).is_dir(), f"{root} does not exist"


@pytest.mark.parametrize("ident", ["persona", "agent-instructions",
                                   "skill-filesystem", "tool-skill"])
def test_every_knowledge_channel_is_mounted(ident: str):
    """Each of these is `disabled: true` at the host level in the web profile.

    Dropping any one of them costs a specific channel and nothing says so: the
    persona is the standing brief, `agent-instructions` reads the workspace's own
    `AGENTS.md`, and the two skill rows are the catalogue and its loader.
    """
    assert _row(ident).get("disabled") is not True


def test_the_patch_opens_new_sessions_into_the_qmine_preset():
    """Installed but not default is the same as not installed.

    A bare entry OVERRIDES an existing id and the override REPLACES the config
    object rather than merging into it, so `default` has to be restated here in
    full — and `agent-presets` has to still be found, or dsh warns and skips.
    """
    patch = yaml.safe_load(_dsh_config("runs"))
    presets = [e for e in patch if e.get("id") == "agent-presets"]
    assert presets, "the patch never names the preset roster"
    assert presets[0]["config"]["default"] == "qmine"
    assert any("insert" in e for e in patch), "the patch stopped inserting the MCP server"


# -------------------------------------------------------------- the skills

def test_every_skill_is_discoverable_where_the_provider_looks():
    """Discovery is ONE level deep: `<root>/<name>/SKILL.md` or `<root>/<name>.md`.

    A nested bundle is not found, and a skill that is not found is not reported —
    it is simply absent from the catalogue. The glob therefore has to be RECURSIVE
    even though the provider's is not: a test that only looks where the provider
    looks cannot see the file the provider missed.
    """
    assert _skill_files(), "no skills at all"
    for path in DSH_SKILLS_DIR.rglob("SKILL.md"):
        assert path.parent.parent == DSH_SKILLS_DIR, (
            f"{path} is nested too deep for the provider to discover")


@pytest.mark.parametrize("path", _skill_files(), ids=lambda p: p.parent.name)
def test_every_skill_has_frontmatter_the_provider_accepts(path: Path):
    """`name` must be kebab-case and `description` must exist, or the skill is
    skipped WITH A WARNING THE MODEL NEVER SEES — the catalogue cannot tell an
    absent skill from an invalid one."""
    text = path.read_text(encoding="utf-8")
    assert text.startswith("---\n"), f"{path} has no frontmatter"
    meta = yaml.safe_load(text.split("---", 2)[1])
    assert re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", meta["name"]), meta["name"]
    assert meta["name"] == path.parent.name, "name must match its directory"
    assert meta.get("description", "").strip(), "description is required"


def test_no_standing_text_names_a_tool_that_does_not_exist():
    """A hint naming a tool nobody serves sends the model to a dead end.

    This already happened once on this surface: a hint pointed at `qmine_progress`
    — the module — instead of `qmine_status`, the tool. The persona and the skills
    name tools far more often than the tool descriptions do.
    """
    from qmine.mcp.server import QMineServer

    real = {t["name"] for t in QMineServer("runs").specs}
    sources = [DSH_PRESET_DIR / "persona.md", *_skill_files()]
    for path in sources:
        # The model sees every tool server-qualified as `mcp__qmine__<raw>`, so
        # the prefix comes off before the raw names are compared. A trailing
        # underscore is a wildcard the prose writes (`qmine_*`), not a name.
        text = path.read_text(encoding="utf-8").replace("mcp__qmine__", "")
        named = {n for n in re.findall(r"qmine_[a-z_]+", text) if not n.endswith("_")}
        unknown = named - real
        assert named, f"{path.name} names no tool at all"
        assert not unknown, f"{path.name} names {sorted(unknown)}"


# ------------------------------------------------- what the rules must say

def test_the_standing_rules_forbid_illustrating_what_was_withheld():
    """A caveat that carries its own counterexample defeats the guard.

    Measured live: asked for examples from a medical study, the assistant
    correctly declined to quote named doctors — and then printed one to show what
    it had declined to quote. The guard had passed that row (it is `机构 + 姓名`,
    which no shipped pattern matches), so nothing mechanical stood in the way; the
    only control is the instruction, and it has to say BOTH that a never-quote
    class stays unquoted whatever the guard returned AND that the withheld thing
    is never illustrated.
    """
    persona = (DSH_PRESET_DIR / "persona.md").read_text(encoding="utf-8").lower()
    # IN THE PERSONA SPECIFICALLY. A skill is loaded on demand; by the time the
    # model decides whether to illustrate what it withheld, it is already writing
    # the sentence. Only the always-on channel arrives in time.
    assert "illustrate what you are withholding" in persona, \
        "the persona does not forbid showing a specimen of the withheld class"
    assert "guard passing a row does not make it quotable" in persona, \
        "the persona does not say a never-quote class survives the guard passing it"


@pytest.mark.parametrize("trap", [
    "noise ceiling",     # a distance below it is not a finding
    "stratum",           # not a timeline
    "fast",              # kappa is absent, not perfect
    "offline",           # a stand-in wrote it
    "detectability",     # a zero never ships alone
])
def test_the_persona_carries_every_trap_that_makes_an_answer_wrong(trap: str):
    """These are not stylistic preferences; each is a way to state a correct
    number so that the reader draws the opposite conclusion. They belong in the
    ALWAYS-ON channel, because by the time a skill could be loaded the sentence
    is already being written."""
    persona = (DSH_PRESET_DIR / "persona.md").read_text(encoding="utf-8").lower()
    assert trap in persona


def test_no_standing_text_carries_a_string_the_quote_guard_blocks():
    """The persona and the skills are shipped prose that the model may echo.

    An example written into them has never been through the guard, and a rule
    illustrated with a real blocked row would hand the model a licence to repeat
    it.
    """
    from qmine.pooled.guards import HARD_RULES, PRESET_HARD_RULES

    patterns = {**HARD_RULES, **PRESET_HARD_RULES}
    for path in [DSH_PRESET_DIR / "persona.md", *_skill_files()]:
        text = path.read_text(encoding="utf-8")
        for name, pattern in patterns.items():
            for line in text.splitlines():
                assert not re.search(pattern, line), f"{path.name}: {name} matches a line"


def test_rendering_the_preset_twice_gives_the_same_bytes():
    """`qmine doctor` compares the installed copy against a fresh render.

    Anything non-deterministic in the rendering — a dict iterated in a different
    order, a timestamp — would make that row read `stale` on every machine
    forever, which is how a real staleness check gets ignored.
    """
    assert _dsh_preset() == _dsh_preset()


# ------------------------------------------------------- launching it twice

def _make_recipe(target: str) -> list[str]:
    """The shell lines of one Makefile target, in order."""
    lines = (DSH_PRESET_DIR.parents[3] / "Makefile").read_text(encoding="utf-8").splitlines()
    out, seen = [], False
    for line in lines:
        if line.startswith(f"{target}:"):
            seen = True
            continue
        if seen:
            if line and not line.startswith(("\t", " ")):
                break
            out.append(line)
    assert out, f"no recipe for {target}"
    return out


def test_a_second_launch_is_refused_before_anything_is_written():
    """`make chat` twice is the common mistake and node answers it badly.

    The second launch dies on `EADDRINUSE` with a 40-line stack trace naming
    neither the process holding the port nor the way out — which is exactly what
    happened. Two things have to hold: the port is checked at all, and it is
    checked BEFORE the patch and the preset are rewritten, so a refused launch
    leaves nothing half-written on disk.
    """
    recipe = _make_recipe("chat")
    guard = [i for i, l in enumerate(recipe) if "lsof" in l and "DSH_PORT" in l]
    assert guard, "`make chat` does not check whether the port is already served"

    writes = [i for i, l in enumerate(recipe)
              if "--install-preset" in l or "--print-dsh-config" in l or "dsh web" in l]
    assert writes, "the chat recipe stopped writing the config — this test is stale"
    assert guard[0] < min(writes), (
        "the port guard runs after the config is rewritten: a refused launch would "
        "still mutate the harness directory")


def test_there_is_a_way_to_stop_the_harness():
    """A preset mounts ONCE PER PROCESS, so an edited persona reaches the model
    only after a restart. Without a stop target the documented fix for "I edited
    the persona and nothing changed" is `kill` on a pid the person has to find."""
    recipe = _make_recipe("chat-stop")
    # `"kill" in line` is NOT enough: the recipe also PRINTS `kill -9 <pid>` as a
    # hint for a wedged process, so a chat-stop that only talks about killing
    # would pass. Require an actual invocation.
    assert any(re.search(r"(?:^|;)\s*kill\s+\$\$pid", l) for l in recipe), \
        "chat-stop mentions kill but never invokes it on the pid"
    assert any("lsof" in l and "DSH_PORT" in l for l in recipe), \
        "chat-stop does not find the process by port"
