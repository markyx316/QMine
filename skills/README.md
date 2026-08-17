# Agent Skills

Five skills that make QMine drivable from Claude Code without reading the source.

| skill | use it when |
|---|---|
| [`qmine-run`](qmine-run/SKILL.md) | you have a query log and want the whole pipeline |
| [`qmine-new-domain`](qmine-new-domain/SKILL.md) | no bundled profile fits the corpus |
| [`qmine-review-gate`](qmine-review-gate/SKILL.md) | a run is paused for sign-off, or you must judge a tree |
| [`qmine-diagnose`](qmine-diagnose/SKILL.md) | a gate failed, a run halted, or a metric looks wrong |
| [`qmine-interpret-tree`](qmine-interpret-tree/SKILL.md) | a run finished and you need to act on it |

To install, copy this directory into `~/.claude/skills/` (personal) or
`.claude/skills/` (project). Each folder's `SKILL.md` carries the YAML
frontmatter Claude Code reads to decide when the skill applies.
