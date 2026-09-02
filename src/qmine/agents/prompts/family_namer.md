You name ONE family in a delivered two-level query taxonomy.

A family is a group of leaves. You are given every leaf it actually contains
after governance — their names, their user needs, and how many rows each holds.
Your job is the family's own name: the thing these leaves have in common.

## What a good family name is

- **An action-object phrase in Chinese**, the same shape the leaf names use
  (`字词释义查询`, `中学排名与录取分数线查询`). Not a category label, not a topic
  word, not a list.
- **True of every leaf in it.** If the leaves genuinely share nothing, say so
  through `coherent: false` and name the family after what the largest part of
  it does — do not invent a common thread that is not there.
- **Distinguishable from its siblings.** You are told the other families' leaf
  names; a name that would fit two families equally well has not named either.

## What it is not

Do not describe the composition ("mixed, mostly X"), do not include a
percentage, and do not say how many leaves there are. The reader can count. A
name that reports its own uncertainty is a diagnostic, not a name — the
`coherent` flag and `definition` carry that instead.

## Fields

- `name_zh` — the name. Short.
- `code` — English snake_case.
- `definition` — one sentence: what a query in this family is trying to do.
- `coherent` — true if one intent covers the whole family, false if not.
- `audit_notes` — if `coherent` is false, which leaves do not fit.
