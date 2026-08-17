---
name: qmine-review-gate
description: Act as the human reviewer at a QMine quality gate — approve or veto a taxonomy draft, a cluster tree, or a metrics panel. Use when a run is paused awaiting sign-off, or when the user asks whether a produced taxonomy or tree is any good.
---

# Reviewing a QMine gate

You hold a veto, and the veto is not symbolic. The playbook's second principle
is that a reviewer's inability to read the output is a **measurement**, not an
objection to be argued with. If the family layer does not make sense to you, the
correct response is to reject it — not to ask the pipeline to explain itself.

## Resuming a paused run

```bash
qmine inspect <run-id> --what leaves      # or: gates, panel, governance
qmine resume <run-id> --decision approve
qmine resume <run-id> --decision reject --reason "families 3 and 7 are the same intent split by wording"
```

A rejection is recorded as a veto in long-term memory, and routes to a **new
generation** rather than an in-place patch. That is deliberate: the thing you
could not read was produced by the configuration you would otherwise be editing
around, and the old generation is kept because a rejected artifact is still
evidence.

## Reviewing a taxonomy (`p2a_taxonomy`)

Reject if any of these is true:

- **A class is defined by query shape** rather than user intent — "short
  queries", "queries with numbers". That is not a category, it is unsorted
  traffic with a label on it.
- **You cannot tell two classes apart** and no adjudication rule covers the pair.
  Annotators will not manage what you cannot.
- **A `user_need` sentence is not checkable.** "The user understands the topic"
  specifies nothing and will fail as an acceptance criterion.
- **The catch-all is over 5%** and no one has said what should come out of it.
- **The pragmatic intents are missing.** If every class is recoverable from
  wording alone, the taxonomy has not done the one job clustering cannot.

## Reviewing a tree (`p7_tree`)

Read the *family* layer first, and read it as a stranger would:

- **Can you say what each family is in one sentence?** If a family needs "and",
  it is two families.
- **Do two families do the same thing in different words?** This is the
  characteristic failure of a representation that let phrasing outvote meaning.
  Check `Report_BottomUp_Approach.md` → the template-spread figure: a phrasing
  family scattered across several clusters is direct evidence.
- **Is risk content isolated?** It must sit in its own family, never blended
  into a topically similar one, whatever its centroid says.
- **Do the `user_need` sentences distinguish siblings?** Two leaves in one family
  whose needs are interchangeable are one leaf.

Mean coherence below 4.0 means the namers themselves saw mixed clusters. Believe
them — they had the member queries in front of them.

## Reviewing a panel (`p9_panel`)

- Confirm every row shares one `panel_id`. Different panels are not comparable
  and the renderer should have refused; if you see mixed ids, that is a bug.
- Read `template_fragmentation` **beside `n_clusters`**. Fewer clusters fragment
  less by construction, so a bare comparison flatters the coarser option.
- Ignore silhouette when deciding. It is reported because it is conventional and
  barred from voting because it rewards exactly the failure mode in question.

## What good feedback looks like

Name the specific defect and the evidence. "Families 3 and 7 both answer 'what
does this word mean', they differ only in whether the query ends in 什么意思" is
actionable. "The tree feels off" costs a full regeneration and teaches the system
nothing.
