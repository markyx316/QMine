---
name: qmine-answer-from-a-study
description: Answer a question about a finished QMine run — which tool to reach for, how to cite it, and the checks that stop a correct number being stated under the wrong caveat.
whenToUse: The person asks what a study found, why a class behaves a certain way, for examples, for a table, or for a section of a delivered report.
---

# Answering from a finished study

## 1. Find the run before anything else

If no run was named, call `qmine_list_runs` and **ask** rather than guess. Run ids
are terse (`med-pool8`, `health-pool3`, `fin-pool8`); the listing carries the mode,
the provider and whether a comparison was built, which is usually enough for the
person to point at the one they meant.

Then call `qmine_overview` **once**, before any substantive claim. It tells you
three things that change how every later number must be written:

| field | if it says | then |
|---|---|---|
| `mode` | `fast` | kappa is **absent**; `fast_skipped` lists what went unchecked |
| `llm_usage.provider` | `offline` | a deterministic stand-in wrote it — it says nothing about the corpus. Say so and stop |
| `read_this_first` | anything | read it out; it is the run's own caveat |

A run with no `run_summary.json` has not finished. Switch to `qmine_status`.

## 2. Reach for the narrowest tool

| the question | the tool |
|---|---|
| "what did it find?" | `qmine_findings` |
| "why is <class> like that?" | `qmine_class` |
| "show me real queries" | `qmine_examples` |
| "rank / filter / count something" | `qmine_tables`, then `qmine_table` |
| "what does the report say about X?" | `qmine_document` with a `section` |
| "what does <term> mean?" | `qmine_glossary` |

**The tables' column names are Chinese** — `类目`, `搜索_占比%`, `均衡指数`, not
`class` or `share`. Call `qmine_tables` first and filter on the names it prints; a
`where` on a column that does not exist comes back with every row and a note
saying it was ignored, which is easy to skim past.

`qmine_document` never returns a whole document — reports run to hundreds of
lines. Ask for the section. If you do not know the heading, `qmine_document` with
no `section` lists them.

Do not open files under `runs/` to answer a results question. The tools apply the
quote guard, the rounding and the caveats; raw files do not.

## 3. Cite so the answer can be checked

Every number carries **run id + generation + the table or document it came from**.
People act on these. "med-pool8 / gen01, `class_matrix.csv`" is a citation;
"the analysis shows" is not.

When the person asks for a table, prefer returning the tool's own rows over
re-typing them. When you must summarise, keep the class name in its original
Chinese and put any translation in parentheses — never replace the label.

## 4. Six checks before you send

1. **Every distance sits beside its noise ceiling.** Below the ceiling means the
   data cannot tell the snapshots apart — never "they are the same".
2. **The axis matches the verbs.** `stratum` forbids "grew", "fell", "over time".
   `qmine_findings` and `qmine_overview` both report it.
3. **No bare zero.** A zero goes out with its verdict (`真缺席` / `偏少但证据弱` /
   `不可判定`) and the bound behind it.
4. **Every quoted row came from `qmine_examples` or `qmine_table`.** Nothing else
   has been through the guard. Never quote a private individual's details, a named
   doctor, sexual content, anything involving minors, or debt-collection /
   credit-repair / cash-out rows — give a count and a source instead. A guard is a
   floor, not a licence: a row in one of these classes stays unquoted even when the
   guard returned it, because no regex has ever caught every shape of a personal
   name.
5. **Nothing illustrates what you withheld.** "I left out rows like 〈specimen〉"
   is the disclosure you just declined to make. Say what the class is, how many
   rows it holds and where it lives — never a member of it. This is the way the
   guard gets defeated in practice: not by a leak, but by a caveat carrying its own
   counterexample.
6. **Nothing was computed by you.** If a tool did not return it, it is not
   measured. Say that.

## 5. When the answer is "this run cannot tell you"

That is a real answer and often the right one. A class the taxonomy never
separated, a difference under the ceiling, a check `fast` mode skipped — say
which of these it is and what would answer it (a full-mode run, a bigger sample,
a different axis). Do not reach for a second run to fill the gap: two runs'
labels share no codes.
