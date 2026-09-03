<!-- Moved out of README.md: the argument is good and it was burying the results.
     The README keeps a short summary and links here. -->

## Why not just prompt a frontier model?

The honest question, and it deserves measurements rather than adjectives. You
*can* paste queries into a strong model and ask for a taxonomy. Every claim below
is a published result about why the answer cannot be trusted, followed by what
this pipeline does instead.

**A long context is not a read corpus.** Put the relevant material in the middle
of a long input and accuracy falls *below* the same model given no documents at
all — the curve is U-shaped, primacy at the front, recency at the end, a trough in
between ([Liu et al., TACL 2024][lim]). A bigger window does not fix it: GPT-3.5
and GPT-3.5-16K trace nearly superimposed position curves, and Claude-1.3 scores
76.1 against Claude-1.3-100K's 76.4. Nor is it a language-understanding problem —
the same U-shape appears on pure UUID key-value lookup. Under [RULER][ruler],
*effective* context runs well under the advertised number (GPT-4: 128k claimed,
64k effective), and degrades fastest on **aggregation** tasks with many
distractors. A taxonomy over 50,000 queries — roughly 500k tokens — is an
aggregation query with 49,999 distractors, several times outside the regime
anyone has validated.

So QMine never asks a model to read the corpus. Phase 3 encodes every row
numerically, Phases 4–6 cluster them, and agents receive a **card** for one
cluster at a time — centre, random and edge members — a bounded, position-
controlled payload. Counts and shares are arithmetic over the full table, never a
model's impression of it.

**Asking a model to check its own work makes it worse.** Intrinsic
self-correction — review with no external feedback — is measured net-negative:
GPT-4 on GSM8K goes 95.5 → 91.5 → 89.0 over two rounds, GPT-3.5 on CommonSenseQA
75.8 → 38.1 ([Huang et al., ICLR 2024][selfcorr]). The published successes used
**oracle labels** to decide when to stop; remove the oracle and the gains invert.
Adding agents does not rescue it either — multi-agent debate loses to plain
self-consistency at matched inference budget (83.2 vs 85.3 at six responses).

This is why QMine's guardrails are not review agents. Every one is arithmetic
against an external record: written numbers checked value-by-value against a fact
sheet, an agent's assertion evaluated as an expression, an edit required to name
the artifact it came from. And it is the strongest argument for Phase 2b — a
refereed gold set *is* the oracle those results say you need.

**A model grading its own output is not a neutral instrument.** LLM judges
recognise their own generations and the strength of that self-preference is
linearly correlated with the recognition ([Panickssery et al., NeurIPS
2024][selfpref]). They also carry heavy position bias — asked which of two
answers is better, then asked again with the order swapped, Claude-v1 was
consistent 23.8% of the time — and are fooled by verbose restatement 91.3% of the
time ([Zheng et al., NeurIPS 2023][judge]). The constructive finding in that same
paper is the one QMine is built on: chain-of-thought only halved a judge's
grading errors (14/20 → 6/20), but giving it a **reference answer** took them to
3/20. An answer key beats a better prompt.

QMine's Phase 9 panel is blind, and Phase 7's fan-out makes blindness structural
rather than an instruction — each naming agent sees only its own payload.
Anything that decides is a measurement, not a judgement.

**A labelling scheme already has an acceptance test, and it is strict.**
Computational linguistics reads κ > 0.8 as reliable and 0.67–0.8 as supporting
only tentative conclusions; Krippendorff's own preconditions include a guide
**fixed in advance**, coders working **independently**, and no discussion or
voting — "any of these practices make the resulting data unusable for measuring
reproducibility" ([Artstein & Poesio, CL 2008][kappa]). A single prompt produces
a scheme for which no such number exists, or can exist.

Each of those preconditions is a test here: the guide is frozen before annotation
and every rule the architect *and* the referee write must reach the annotator;
each annotator's labels come back as its own; a row nobody labelled is missing
data, not agreement; guide repair runs on a fresh sample so refereed rows survive.
And because an annotator disagrees with **itself** about a quarter of the time
([Abercrombie et al.][intra]), κ is read against a self-consistency ceiling —
which is what separates a fixable guide gap from irreducible ambiguity.

**The convenient clustering scores do not mean what they look like.** Silhouette
is defined on the dissimilarity you hand it ([Rousseeuw 1987][sil]), so comparing
one embedding's silhouette to another's compares two different measurements that
share a name — and embedding geometry makes the absolute values untrustworthy
across spaces anyway ([Ethayarajh, EMNLP 2019][aniso]; [Steck et al., WWW
2024][cosine]). Stability is asymmetric for a deeper reason: high instability is
provably informative, but the converse **is not a theorem** — counter-examples
exist, and in the large-*n* K-means limit stability tracks the symmetry of the
distribution rather than whether K is right ([von Luxburg 2010][stab]).

Hence the rule that K is *located* by intent alignment and stability may only
*reject*; hence silhouette is reported with no voting rights; hence a null test,
because a model asked for a taxonomy will always return one ([Ben-Hur et
al. 2002][benhur]).

![The unified panel — silhouette is reported in hatched bars marked "no voting rights"](img/fig6_panel.png)

Measured on this corpus, that exclusion changes the answer. The same intent, in
the three candidate spaces — the α silhouette would have picked (0.8) scatters it
across 2.37 families where the chosen α=0.1 keeps it in 1.86:

![The same intent, split across three embedding spaces](img/fig5_intent_split.png)

**There is no single true taxonomy to find.** What counts as correct is set by
the aim, so clustering becomes scientific through open communication of the aims
and choices, not through uniqueness of the solution ([Hennig 2015][hennig]).
That makes the record the deliverable. A dataset others build on is expected to
document its motivation, composition and collection process
([Datasheets][datasheets]); a model shipped for use is expected to report
disaggregated performance, not one aggregate ([Model Cards][cards]); and
provenance — which entity was generated by which activity, derived from what — is
a [W3C Recommendation][prov] with a data model, not a vibe. A run directory here
answers those by construction: which rows, which sampling, which encoder, which
guide version, which annotators, what *n*, what was excluded and why.

The [decision chain figure](../README.md#how-it-works) in the README shows this
for a whole run: every choice, its rejected candidates, and the metric that
settled it.

**The fair comparison, stated honestly.** Reported self-correction gains have
turned out to be artefacts of a weak starting prompt — on CommonGen-Hard, one
well-written prompt beat seven rounds of refinement, 81.8 to 67.0
([Huang et al.][selfcorr] §5). The same standard applies to us: a pipeline
costing *N* model calls should be compared against the best *N*-call baseline,
not against one lazy call, and any advantage that a better prompt would also
supply is not an advantage of the architecture. What is left after that test is
the part a prompt cannot produce at any length: a labelled gold set with a κ and
an *n*, a K chosen against a null, an adversary paid to falsify the labels, and a
partition re-measured *after* governance rewrote it.

[lim]: https://aclanthology.org/2024.tacl-1.9/
[ruler]: https://arxiv.org/abs/2404.06654
[selfcorr]: https://arxiv.org/abs/2310.01798
[selfpref]: https://arxiv.org/abs/2404.13076
[judge]: https://arxiv.org/abs/2306.05685
[kappa]: https://aclanthology.org/J08-4004/
[intra]: https://arxiv.org/abs/2301.10684
[sil]: https://doi.org/10.1016/0377-0427(87)90125-7
[aniso]: https://aclanthology.org/D19-1006/
[cosine]: https://arxiv.org/abs/2403.05440
[stab]: https://arxiv.org/abs/1007.1075
[benhur]: https://psb.stanford.edu/psb-online/proceedings/psb02/benhur.pdf
[hennig]: https://doi.org/10.1016/j.patrec.2015.04.009
[datasheets]: https://arxiv.org/abs/1803.09010
[cards]: https://arxiv.org/abs/1810.03993
[prov]: https://www.w3.org/TR/prov-overview/

---
