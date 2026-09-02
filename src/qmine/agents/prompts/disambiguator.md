Two or more clusters in the same family were each given the SAME name by
independent blind namers. A reader choosing between them cannot choose at all.

You are shown every one of those clusters — their member queries and nothing
else. Decide which of two things is true, and say so honestly.

## 1. There IS a distinction, and the earlier names missed it

Give each cluster a name that names the difference. The names must be
action-object phrases in Chinese, the same shape as any other leaf name, and
**no two may be equal**. Do not manufacture a difference out of wording: if one
cluster is "X的读音" and the other is "X怎么读", that is the same user need
phrased twice, and it belongs in case 2.

## 2. They are genuinely the same user need

Say so by setting `same_need: true`. Do not invent a distinction to satisfy the
request — a merge is the correct outcome and the pipeline can execute it. A
laboured name that no user would use is worse than a merge.

## What counts as a real distinction

A different **thing the user wants done**, not a different surface form. Some
that are real: asking for a pronunciation versus asking which character has a
pronunciation; a stroke *count* versus a stroke *order*; a definition versus a
usage example. Some that are not: word order, question particles, whether the
query ends in 吗, or which of two synonyms the asker happened to type.

Return one entry per cluster, in the order given, with the cluster's `leaf_id`.
