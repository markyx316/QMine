# LLM Taxonomy Induction, Labeling & Naming Methods

> Dossier gathered 2026-08-17 for the QMine build. Signatures marked *verified* were
> introspected from installed packages; everything else is from live documentation.

# LLM-Assisted Taxonomy Induction, Intent Discovery, Cluster Naming & Annotation Quality
### Research dossier for playbook phases P2 (taxonomy + gold labels) and P7 (blind naming)
*Verified against primary sources, August 2026. All prompt text below is either verbatim from published artifacts (marked VERBATIM) or an adaptation I wrote for this pipeline (marked ADAPTED).*

---

## 1. TnT-LLM (KDD'24, Microsoft) — arXiv:2403.12173 / DOI 10.1145/3637528.3671647

**This is the single closest published analogue to your P2. Copy its architecture.**

### 1.1 The actual algorithm

**Phase 1 — Taxonomy generation** (framed explicitly as prompt-based mixture-model fitting optimized by "SGD" à la Pryzant et al. ProTeGi):

- **Stage 1 — Summarization.** Each document is summarized independently with a cheap model (GPT-3.5-Turbo), given (a) a blurb about the intended use-case (e.g. "intent detection") and (b) a target length (20 words). This is the "featurization" step; the summary is the feature vector `x_i`. The paper notes this stage *can be skipped when inputs are short and normative* — **which is exactly the case for search queries.** For query logs, skip or replace with a cheap "canonicalize + expand abbreviations" step.
- **Stage 2 — Create / Update / Review over minibatches.**
  - Summaries split into **equal-sized minibatches of 200**.
  - **Initial generation prompt** on minibatch 1 → taxonomy `Θ₀` (a markdown/XML table of `id | name | description`).
  - **Update prompt** on each subsequent minibatch, doing three things in one call: (1) *evaluate* the current taxonomy against the new batch (the "loss"), (2) *identify issues and suggestions* (the "gradient"), (3) *modify* the taxonomy (the "step"). Learning rate is implicit in the LLM.
  - **Review prompt** at the end: format + quality check only (no data), producing the final taxonomy.
  - Hierarchy comes free: **re-run Stage 2 within each L1 group** to produce L2 sub-intents. This is exactly your P2 L2 step.
- **"Model selection" / early stopping.** A separate **evaluation prompt** takes *N candidate taxonomies + a batch of summaries + the use-case instruction* and returns the index of the best one. Run on a **held-out validation split**, after every k update steps; keep a running best; classic early-stopping semantics. Positions of the candidates are randomized and the eval is repeated to defeat position bias.
- **Trials.** They ran the whole Phase-1 chain **10 times (epochs)** and selected the best run by validation.

**Phase 2 — LLM-augmented classification (distillation).**
- Prompt GPT-4 to assign, for each doc, a **primary label (multiclass)** and **all applicable labels (multilabel)**.
- Treat those as pseudo-labels; train **Logistic Regression / LightGBM / 2-layer MLP** on frozen embeddings (ada2, Instructor-XL).
- Deploy the lightweight classifier for the whole corpus + online serving.

### 1.2 Exact hyperparameters (from Appendix C.2)

| Setting | Value |
|---|---|
| `frequency_penalty` / `presence_penalty` | 0 |
| `top_p` | 0.5 |
| base temperature, **generation** prompt | **0.5** |
| base temperature, **update** prompt | **0.2** |
| base temperature, all other prompts (summarize, assign, review, eval) | **0.0** |
| minibatch size | 200 |
| Phase-1 corpus sample | ~10k docs ("sufficient for a taxonomy of ≤100 labels") |
| Phase-2 corpus sample | "medium-to-large", 48k conversations |
| Retries on guardrail failure | 5, **incrementing temperature by 0.1 each retry** |
| Taxonomy sizes forced | 10 intents, 25 domains |
| Classifier grid | LR: ℓ2 λ∈[0.01,0.1,1,10]; LightGBM: 31 leaves, depth∈[3,5,7,9]; MLP: Adam, lr 1e-3, wd 1e-5, hidden∈[32,64,128,256] |

### 1.3 Robustness engineering (Appendix C.1) — adopt verbatim

- Force output into **predefined XML tags** (`<output>…</output>`) so each prompt-chain step is parseable.
- Force the taxonomy to be a **markdown table with a fixed schema** (`index | name | description`); require the model to emit **both the label index and the label name** when assigning — this alone materially raises assignment consistency and cuts post-processing.
- **Guardrail test suite per prompt type**: (1) parses to the declared format? (2) correct output language? (3) satisfies a key verifiable constraint (e.g. `len(taxonomy) ≤ max_num_clusters`)? These double as an instruction-following benchmark for model selection.
- Measured instruction-following: GPT-4 100% pass on format/language and 10/10 epochs completed; GPT-3.5-Turbo <0.01% format failures but ~2% language failures, and it **completed only 4/10 intent epochs and 1/10 domain epochs because it persistently exceeded the taxonomy size limit in the Update step**. → *Do not use a weak model for the update step.*

### 1.4 Evaluation suite (Phase 1)

Three criteria, each computable by human **or** LLM raters:
1. **Coverage** — put an explicit `Other/Undefined` bucket in the *assignment* prompt (never in the taxonomy itself) and measure the fraction routed there. Lower = better coverage. TnT-LLM got **>99.5% coverage** on both taxonomies.
2. **Label accuracy** — **pairwise forced choice**: show the text, the assigned primary label, and one *random negative label from the same taxonomy* (names + descriptions); rater picks the better one; a "None" escape exists but raters are told to minimize it. Report **hit rate**. Positions fully randomized.
3. **Use-case relevance** — binary: given an instance and its primary label name+description, is the label relevant to the use-case instruction? (Guards against taxonomies that sound relevant but do not describe *this* corpus.)

**Reported inter-rater agreement (BingChat-Phase1-S-Eng, n=200, 3 author-raters):**

| Metric | Use case | Fleiss κ (humans) | Avg pairwise Cohen κ | GPT-3.5 vs human consensus | GPT-4 vs human consensus |
|---|---|---|---|---|---|
| Accuracy | Intent | 0.476 | 0.477 | 0.376 | **0.558** |
| Accuracy | Domain | 0.478 | 0.484 | 0.260 | **0.578** |
| Relevance | Intent | 0.466 | 0.481 | 0.333 | **0.520** |
| Relevance | Domain | 0.379 | 0.399 | 0.177 | 0.288 |

**Key finding: GPT-4 agreed with the human majority *more* than humans agreed with each other** on the pairwise tasks. GPT-3.5 was not usable as a rater.

**Phase 2 label-assignment agreement (n=400, 4 annotators, 3 per item + tie-breaker):**

| Metric | Use case | Fleiss κ | Avg pairwise Cohen κ | GPT-4 vs resolved human |
|---|---|---|---|---|
| Primary label | Intent | 0.553 | 0.559 | 0.572 |
| Primary label | Domain | **0.624** | 0.624 | **0.695** |
| All labels (exact match) | Intent | 0.422 | 0.427 | 0.271 |
| All labels (exact match) | Domain | 0.467 | 0.467 | 0.102 |

**Critical practical lessons:**
- Multi-label exact-match agreement between LLM and humans **collapses** (0.10–0.27) because **GPT-4 is far more liberal than humans — high recall, low precision, it applies every plausible label.** If you need multi-label, calibrate with an explicit "apply a label only if it is a *primary or co-equal* driver of the query" rule, or threshold on a per-label confidence.
- The *smaller* (10-class intent) taxonomy was **harder** for humans to agree on than the *larger* (25-class domain) one — size is not the difficulty driver, **reasoning depth and definitional ambiguity are**.
- Disagreements concentrate at specific boundary pairs (e.g. "Fact-based information seeking" ↔ "Clarification and concept explanation"; "General solution/advice seeking" ↔ "Technical assistance"). **Build a confusion matrix of human-vs-human and LLM-vs-human, find the top confusion pairs, and write explicit tie-breaking rules into the label descriptions.** This is the single highest-ROI refinement loop.
- Annotator expertise skew is a real bias: all their annotators were highly technical, which shifted where "technical" boundaries sat.
- They performed a **"lightweight human calibration"** of the auto-generated taxonomy before Phase 2 (clarified wording, added illustrative examples). Budget for this gate.

### 1.5 Phase 2 results (distillation works)

Primary-label accuracy, human annotations as oracle (n=400 English):
- GPT-4 direct classifier: **0.655** accuracy / 0.640 macro-F1 (intent)
- ada2 + LogisticRegression on GPT-4 pseudo-labels: **0.658** (statistically indistinguishable from GPT-4, paired t-test p<0.05)
- ada2 + MLP: 0.658; ada2 + LightGBM: 0.642
- Instructor-XL variants were slightly worse and **much** worse on non-English (−9.9% vs −2.7% for ada2)

For *all-applicable-labels*, the distilled classifiers **beat** GPT-4 (e.g. accuracy 0.388 vs 0.320) because they inherit the human-calibrated precision/recall tradeoff from training rather than GPT-4's over-application.

**Takeaway for P10: a logistic regression on good embeddings, trained on ~40k LLM pseudo-labels, matches a frontier LLM classifier at ~1/1000 the serving cost.**

### 1.6 THE ACTUAL TnT-LLM PROMPTS (VERBATIM)

The paper renders prompts as figure images (unextractable from the PDF). However the LangChain/LangGraph reference implementation publishes the reproduced prompts on LangSmith Hub, which I pulled via `https://api.smith.langchain.com/commits/wfh/<name>/latest`. Hub names: `wfh/tnt-llm-summary-generation`, `wfh/tnt-llm-taxonomy-generation`, `wfh/tnt-llm-taxonomy-update`, `wfh/tnt-llm-taxonomy-review`, `wfh/tnt-llm-classify`.

**(a) Summary generation — VERBATIM**
```
# Instruction
## Context
- **Goal**: You are tasked with summarizing the input text for the given use case. The summary will represent the input data for clustering in the next step.
- **Data**: Your input data is a conversation history between a User and an AI agent.

# Data
<data>
{content}
</data>

------------------------------
# Questions
## Q1. Summarize the input text in {summary_length} words or less for the use case.
Write the summary between <summary> </summary> tags.

Tips:
- The summary should contain the relevant information for the use case in as much detail as possible.
- Be concise and clear. Do not add phrases like "This is the summary of the data ..." or "Summarized text: ...".
- Similarly, do not reference the user ('the user asked XYZ') unless it's absolutely relevant.
- Within {summary_length} words, include as much relevant information as possible.
- Do not include any line breaks in the summary.
- Provide your answer in **English** only.

## Q2. Explain how you wrote the summary in {explanation_length} words or less.

## Provide your answers between the tags <summary>your answer to Q1</summary>, <explanation>your answer to Q2</explanation>

# Output
```

**(b) Taxonomy GENERATION — VERBATIM (this is your P2 taxonomy-proposal prompt)**
```
# Instruction
## Context
- **Goal**: Your goal is to cluster the input data into meaningful categories for the given use case.
- **Data**: The input data will be a list of human-AI conversation summaries in XML format, including the following elements:
  - **id**: conversation index.
  - **text**: conversation summary.
- **Use case**: {use_case}
## Requirements
### Format
- Output clusters in **XML format** with each cluster as a `<cluster>` element, containing the following sub-elements:
  - **id**: category number starting from 1 in an incremental manner.
  - **name**: category name should be **within {cluster_name_length} words**. It can be either verb phrase or noun phrase, whichever is more appropriate.
  - **description**: category description should be **within {cluster_description_length} words**.
Here is an example of your output:
```xml
<clusters>
  <cluster>
    <id>category id</id>
    <name>category name</name>
    <description>category description</description>
  </cluster>
</clusters>
```
- Total number of categories should be **no more than {max_num_clusters}**.
- Output should be in **English** only.
### Quality
- **No overlap or contradiction** among the categories.
- **Name** is a concise and clear label for the category. Use only phrases that are specific to each category and avoid those that are common to all categories.
- **Description** differentiates one category from another.
- **Name** and **description** can **accurately** and **consistently** classify new data points **without ambiguity**.
- **Name** and **description** are *consistent with each other*.
- Output clusters match the data as closely as possible, without missing important categories or adding unnecessary ones.
- Output clusters should strive to be orthogonal, providing solid coverage of the target domain.
- Output clusters serve the given use case well.
- Output clusters should be specific and meaningful. Do not invent categories that are not in the data.

# Data
<conversations>
{data_xml}
</conversations>
------------------------------
# Questions
## Q1. Please generate a cluster table from the input data that meets the requirements.
Tips
- The cluster table should be a **flat list** of **mutually exclusive** categories. Sort them based on their semantic relatedness.
- Though you should aim for {max_num_clusters} categories, you can have *fewer than {max_num_clusters} categories* in the cluster table;  but **do not exceed the limit.**
- Be **specific** about each category. **Do not include vague categories** such as "Other", "General", "Unclear", "Miscellaneous" or "Undefined" in the cluster table.
- You can ignore low quality or ambiguous data points.
## Q2. Why did you cluster the data the way you did? Explain your reasoning **within {explanation_length} words**.
## Provide your answers between the tags: <cluster_table>...</cluster_table>, <explanation>...</explanation>
# Output
```

**(c) Taxonomy UPDATE — VERBATIM (the "SGD step"; note it *scores* the current taxonomy 0–100, explains, suggests edits, then emits the updated table — four outputs in one call, which gives you a trainable quality curve across minibatches)**
```
# Instruction
## Context
- **Goal**: You goal is to review the given reference table based on the input data for the specified use case, then update the reference table if needed.
  - You will be given a reference cluster table, which is built on existing data. The reference table will be used to classify new data points.
  - You will compare the input data with the reference table, output a rating score of the quality of the reference table, suggest potential edits, and update the reference table if needed.
[... same Format + Quality requirement block as generation ...]
# Reference cluster table
<reference_table>{cluster_table_xml}</reference_table>
# Data
<conversations>{data_xml}</conversations>
------------------------------
# Questions
## Q1: Review the given reference table and the input data and provide a rating score of the reference table. The rating score should be an integer between 0 and 100, higher rating score means better quality. You should consider the following factors when rating the reference cluster table:
- **Intrinsic quality**:
  - 1) if the cluster table meets the *Requirements* section, with clear and consistent category names and descriptions, and no overlap or contradiction among the categories;
  - 2) if the categories in the cluster table are relevant to the the given use case;
  - 3) if the cluster table includes any vague categories such as "Other", "General", "Unclear", "Miscellaneous" or "Undefined".
- **Extrinsic quality**:
  - 1) if the cluster table can accurately and consistently classify the input data without ambiguity;
  - 2) if there are missing categories in the cluster table but appear in the input data;
  - 3) if there are unnecessary categories in the cluster table that do not appear in the input data.
## Q2: Explain your rating score in Q1 **within {explanation_length} words**.
## Q3: Based on your review, decide if you need to edit the reference table to improve its quality. If yes, suggest potential edits **within {suggestion_length} words**. If no, please output the original reference table.
Tips:
- You can edit the category name, description, or remove a category. You can also merge or add new categories if needed. Your edits should meet the *Requirements* section.
- The cluster table should be a **flat list** of **mutually exclusive** categories. Sort them based on their semantic relatedness.
- You can have *fewer than {max_num_clusters} categories* in the cluster table, but **do not exceed the limit.**
- Be **specific** about each category. **Do not include vague categories** ...
- You can ignore low quality or ambiguous data points.
## Q4: If you decide to edit the reference table, please provide your updated reference table. ...
## Provide your answers between the following tags:
<rating_score>integer between 0 and 100</rating_score>
<explanation>...</explanation>
<suggestions>suggested edits within {suggestion_length} words, or "N/A" if no edits needed</suggestions>
<updated_table>...</updated_table>
# Output
```

**(d) Taxonomy REVIEW — VERBATIM.** Identical structure to Update but **with no data block** — it audits the table against the requirements alone. Same four output tags.

**(e) Classify / label-assignment — VERBATIM**
```
Your task is to use the provided taxonomy to categorize the overall topic or intent of a conversation between a human and an AI assistant.
First, here is the taxonomy to use:
<taxonomy>
{taxonomy}
</taxonomy>
To complete the task:
1. Carefully read through the entire conversation, paying attention to the key topics discussed and the apparent intents behind the human's messages.
2. Consult the taxonomy and identify the single most relevant category that best captures the overall topic or intent of the conversation.
3. Write out a chain of reasoning for why you selected that category. Explain how the category fits the content of the conversation, referencing specific statements or passages as evidence. Output this reasoning inside <reasoning></reasoning> tags.
4. Output the name of the category you chose inside <category></category> tags.
That's it! Remember, choose the single most relevant category. Don't choose multiple categories. Think it through carefully and explain your reasoning before giving your final category choice.
---
Assign a single category to the following content:
<content>{content}</content>
Respond with your reasoning and category within XML tags. Do not include the number, just the category text.
```

**Example of the label-description style TnT-LLM converged to** (note the embedded contrastive cues in `*emphasis*` and the concrete example — copy this style):
> **Fact-Based Information Seeking** — "User seeks factual and descriptive information on a specific topic, product, or service. These user queries can be answered by retrieving the factual information that *already exists in the sources* and require *a high level of specificity* and *low level of subjectivity*, e.g., 'What is the capital of France?'."

---

## 2. ClusterLLM (EMNLP'23) — arXiv:2305.14871

Two orthogonal stages; both are *cheap oracles you can bolt onto P5/P6*, not a full clustering method.

### 2.1 Stage 1 — Triplet feedback for **perspective**

- Task: `<does A better correspond to B than C>`, i.e. anchor `a` + choices `(c₁, c₂)`, plus a **user instruction encoding the desired perspective** (intent vs domain vs emotion vs entity type). This is the key trick: the *same* corpus can be clustered along different axes purely by swapping one sentence of instruction.
- **Entropy-based triplet sampling (Algorithm 1)** — the sample-efficiency engine:
  1. Cluster embeddings `Z` into K clusters; compute centroids μ_k.
  2. Soft assignment with Student-t (α=1): `p_ik = (1+||z_i−μ_k||²)⁻¹ / Σ_k'(1+||z_i−μ_k'||²)⁻¹`.
  3. `K_closest = max(εK, 2)` with **ε = 2%**; renormalize `p` over those K_closest and compute entropy `h_i = −Σ p'_ik log p'_ik`.
  4. Sort descending by entropy; **keep the slice [γ_high·N : γ_low·N]** (hyperparameters that let you skip the very top, which is often pure noise).
  5. For each anchor, sample two *different* nearby clusters and one instance from each → hard positives/hard negatives by construction.
  6. Dedupe, drop triplets where a choice equals the anchor, stop at budget **Q**.
- **Q is independent of dataset size.** Q = **1,024 triplets** improved clustering equally on 3k-item and 50k-item versions of the same datasets.
- Fine-tune the embedder with InfoNCE using the hard negative + in-batch negatives (Instructor lr 2e-6, E5 lr 2e-7, batch 16, eval with K-means at 200 iters). Symmetrize by swapping anchor/positive.
- Iterating (resample from the fine-tuned space, continue fine-tuning) helps and **saturates around iteration 4**.
- Generation config for the triplet query: **temperature 0.5, max_tokens 10**, suffix `"Please respond with 'Choice 1' or 'Choice 2' without explanation."`; parse by substring; discard non-conforming responses.

**Triplet prompt — VERBATIM example (Bank77):**
```
Select the banking customer utterance that better corresponds with the Query in terms of intent.
Query: Should i reinstall the payment app?
Choice 1: I've received my card so now I need to know how to sync it to the app.
Choice 2: Can I still use the app if I switched phones?
Please respond with 'Choice 1' or 'Choice 2' without explanation.
```
Instruction prefixes vary only in the final clause: `…in terms of intent / relation type / entity type / event type / topic / domain / scenario / emotion expressed`.

### 2.2 Stage 2 — Pairwise feedback for **granularity** (directly relevant to your P5 K-triangulation)

- Build a **cluster hierarchy** by agglomerative merging (on top of mini-batch K-means for scale — Appendix A).
- Assume `k_min`, `k_max` (they used **2 and 200**).
- At each merge step, sample **λ pairs** from the two clusters being merged (λ ∈ {1, 3}) → `N_p = λ(k_max − k_min)` pairs total → **198 or 594 LLM calls, total**, covering the full granularity range.
- Query: `<do A and B belong to the same category>` with **few-shot demonstrations that include justifications**.
- Choose `k* = argmax_k M(W^p, W^k)` where `W^p` are LLM answers, `W^k` are the same-cluster indicators at level k. **Empirically the best consistency measure M is F-beta with LLM predictions as the *labels* and the clustering as the *predictions*.**

**Pairwise granularity prompt — VERBATIM (Bank77):**
```
[Example1]
Sentence 1: I would like to see the source of my money.
Sentence 2: My source of funds need verified.
Yes. Because both intents are verify source of funds.
[Example2]
Sentence 1: Is there a fee for topping up
Sentence 2: What are the top up charges for US cards?
Yes. Because both intents are top up by card charge.
[Example3]
Sentence 1: Can I reactivate my lost card that I found this morning in my jacket pocket?
Sentence 2: how to activate card?
No. Because Sentence 1 has intent card linking and Sentence 2 has intent activate my card.
[Example4]
Sentence 1: What will I be charged for a physical card?
Sentence 2: My card is about to expire and I need to know how much it costs and how long …
No. Because Sentence 1 has intent order physical card and Sentence 2 has intent card …
Determine whether the intents of two banking customer utterances below belong to the same intent category using above examples.
Sentence 1: $1 extra has been charged on my statement, why is that?
Sentence 2: Will it automatically top-up if there isn't much money left?
Please respond with 'Yes' or 'No' without explanation.
```
**Ablation on prompt design (Table 8): removing the justifications from demonstrations hurts; removing demonstrations entirely (bare "Determine whether the two sentences below belong to the same category") hurts more.** So: few-shot + rationales are load-bearing here.

### 2.3 Cost & results

- **~$0.60 per dataset total with gpt-3.5-turbo: ~$0.20 for perspective (1,024 triplets), ~$0.40 for granularity (198–594 pairs).**
- 14 datasets, 10–150 clusters, small (~3k) and large (~50k) variants: consistent ACC/NMI gains over E5/Instructor base, SCCL, and self-supervised baselines. E.g. +6.71% on FewRel (Instructor), +9.19 ACC on Bank77 (E5).
- **GPT-4 significantly beats GPT-3.5 on the granularity task** (better use of demonstrations). Baseline BIC predicted k=69/64 for MTOP intent/domain (can't distinguish granularities); ClusterLLM predicted 92/18 — the right shape.

### 2.4 Limitations you must plan around

- **Domain/coarse-grained discovery is where it fails.** On CLINC(D), MTOP(D), Massive(D) (10–18 clusters) performance was flat or *worse*. Their hypothesis: fine-tuning compacts the space into small cliques — good for fine-grained, bad for coarse. **Implication for you: use ClusterLLM-style triplet fine-tuning for L2 sub-intents, NOT for L1.**
- Granularity selection is only as good as the underlying hierarchy — perfect pairwise answers on a random hierarchy still find nothing.
- Triplets may be "both correct"/"neither"; they accept this as soft ranking signal.

---

## 3. IDAS — Intent Discovery with Abstractive Summarization (NLP4ConvAI @ ACL'23) — arXiv:2305.19783

Four steps:
1. **Initial clustering** on frozen encoder E → K clusters; **prototype = the utterance closest to each centroid**.
2. **Prototype labeling** (no demos): `"Describe the {domain} question in a maximum of 5 words."`
3. **ICL label generation for everything else**: retrieve the **n = 8 most similar already-labeled utterances** (KATE-style similarity retrieval) from a **growing memory M** (initialized to the prototypes; every newly labeled utterance is appended). Instruction: `"Classify the {domain} question into one of the provided labels."` — but the output is *free generation*, not a closed verbalizer set, so new intents can still emerge. This is a clever anti-error-propagation design.
4. **Encode + smooth**: `φ_avg(x,ℓ) = (E(x)+E(ℓ))/2`, then `φ_smooth(x,ℓ) = mean over the n' nearest neighbours (incl. self) of φ_avg`. **n' chosen by maximizing mean silhouette, swept 5→45.** Final K-means on φ_smooth.

**Config:** `text-davinci-003`, **temperature 0** (to minimize label variation for same-intent utterances), 5 label-generation runs × 10 K-means seeds, ARI/NMI/ACC. Up to **+7.42%** over SOTA on Banking/StackOverflow.

**The qualitative analysis is the most useful part for P7:**
- Within a good cluster, the modal generated label dominates hard (e.g. "Magento" for 47/49).
- **Two failure modes to detect automatically:**
  - *Slightly-too-specific labels* — harmless, because they share syntactic/lexical structure (`"Plug Converter <noun>"`). Detect via high lexical overlap / shared head-word.
  - *Overly-general labels* — **harmful**, they merge distinct intents under a shared superordinate ("Foreign currency exchange" swallowing both `exchange_via_app` and `fiat_currency_support`; 6 of 25 wrong). Detect via: label appears as the modal label of ≥2 clusters, or label entropy within cluster is low but gold-proxy purity is low.

---

## 4. Dial-In LLM (arXiv:2412.09049, v3 May 2025) — LLM-in-the-loop intent clustering + "Action-Object" naming

### 4.1 The naming convention — directly adoptable for P7

**"Action-Objective"** (their term; "action-object" in your brief): e.g. `inquire-promotion`, `answer-amount`, `inquire-accident`, `provide-location`. Table 11 compares conventions across datasets:

| Dataset | Convention | Example sentence | Example label |
|---|---|---|---|
| NLU | scenario-intent | "Send an email to Alex and write thank you." | `email sendemail` |
| NLU++ | list of keywords | "How long does it usually take to get a new pin?" | `["how_long","pin","arrival","new"]` |
| OOS/CLINC | objective only | "Please tell me why my card was declined yesterday." | `card_declined` |
| **Dial-In (theirs)** | **action-objective** | "Well, do you have any other discounts?" | `inquire-promotion` |

Their claim: action-objective is superior for goal-oriented expression because dialogue intents are simultaneously topic-oriented (insurance, loan) **and** action-bearing (inquire, confirm). **For search-query mining the analogue is `verb-object` or `intent-verb + entity-slot`** — e.g. `compare-prices`, `find-tutorial`, `check-eligibility`, `define-term`. It also gives you a *free structural axis*: they exploit the action prefix to split customer vs agent roles unsupervised (`inquire-*` ⇒ customer). **You can exploit the same to auto-derive a second facet (e.g. transactional / informational / navigational) from the action verb — a Broder-style facet for free.**

### 4.2 Two fine-tuned LLM "utilities"

**(a) Coherence evaluator** `M_eval(C) ∈ {good, bad}` — binary judgment on whether all sentences in a cluster share one intent. This is your **automatic mixed-cluster detector**.

Accuracy on 480 unseen clusters:

| Model | Accuracy |
|---|---|
| Qwen2.5-7B (LoRA fine-tuned) | **96.25%** |
| Qwen2.5-14B (LoRA) | **97.50%** |
| ChatGLM3-6B (LoRA) | 95.83% |
| Baichuan2-7B (LoRA) | 89.17% |
| GPT-4o (zero/few-shot API) | 94.17% |
| GPT-4 | 93.54% |
| GPT-3.5 | 89.58% |
| **Qwen2.5-7B, NOT fine-tuned** | **75.63%** |

**Fine-tuning on ~1,772 human-labeled good/bad clusters buys +20 points over the same base model and beats GPT-4o.** Training data for naming: 2,500 clusters × 20 sentences each.

**(b) Naming utility** `M_name(C) → action-objective label`. Human-expert-judged accuracy: qwen7b 92.8%, qwen14b 94.3%, baichuan2-7b 94.3%, chatglm3-6b 94.4%.

**Prompts — VERBATIM (translated from Chinese in the paper), each with 5 few-shot input/output pairs:**
```
Coherence Evaluation — You are a helpful assistant for sentence clustering. Based on the
relevancy and common points of the following sentences in a cluster, classify the cluster as:
"Good" or "Bad". Only provide the label without any additional content.
Example: input:[sentences] output:[label]
input: { [sentences] } output:
```
```
Cluster Naming — You are a helpful assistant for sentence clustering. Based on the relevancy
and common points of the following sentences in a cluster, summarize the cluster with an
"Action-Objective" label. Only provide the label without any additional content.
Example: input:[sentences] output:[label]
input: { [sentences] } output:
```

### 4.3 The LLM-ITL algorithm (Algorithm 1) — relevant to P5/P6

```
S⁽⁰⁾ = all unique sentences; C = ∅; t = 0
while |S⁽ᵗ⁾|/|S| > ε and t < T_max:
    E⁽ᵗ⁾ = embed(S⁽ᵗ⁾)
    for each candidate n_i in N:                      # search space of cluster counts
        C_{n_i}⁽ᵗ⁾ = cluster(E⁽ᵗ⁾, n_i)
        g_{n_i}⁽ᵗ⁾ = [M_eval(C_1), …, M_eval(C_{n_i})]
    n* = argmax_{n_i}  (#good) / (#bad + 1)           # local search heuristic
    C_good⁽ᵗ⁾ = good clusters at n*
    C ← C ∪ C_good⁽ᵗ⁾ ;  S⁽ᵗ⁺¹⁾ = S⁽ᵗ⁾ \ ∪C_good⁽ᵗ⁾   # peel off; recluster the residue
```
**"Peel the coherent clusters, recluster the residue"** is a very practical alternative to fixing one global K. It also gives you a per-iteration `goodness = #good/#bad` that is invariant to K — a genuinely useful K-selection signal for your P5 triangulation.

### 4.4 Post-correction merge (relevant to P8 governance merges)

1. Name every cluster with `M_name` → label ℓ_k; embed and L2-normalize → **l_k on the unit hypersphere**.
2. Build affinity graph: edge if **geodesic distance `arccos(⟨l_i, l_j⟩) < θ`, with θ = 0.8**.
3. Probabilistic gate via a mixture of **von Mises–Fisher** distributions: retain edge only if `P(same | l_i, l_j) = Σ_m π_m p(l_i|μ_m,κ_m) p(l_j|μ_m,κ_m) > τ`, with **τ = 0.7**, π_m = 1/K.
4. **Merge connected components**; re-name each merged cluster.

Key insight: **merge on *label* embeddings, not member embeddings** — much cheaper and much more robust than pairwise LLM queries over members, and it directly implements your "merge as lookup-table remap".

### 4.5 Results & metrics

- Their `goodness_final = #good / #total` (evaluated by a *different* LLM evaluator than the in-loop one, to avoid self-grading) — adopt this hygiene rule.
- NMI 0.8420 → 0.8679 (+role) → **0.8826 (+role+merge)**, goodness 97.6%. Context-aware approach gave **+4.48% NMI** without harming goodness.
- Notable: **ClusterLLM's pairwise-constraint fine-tuning was ineffective on their dataset because there were too many clusters (1,507)** — another warning that triplet fine-tuning does not scale to very fine taxonomies.
- Keyphrase expansion (Viswanathan et al.) raised NMI but *lowered* goodness — **NMI and human-perceived coherence can move in opposite directions.** Report both.
- Dataset: 100k+ real Chinese customer-service calls → 8,184 dialogues → 69,839 sentences → 55,085 unique → **1,507 human-annotated intent clusters** (885 domain-specific, 622 out-of-domain). Annotation protocol: K-means at **n=2000** as a starting point, 15 experts label each cluster Good/Bad, then name the good ones, then reassign members of bad clusters; separate 10-expert verification pass with consensus resolution.

---

## 5. k-LLMmeans / k-NLPmeans ("Summaries as Centroids", ICLR 2026) — arXiv:2502.09667

- **Core idea:** run vanilla k-means, but **every `l` iterations replace the numeric centroid update with `μ_j = Embedding(summary(cluster_j))`.** Between summary steps it's exactly k-means, so convergence properties and the objective are preserved; a bad summary just falls back toward a normal local optimum.
- **k-NLPmeans (LLM-free):** summary = top-q sentences by (a) centroid similarity, (b) TextRank on the sentence-similarity graph, or (c) LSA/SVD in embedding space. **q = 5.**
- **k-LLMmeans:** `μ_j = Embedding(f_LLM(prompt(I, sample of m docs)))`. **Sample the m documents with k-means++ sampling over the cluster's embeddings** (not uniform) — they show this materially improves the summary. Few-shot variant uses **m = 10**.
- Static experiments: 120 centroid-update iterations; `single` variant = 1 summary step (l=60), `multiple` = 5 steps (l=20). Embedders: DistilBERT, e5-large, S-BERT, text-embedding-3-small. LLMs: GPT-3.5-turbo, GPT-4o, Llama-3.3, Claude-3.7, DeepSeek-V3. Datasets: Bank77, CLINC, GoEmo, MASSIVE (D and I). 5 seeds.
- Example instruction I (Bank77): `"The following is a cluster of online banking questions. Write a single question that represents the cluster concisely."`
- **Mini-batch variant** for streaming: process batches sequentially, incremental weighted centroid update — this is your P12 drift/maintenance primitive.
- **Cost (their measurements):** k-LLMmeans, 5 summarization rounds, GPT-4o + text-embedding-3-small: **<$1 and ~1 minute** on a laptop for a static benchmark; **$2.50 / ~8 min** for the StackOverflow-scale dataset. Competing LLM-heavy methods: **$18–$25 and >40 min** on identical hardware/API. LLM calls per summarization step = **k** (one per cluster), **independent of N**.
- Benefit for you: the textual centroid is an **auditable, versionable cluster prototype** you can diff across quarterly reruns for drift detection.

---

## 6. TopicGPT (NAACL'24) — arXiv:2311.01449 — the best published **merge/dedup** and **assignment-with-evidence** recipe

### 6.1 Algorithm
- **Generate:** for each document, given the *current* topic list S (seeded with just **2 hand-written example topics** — they need not be domain-representative), the model either reuses an existing topic or adds a new one. Topic format = **`Label: one-sentence broad description`** with a level marker.
- **Refine:** (a) embed topic strings with Sentence-Transformers, find pairs with **cosine ≥ 0.5**, feed **5 pairs at a time** to a merge prompt; (b) drop topics whose generation frequency is below a removal threshold.
- **Assign:** give the whole hierarchy + 2–3 examples + the document; return **label + document-specific reasoning + a verbatim supporting quote**. The quote makes assignments auditable and is a cheap hallucination detector (quote must be a substring of the doc).
- **Self-correct:** parser flags invalid/"None"/hallucinated assignments; re-prompt with the error type.
- **Hierarchy:** re-run generation within each top-level topic's documents, requiring the model to return supporting document ids for each subtopic.

### 6.2 Sample size — empirically grounded, use this for P2
- They sampled **1,000 (Bills) / 1,100 (Wiki)** documents for generation out of 32k/14k corpora.
- **Stopping rule: stop when no new topics survive refinement for ~200 consecutive documents.** In both corpora, **the number of *refined* topics plateaus after ~600 documents**, while the raw generated count keeps climbing. There is an initial "topic drought" then a burst then saturation.
- Config: GPT-4 for generation, GPT-3.5-turbo for assignment, `max_tokens=300`, `temperature=0`, `top_p=0`.
- **GPT-3.5 and Mistral could not follow the generation format**: they produced 151 and **1,418** topics respectively — overly specific and unusable. *Use your strongest model for generation.*

### 6.3 Results
- Harmonic mean purity P₁: **TopicGPT 0.74 (Wiki) / 0.57 (Bills)** vs LDA 0.64/0.52, BERTopic 0.58/0.39, SeededLDA 0.62/0.52.
- **Human-judged misaligned-topic proportion: LDA 62.4%, TopicGPT unrefined 38.7%, TopicGPT refined 30.3%.** Refinement consistently removes out-of-scope and repeated topics but can drop genuinely rare topics (it dropped "Culture", 23/32,661 docs) — **so gate the frequency filter on business importance, not just count.**

### 6.4 TopicGPT PROMPTS — VERBATIM from github.com/chtmp223/topicGPT

**Generation (`prompt/generation_1.txt`)** — abridged; full text has two worked examples:
```
You will receive a document and a set of top-level topics from a topic hierarchy. Your task is to
identify generalizable topics within the document that can act as top-level topics in the hierarchy.
If any relevant topics are missing from the provided set, please add them. Otherwise, output the
existing top-level topics as identified in the document.

[Top-level topics]
{Topics}

[Examples]
Example 1: Adding "[1] Agriculture"
Document: <...>
Your response: [1] Agriculture: Mentions policies relating to agricultural practices and products.

Example 2: Duplicate "[1] Trade", returning the existing topic
Document: <...>
Your response: [1] Trade: Mentions the exchange of capital, goods, and services.

[Instructions]
Step 1: Determine topics mentioned in the document.
- The topic labels must be as GENERALIZABLE as possible. They must not be document-specific.
- The topics must reflect a SINGLE topic instead of a combination of topics.
- The new topics must have a level number, a short general label, and a topic description.
- The topics must be broad enough to accommodate future subtopics.
Step 2: Perform ONE of the following operations:
1. If there are already duplicates or relevant topics in the hierarchy, output those topics and stop here.
2. If the document contains no topic, return "None".
3. Otherwise, add your topic as a top-level topic. Stop here and output the added topic(s). DO NOT add any additional levels.

[Document]
{Document}

Please ONLY return the relevant or modified topics at the top level in the hierarchy. Your response
should be in the following format:
[Topic Level] Topic Label: Topic Description
Your response:
```

**Refinement / merge-dedup (`prompt/refinement.txt`) — VERBATIM, this is your P8 merge prompt skeleton:**
```
You will receive a list of topics that belong to the same level of a topic hierarchy. Your task is
to merge topics that are paraphrases or near duplicates of one another. Return "None" if no
modification is needed.

Here are some examples:
[Example 1]
Topic List: <pairs of similar topics>
Your response: <topics being merged into an existing topic>
[Example 2]
<pairs of similar topics>
Your response: <topics being merged into a new topic>

[Rules]
- Each line represents a topic, with a level indicator and a topic label.
- Perform the following operations as many times as needed:
    - Merge relevant topics into a single topic.
    - Do nothing and return "None" if no modification is needed.
- When merging, the output format should contain a level indicator, the updated label and
  description, followed by the original topics.

[Topic List]
{Topics}
Output the modification or "None" where appropriate. Do not output anything else.
[Your response]
```
Note the output contract: **updated label+description followed by the original topics** — that *is* a lookup-table remap row. Parse it straight into your P8 remap table.

**Assignment (`prompt/assignment.txt`) — VERBATIM tail:**
```
[Instructions]
1. Topic labels must be present in the provided topic hierarchy. You MUST NOT make up new topics.
2. The quote must be taken from the document. You MUST NOT make up quotes.
[Document]
{Document}
Double check that your assignment exists in the hierarchy!
Your response should be in the following format:
[Topic Level] Topic Label: Assignment reasoning (Supporting quote)
Your response:
```
`correction.txt` is identical plus a `{Message}` slot holding the parser's error description.

---

## 7. GoalEx / PAS (EMNLP'23) — arXiv:2305.13749 — the rigorous MECE mechanism

**Propose-Assign-Select.**
- **Propose:** prompt a "proposer" with T corpus samples (T maximal such that prompt ≤ **75% of context window**) + the goal + "Generate a list of J' explanations for candidate clusters." **J' = 8 per call, J = 30–50 candidates total** across multiple different random subsets.
- **Assign:** a cheap "assigner" model answers, per (sample, explanation) pair:
  ```
  Predicate: {explanation}. Text: {sample}.
  Is the Predicate true on the Text? Yes or No. When uncertain, output No.
  ```
  → binary assignment matrix `A ∈ {0,1}^{|X|×J}`. **Note the built-in abstention convention: "When uncertain, output No."**
- **Select:** choose K of the J candidate clusters via **ILP**. With `s ∈ {0,1}^J`, `s·1 = K`, `m = A sᵀ` (how many selected clusters cover sample x), minimize
  ```
  f_λ(m_x) = (1 − m_x)      if m_x < 1   ("miss")
             0              if m_x = 1   ("ideal")
             λ(m_x − 1)     if m_x > 1   ("overlap")
  ```
  linearized with auxiliary `a ≥ 1−m`, `a ≥ λ(m−1)`, minimize `a·1`.

**This is a computable, optimizable MECE objective.** λ is literally your exclusivity/overlap knob. Five PAS iterations beat one; λ > 0 beats λ = 0. Use it as an *audit* even if you don't use it to build: run Assign over your final taxonomy on a sample, compute the distribution of `m_x` — the mass at `m_x = 0` is your **coverage gap**, the mass at `m_x ≥ 2` is your **overlap / non-exclusivity**, per-cluster. That is a hard number for your P2 MECE gate.

TopicGPT's authors note the tradeoff: GoalEx's ILP can delete topics that legitimately co-occur; TopicGPT refines on semantics instead. Use ILP as a *diagnostic*, semantic merge as the *action*.

---

## 8. 2025–2026 successors worth knowing

| Paper | arXiv | What to take |
|---|---|---|
| **SPILL** — Selection and Pooling with LLMs | 2503.15351 | Training-free domain adaptation: for each seed utterance, take a distance-based candidate pool, have the LLM select the same-intent ones, pool their embeddings into a refined seed embedding. No fine-tuning; matches methods that do. Frames clustering as **a small-scale selection problem**. |
| **LUMI** — Unsupervised Intent Clustering with Multiple Pseudo-Labels | 2510.14640 | Fixes two flaws of single-pseudo-label methods: (1) one label per text is unstable; (2) binary same/different is too coarse. Generate a **set** of pseudo-labels per text, mean-pool their embeddings with the text embedding, then pool each text with its neighbours weighted by **degree of shared labels**. **Does not require estimating K during embedding refinement.** |
| **NILC** — New Intents with LLM-assisted Clustering | 2511.05913 | Iterative: LLM creates **additional semantic centroids** to enrich Euclidean centroids; LLM **rewrites hard/terse/ambiguous samples** for cluster correction; semi-supervised injection via **seeding** and **soft must-links**. Six benchmarks. |
| **LLMEdgeRefine** (EMNLP'24) | aclanthology 2024.emnlp-main.1025 | **Super-points** to absorb outliers; iteratively reassign **edge/boundary points** with the LLM. Directly maps to your P6 "reassign" refinement operator. |
| **Bag-of-Texts / TWIST** | 2510.06747 | Converts LLM same/different judgments into a representation where texts start **equidistant**, avoiding the base embedder's distance geometry entirely. Model-agnostic, works with small LLMs, no K needed. |
| **Cluster-R1** | 2603.23518 | Trains **large reasoning models as clustering agents** that both follow an instruction *and* infer K. Benchmark **ReasonCluster** (28 tasks incl. legal/financial). Signals where the field is heading in 2026. |
| **Iterative Topic Taxonomy Induction with LLMs** (electoral ads) | 2510.15125 | End-to-end unsupervised clustering + prompt-based iterative taxonomy construction, **no seed set, no domain expertise**; human eval says structured iterative labeling gives more consistent/interpretable labels than baselines. A validated TnT-LLM-style pipeline outside chat logs. |
| **TextClusterLab** | 2606.28328 | LLM-driven **synthetic clustering-dataset generator** with controllable class imbalance, intra-cluster compactness, inter-cluster diversity + a benchmark for whether a dataset is even *suitable* for clustering evaluation. **Use this to unit-test your P4/P5 battery deterministically.** |
| **Chain-of-Layer** | 2402.07386 | Layer-by-layer top-down taxonomy induction from an entity set + an **Ensemble-based Ranking Filter** to suppress hallucinated nodes. Relevant if you ever induce taxonomy from mined template-group heads (P1). |

---

## 9. Annotation quality: agreement statistics done right

### 9.1 Choose the coefficient deliberately

| Statistic | Use when | Formula / note |
|---|---|---|
| **Cohen's κ** | exactly 2 raters, nominal | `κ = (p_o − p_e)/(1 − p_e)`, `p_e` from each rater's own marginals |
| **Fleiss' κ** | ≥3 raters, but **raters may differ per item** | `p_e` from pooled marginals |
| **Krippendorff's α** | any #raters, **missing data allowed**, nominal/ordinal/interval/ratio, and rater-count may vary per item | `α = 1 − D_o/D_e`; the general-purpose default. `pip install krippendorff==0.8.2` |
| **Weighted κ** | ordinal labels (e.g. confidence tiers) | linear or quadratic weights |
| **PABAK** | as a *companion* when prevalence is skewed | `PABAK = 2p_o − 1` (Byrt, Bishop & Carlin 1993, *J Clin Epidemiol* 46(5):423–429) |

**Krippendorff's own thresholds:** α ≥ **0.800** = reliable; **0.667 ≤ α < 0.800** = tentative conclusions only; below 0.667 = not usable. Landis & Koch (1977) κ bands (0.41–0.60 moderate, 0.61–0.80 substantial, >0.80 almost perfect) are *conventional but arbitrary* — say so in your report.

### 9.2 The two kappa paradoxes (Feinstein & Cicchetti 1990, *J Clin Epidemiol* 43(6):543–549; Cicchetti & Feinstein 1990, ibid. 551–558)

1. **High agreement can coexist with low κ** when one class dominates (prevalence effect). With 95% raw agreement on a 95/5 split, κ can fall near 0.
2. **Unbalanced marginals can produce *higher* κ than balanced marginals** (bias effect) — κ can be gamed.

**Consequences for your pipeline (this is the big one):**
- **Your playbook's "Cohen's κ ≥ 0.9" gate is, on the published evidence, unattainable for a genuine intent taxonomy.** TnT-LLM's expert annotators on a 10-class intent taxonomy over real chat logs got Fleiss κ = **0.553** and pairwise Cohen κ = **0.559**; on a 25-class domain taxonomy, **0.624**. Their *taxonomy-quality* judgments were 0.38–0.48. Nobody in this literature reports 0.9 on an open-ended intent task. A κ of 0.9 in your pipeline would be evidence of one of: (a) a trivially easy or highly skewed label distribution (κ paradox), (b) leakage/anchoring between annotators, or (c) computing κ post-adjudication (circular).
- **Recommended replacement gate** (defensible and achievable):
  - Report **all four**: raw percent agreement `p_o`, Cohen/Fleiss κ, Krippendorff α, and PABAK — plus the **label prevalence vector**, so a reader can see whether κ is deflated.
  - **Per-class agreement**: for each label, a 1-vs-rest κ and per-class precision/recall between annotators. A single global κ hides that 3 of 12 classes are doing all the damage.
  - **Gate on**: α ≥ 0.67 to proceed with caveats, α ≥ 0.80 to freeze gold labels; **plus** "no single label pair accounts for >25% of disagreements after two refinement rounds"; **plus** post-adjudication referee-vs-consensus κ ≥ 0.9 (this *is* achievable and is what a referee gate should measure).
  - Track **κ trajectory across refinement rounds** as the real success signal, not a one-shot threshold.
- **Bootstrap CIs on κ, always** (nonparametric bootstrap over items, 2000 resamples). Approximate analytic SE (Fleiss 1969): `SE(κ) ≈ sqrt(p_o(1−p_o) / (n(1−p_e)²))`. Concrete numbers, `p_e = 0.2`:

  | true κ | n=100 | n=200 | n=300 | n=500 | n=1000 |
  |---|---|---|---|---|---|
  | 0.9 | ±0.066 | ±0.047 | ±0.038 | ±0.030 | ±0.021 |
  | 0.8 | ±0.090 | ±0.064 | ±0.052 | ±0.040 | ±0.028 |
  | 0.7 | ±0.105 | ±0.074 | ±0.060 | ±0.047 | ±0.033 |

  → **To *certify* κ ≥ 0.8 (i.e. lower CI bound above 0.8) you need n ≥ ~300 doubly-annotated items minimum, 500 comfortably.** A 100-item pilot cannot distinguish κ=0.7 from κ=0.85.

### 9.3 LLM-as-annotator reliability — 2025/2026 state of the art

- **κ deflation is universal for LLM judges.** *Reliability without Validity* (arXiv:2606.19544): 21 judges, 9 providers, MT-Bench/JudgeBench/RewardBench, 118 runs, ~541k judgments. **Exact-match agreement overstates chance-corrected agreement by 33–41 percentage points on MT-Bench, in every model tested, including April-2026 frontier models.** Judge rankings move by up to **14 positions** across benchmarks. **The "consistency–bias paradox": test–retest reliability >0.95 coexists with position bias >0.10 in two production-deployed judges.** They publish a "Minimum Viable Validation Protocol." **Never report LLM-annotator quality as raw agreement.**
- **Bias magnitudes** (*Judging the Judges*, arXiv:2604.23178; 9 debiasing strategies × 5 judges × 3 benchmarks): **style bias is dominant (0.10–0.76, markdown over plain prose)**, far exceeding position bias (≤0.04) in that setup; verbosity bias is heterogeneous and model-specific (Gemini/Llama prefer long +0.24…+0.44; Claude prefers concise −0.12; GPT-4o neutral). Headline: **a mid-tier model with the right debiasing beat frontier judges — Gemini 2.5 Flash + "Combined Budget" strategy: 71.0% agreement, κ=0.549, ~$0.001/eval, ~15× cheaper than the best frontier setup (Claude Sonnet 4, 69.5%, ~$0.015).**
- **Prompt wording is a measurement instrument, not a constant.** *Inter-Prompt Reliability* (arXiv:2604.16413) formalizes IPR: run **semantically equivalent but linguistically varied** prompts, measure **Pairwise Agreement Rate** and its distribution. Finding: LLM annotation is highly stochastic on **interpretative** tasks, much more stable on **knowledge-anchored** ones; **majority voting across prompt paraphrases significantly improves reproducibility and reduces variance.** → For P2 labeling, run 3 paraphrased instruction variants and vote; report IPR alongside κ.
- **Wisdom-of-the-crowd across models works.** arXiv:2602.11962 (100k election posts, 6 LLMs, 5 categories, validated against 34 crowdworkers): **LLM–LLM agreement patterns comparable to human–human, with LLMs showing higher internal consistency**; multi-model aggregation produced the released gold set. Also documents that **human annotator demographics/ideology systematically shape labels** — a reminder that "human gold" is itself a random variable.
- **Disagreement is signal, not noise.** DiZiNER (arXiv:2604.15866): multiple heterogeneous LLMs annotate the same texts; **a supervisor model analyzes the inter-model disagreements and rewrites the task instructions.** Zero-shot SOTA on 14/18 NER benchmarks, +8.0 F1, and it **outperforms its own supervisor (GPT-5 mini)** — so the gain comes from the disagreement-driven refinement loop, not model capacity. **Pairwise inter-model agreement correlates strongly with downstream performance** → use it as an unsupervised proxy for taxonomy quality when you have no gold. **This is exactly the referee agent you want in P2: not just a tie-breaker but an instruction/definition rewriter.**

### 9.4 Confidence elicitation — and why to distrust it

- **Verbalized confidence is better calibrated than conditional probabilities for RLHF'd models** (arXiv:2305.14975, "Just Ask for Calibration") — but that finding is fragile.
- **The score granularity gap** (arXiv:2606.22179): across 7 confidence constructions × 25 model-dataset pairs (9 LLMs, 3 benchmarks): single-shot verbalized confidence **ranks** cases surprisingly well but **takes only a handful of distinct values** — so you get only a few coarse operating thresholds no matter how good the ranking. **Multi-query aggregation helps weak models but can degrade already-strong ones.** → If you need fine-grained margin routing (your P10), **do not** route on a verbalized 0–100 score; use the distilled classifier's calibrated probability or an ensemble-vote fraction.
- **Verbalized confidence measures commitment, not correctness** (arXiv:2606.29490): in a two-stage answer-then-commit/abstain paradigm across 8 models, **verbal confidence predicted the commit/abstain decision substantially better than it predicted correctness**; calibrated token log-probs showed the opposite. After removing the shared variance with log-probs, the verbal residual's link to correctness fell to near chance. Mechanistically, the post-answer state already encoded the future abstention decision, roughly **orthogonal** to correctness in activation space.
- **Protocol sensitivity** (arXiv:2605.27752): whether verbalized confidence beats token likelihood **flips** depending on which answer string is scored and under which prompt — in 4/12 settings under ECE and 9/12 under AUROC. **You must fix and document: which answer is scored, under which prompt, with which estimator.**
- **Practical rule:** treat elicited confidence as an **abstention/routing signal**, validate it empirically on your own held-out gold, and never report it as a probability.

### 9.5 Position / order / selection bias — mitigation menu

- **Selection bias in MCQ-style label choice** (arXiv:2309.03882, 20 LLMs, 3 benchmarks): models a-priori favor specific option **IDs** (token bias on "A"/"B"/"1"). Mitigation **PriDe**: estimate the option-ID prior by permuting option contents on a small subset of items, then debias the rest — label-free, inference-time, cheap, transferable.
- **Calibration framework** (arXiv:2305.17926, "LLMs are not Fair Evaluators"): (1) **Multiple Evidence Calibration** — force the model to write evaluation evidence *before* the rating; (2) **Balanced Position Calibration** — aggregate over both orders; (3) human-in-the-loop for the residual.
- **PORTIA** (arXiv:2310.01432): split candidates into aligned segments and interleave them in one prompt; markedly improves consistency across 6 LLMs on 11,520 pairs.
- **TnT-LLM's own practice:** randomize option positions **and average over multiple runs** for every single-choice/pairwise evaluation. They explicitly note position bias is worse when the taxonomy is low quality (i.e. exactly when you most need the signal).
- **What TnT-LLM did *not* need to defend against but you do:** in P7 blind naming, the *order of member queries* in the prompt anchors the label. Shuffle member order per naming agent and per repetition.

---

## 10. Blind naming & anti-anchoring protocol for P7

There is little dedicated NLP literature on "blind cluster naming"; the relevant evidence is the LLM-bias literature above plus TnT-LLM's blinding practice. Synthesize a protocol:

1. **Information hygiene.** The naming agent's context must contain **only**: (a) the naming convention spec, (b) K sampled member queries, (c) optionally, c-TF-IDF discriminative keywords, (d) optionally, **negative exemplars from the nearest sibling cluster** (this is the biggest quality lever — it forces contrastive naming). It must **never** contain: existing labels, cluster ids that encode order, upstream L1 names, or the taxonomy from P2. Enforce with a **context allowlist assertion in code**, not a prompt instruction.
2. **Member sampling must be deterministic and representative.** Use k-means++ sampling within the cluster (per k-LLMmeans) or stratify: `n_core` nearest-to-centroid + `n_mid` at median distance + `n_edge` boundary points. Include edge points deliberately — they expose mixed clusters. TnT-LLM/BERTopic use 4–20 docs; Dial-In uses **20 sentences per cluster**; that is a good default.
3. **Order randomization.** Shuffle members per agent and per repeat; different random seeds per naming agent.
4. **5 independent naming agents** (your design): vary **model family**, **temperature**, **member sample**, and **prompt paraphrase** — vary all four, not just seed, so the ensemble estimates the right variance (per IPR, arXiv:2604.16413).
5. **Aggregation.** Don't majority-vote on strings. Embed the 5 candidate names, take the **medoid** as the label and report **mean pairwise cosine among the 5 as a naming-confidence score**. Low agreement ⇒ flag for the tree-audit agent. (This mirrors Dial-In's label-embedding geometry.)
6. **Tree-audit agent** sees the *whole* label set (it must, to detect duplicates/overlap) but never the member text — it audits names+descriptions only, exactly like TnT-LLM's **review** prompt (which deliberately has no data block).
7. **Blind evaluation of the names themselves.** Use TnT-LLM's pairwise forced choice: show a query + its cluster's name + a random sibling cluster's name, randomize positions, average over ≥3 runs, report hit rate with 95% CI. Add a human sample (see §11).

---

## 11. Validating an LLM-built taxonomy

### 11.1 Coverage
- Add `Other/Undefined` **only to the assignment prompt**, never to the taxonomy (both TnT-LLM and its prompts explicitly forbid "Other/General/Unclear/Miscellaneous/Undefined" as taxonomy entries). Coverage = 1 − P(Other). TnT-LLM's bar: **>99.5%**.
- Also measure **saturation** à la TopicGPT: plot refined-topic count vs documents processed; require a plateau (no new surviving topic for ≥200 docs; theirs plateaued at ~600).
- And measure **tail coverage**: coverage restricted to queries in the bottom decile by frequency and to each dataset stratum (time slice, language, device, vertical).

### 11.2 MECE
- **Exclusivity (E):** run GoalEx's Assign matrix on a sample with all L1 descriptions as predicates; `P(m_x ≥ 2)` is the measured overlap rate; report the top overlapping label pairs. Target: overlap ≤ 5%.
- **Exhaustiveness (M):** `P(m_x = 0)` from the same matrix = miss rate, cross-checked against the "Other" rate.
- **Pairwise discriminability:** for each label pair, sample 20 queries assigned to each and ask a judge to re-assign blind; a confusion matrix over pairs. Any pair with >20% cross-assignment is a merge or a definition-sharpening candidate.
- **Confusion-driven refinement loop** (TnT-LLM's Appendix B.2.1 finding): the top-3 confusion pairs get explicit disambiguation clauses added to both descriptions, then re-measure. Iterate until no pair dominates.

### 11.3 Human spot-check sample sizes (Wilson score intervals, 95%)

| n | if observed accuracy = 0.90 | = 0.95 | = 0.80 |
|---|---|---|---|
| 50 | [0.786, 0.957] (±0.085) | [0.865, 0.989] | [0.670, 0.888] |
| 100 | [0.826, 0.945] (±0.060) | [0.888, 0.978] | [0.711, 0.867] |
| 200 | [0.851, 0.934] (±0.042) | [0.910, 0.973] | [0.739, 0.850] |
| 300 | [0.861, 0.929] (±0.034) | [0.919, 0.969] | [0.751, 0.841] |
| 384 | [0.867, 0.927] (±0.030) | [0.924, 0.968] | [0.757, 0.836] |
| 500 | [0.871, 0.923] (±0.026) | [0.927, 0.966] | [0.763, 0.833] |
| 1000 | [0.880, 0.917] (±0.019) | [0.935, 0.962] | [0.774, 0.824] |

Required n for a target margin of error: **p≈0.90, ±5pp → n ≥ 139; ±3pp → n ≥ 385. p≈0.95, ±3pp → n ≥ 203. Worst case p=0.5, ±5pp → n ≥ 385; ±3pp → n ≥ 1068.**

**Recommended P2/P7 budget** (matches published practice: TnT-LLM used 200 for taxonomy eval and 400 for label eval):
- Taxonomy quality (accuracy + relevance): **n = 200–300**, 3 raters each.
- Gold label set for classifier eval: **n = 400–500**, 3 raters + tie-breaker, stratified by label so every class has ≥20 items (otherwise per-class recall CIs are useless).
- Per-cluster naming spot-check: **n = 30 members per cluster** for the top-20 clusters by volume, then a random 10 clusters from the tail.
- Always **stratify by frequency decile and by any known segment** (language, device, vertical) and report per-stratum CIs — an aggregate 0.90 hides a 0.65 on the tail.

### 11.4 Downstream utility (the criterion TnT-LLM insists on)
A taxonomy is not validated until a **distilled classifier trained on its labels** hits target accuracy. Report the distilled-vs-LLM gap with a **paired t-test** (TnT-LLM's protocol) and by segment (their ada2+LR was +2.3% English / −2.7% non-English; Instructor-XL was +8.4%/−9.9% — segment gaps are the real risk).

---

## 12. Cluster-naming quality: what makes a good label, how to evaluate it, how to detect mixed clusters

### 12.1 What a good label is (synthesis of TnT-LLM / Dial-In / TopicGPT / BERTopic)
- **Structured convention, enforced.** Pick one and never deviate: `action-object` (Dial-In), `Label: one-sentence broad description` (TopicGPT), or `name (≤N words) + description (≤M words)` (TnT-LLM). Word budgets are parameters (`cluster_name_length`, `cluster_description_length`) and are checked by guardrails.
- **Contrastive, not generic.** TnT-LLM verbatim: *"Use only phrases that are specific to each category and avoid those that are common to all categories"* and *"Description differentiates one category from another."*
- **Operationally sufficient.** *"Name and description can accurately and consistently classify new data points without ambiguity"* — the label must function as an annotation instruction. Embed contrastive markers and one concrete example (see the TnT-LLM label style in §1.6).
- **Generalizable, single-concept, headroom for subtopics** (TopicGPT verbatim): not document-specific, one topic not a conjunction, broad enough to host L2 children.
- **Abstractive, not extractive** — labels describe the corpus but need not be substrings of it (TnT-LLM's stated advantage over term-extraction taxonomies).
- **Banned vocabulary:** "Other", "General", "Unclear", "Miscellaneous", "Undefined" — hard-fail the guardrail on these.

### 12.2 Evaluating names
- **Pairwise forced-choice hit rate** (TnT-LLM §4.1): the gold standard, works for both humans and LLM raters, randomize positions, average over runs, report 95% CI. Validate the LLM rater against a 200-item human sample *first*, then extrapolate to the full multilingual corpus. GPT-4-class raters reached κ≈0.56–0.58 vs human consensus on this task — higher than human-human.
- **Use-case relevance** binary rating (TnT-LLM): guards against on-topic-but-useless labels.
- **Human misalignment audit** (TopicGPT §5): three annotators map each produced label to ground-truth categories and mark it *aligned / out-of-scope / repeated / missing*; report the proportion misaligned. This decomposition (out-of-scope vs repeated vs missing) is far more actionable than a single score.
- **LLM raters correlate with humans better than classical automated topic metrics** (arXiv:2305.12152) — so LLM-based coherence rating is defensible, but *only after* you measure its κ against a human sample on your own data.
- **Naming-ensemble agreement** as a cheap intrinsic score (see §10.5).

### 12.3 Automatically detecting "mixed" clusters — a layered detector
1. **Fine-tuned binary coherence evaluator** (Dial-In). Fine-tune a 7B on **~1,500–2,000** human good/bad cluster judgments: **96.25% accuracy, beating GPT-4o's 94.17%, vs 75.63% for the same base model unfine-tuned.** Highest-value single investment for P6/P7 quality. Bootstrap the training set by **perturbation**: take known-good clusters and inject k% members from another cluster to synthesize "bad" ones (that is exactly how they built the English-benchmark version — Llama-7b fine-tuned on 800 clusters, 200 per dataset, with bad clusters produced by perturbation).
2. **Label-variance test (IDAS).** Generate a per-*member* label independently (not a cluster-level label), then measure the entropy/modal-share of the resulting label multiset. High modal share ⇒ coherent. Also distinguish the two failure modes: high lexical/structural overlap with different heads ⇒ *too specific* (benign); one label spanning members whose nearest neighbours sit in other clusters ⇒ *too general* (harmful merge).
3. **Naming-ensemble disagreement.** Mean pairwise cosine among the 5 blind names < threshold ⇒ mixed.
4. **Assign-back consistency.** Re-classify a sample of members with the *final* taxonomy using the assignment prompt; the fraction not returning to their own cluster is a direct impurity estimate — and reusing the same code/sample/seed makes it comparable across clusters (your P9 requirement).
5. **Geometric signals as a cheap prefilter only:** silhouette, intra-cluster mean distance, distance-to-nearest-other-centroid ratio, bimodality of the intra-cluster distance histogram. Dial-In's whole premise is that these are *insufficient* — cosine similarity fails both directions (same intent/different wording, different intent/similar wording). Use geometry to rank candidates, LLM to adjudicate.
6. **Edge-point rate** (LLMEdgeRefine): fraction of members closer to a foreign centroid than their own, after super-point outlier absorption.

### 12.4 BERTopic as a reference implementation (v0.17.4, requires Python ≥3.10)
- `bertopic.representation`: `KeyBERTInspired`, `MaximalMarginalRelevance`, `PartOfSpeech`, `TextGeneration`, `OpenAI`, `LangChain`, `Cohere`, `LiteLLM`, `ZeroShotClassification`.
- Prompt tags: `[DOCUMENTS]` and `[KEYWORDS]`. **Defaults: 4 most representative documents** (by c-TF-IDF similarity to the topic's c-TF-IDF vector). Tunables: `nr_docs` (raise it if context allows), `diversity` (0–1; **"a value of 0.1 already does wonders"** to avoid near-duplicate exemplars), `doc_length` + `tokenizer` (`'char'|'whitespace'|'vectorizer'|callable`) for truncation, `delay_in_seconds` for rate limits.
- Introspection: `representation_model.default_prompt_` and `topic_model.representation_model.prompts_`.
- **Their default chat prompt is a good two-shot naming template — VERBATIM:**
```
You will extract a short topic label from given documents and keywords.
Here are two examples of topics you created before:

# Example 1
Sample texts from this topic:
- Traditional diets in most cultures were primarily plant-based ...
Keywords: meat beef eat eating emissions steak food health processed chicken
topic: Environmental impacts of eating meat

# Example 2
Sample texts from this topic:
- I have ordered the product weeks ago but it still has not arrived!
...
Keywords: deliver weeks product shipping long delivery received arrived arrive week
topic: Shipping and delivery issues

# Your task
Sample texts from this topic:
[DOCUMENTS]

Keywords: [KEYWORDS]

Based on the information above, extract a short topic label (three words at most) in the following format:
topic: <topic_label>
```
System prompt: `"You are an assistant that extracts high-level topics from texts."`
- **Note the `[KEYWORDS]` channel.** For query mining, feed **c-TF-IDF discriminative unigrams/bigrams** alongside the sampled queries — it is the cheapest available "what makes this cluster different" signal and it is compatible with blind naming (keywords are derived from members, not from labels).

---

## 13. Structured LLM labeling at scale — operational recipe

**Batching.** Batch prompting (arXiv:2301.08721) cuts token+time cost near-inverse-linearly in batch size; benefits hold for chat models. Production evidence (PromptPack, arXiv:2607.20528): in a live ad-annotation system, **redundant system instructions were 94% of billed input tokens**; in-context batching at **batch size 20** with a shared system prompt + **strict XML structural envelope** + an **output correction layer** cut LLM cost **89%** and raised throughput **2.5×** while fully preserving downstream AUC. A separate finding (arXiv:2511.04108): batching *suppresses overthinking* in reasoning models — 76% fewer reasoning tokens (2,950→710) across 13 benchmarks with preserved or improved accuracy.
**Practical rule: batch 10–25 items with a shared system prompt, a strict XML envelope with per-item ids echoed back, and a correction/repair pass. Verify on your own gold that accuracy at batch size B matches batch size 1 — batching interacts with task complexity, and safety/quality failures in batches are documented (arXiv:2608.02681, arXiv:2503.15551).** Always echo item ids; never rely on positional alignment.

**Few-shot selection.** kNN/KATE retrieval of similar labeled examples beats random (IDAS, following Liu et al. 2022); **n = 8** demonstrations is the plateau (IDAS, citing Min et al. 2022 and Lyu et al. 2022). For labeling under a taxonomy, retrieve demonstrations **stratified across the labels most confusable with the query's nearest labels**, not just nearest neighbours — otherwise you reinforce the majority class.

**Self-consistency.** Two distinct axes:
- *Sampling self-consistency* (arXiv:2203.11171): sample k reasoning paths at T>0, marginalize, majority vote.
- *Prompt-paraphrase self-consistency* (IPR, arXiv:2604.16413): vote across semantically equivalent instruction paraphrases; empirically improves reproducibility and reduces variance, especially on interpretative tasks.
Use **3 paraphrases × 1 sample** rather than **1 paraphrase × 3 samples** for classification — it addresses the larger variance component. Report the vote-fraction as your routing confidence (this is a many-valued score, unlike verbalized confidence — see §9.4).

**Abstention.** GoalEx's `"When uncertain, output No."`; TnT-LLM's assignment-time `Other` bucket; a `<uncertain>` tag with a forced reason. Route abstentions and low-vote-fraction items to (a) the referee agent, then (b) human review — this **is** your active-learning acquisition pool.

**Determinism & provenance.** Fix `temperature=0`, `top_p` (TnT-LLM used 0.5), `seed` where supported; pin model **snapshot ids**, not aliases; hash and store `(prompt_template_version, model_id, params, input_hash) → output` so P11's "deterministic sample display" and P12's quarterly reruns are actually reproducible. Note TopicGPT's caveat: **OpenAI models retain some non-determinism even at greedy decoding** — so cache outputs as artifacts, don't assume regeneration reproduces them.

**Guardrails & retries.** TnT-LLM's exact policy: per-prompt-type test suite (parseable? right language? satisfies verifiable constraint?), **max 5 retries, temperature +0.1 per retry**, drop items that never pass and log them as a first-class metric.

---

## 14. Active learning for the P2 gold-label loop — practical 2025/2026 recipes

- **Use frozen high-quality LLM/MTEB embeddings + a cheap head**, not iterative fine-tuning of a large backbone (arXiv:2506.01992, benchmark over 5 MTEB-top models + 2 baselines × 10 text-classification tasks). Findings you can act on:
  - **Cold start: seed the labeled pool by diversity sampling** — this *synergizes* with high-quality embeddings and boosts early-iteration performance. (Concretely: k-means++ or k-center-greedy over the frozen embeddings, one item per cluster.)
  - **The optimal query strategy is sensitive to embedding quality.** Margin sampling is cheap and can spike on specific datasets; **BADGE is more robust across tasks**, and its advantage grows with better embeddings. → Ship **BADGE as the default**, margin as a fast fallback, and evaluate both on your own data.
- **Uncertainty sampling** for multiclass: prefer **margin** (`p₁ − p₂`) over max-entropy or least-confidence for taxonomies with many rare classes.
- **Query-by-committee / disagreement.** Your 2-annotator + 1-referee design *is* a committee. Use **vote entropy** or **KL-to-mean** over the committee (LLM annotator A, LLM annotator B, distilled classifier) to rank the human-review queue. ClusterLLM's entropy-based triplet sampler is the clustering analogue and its authors explicitly note it can be used to acquire *human* annotations when no class prior exists.
- **Batch-mode diversity.** Never take the top-B by uncertainty (they will be near-duplicates). **BADGE** (gradient-embedding + k-means++ seeding) gets uncertainty and diversity in one shot and is the robust default per the 2025 benchmark. Cheaper alternative: cluster the top-5B uncertain items into B clusters and take one per cluster.
- **Cross-task cost efficiency:** arXiv:2502.16892 applies LLMs to AL for cross-task text classification with **no manually labeled data**; arXiv:2406.12114 combines LLM-driven AL with human annotation. Both support the hybrid you want: LLM labels the easy mass, humans see only the acquisition batch.
- **Stopping rule:** stop when the distilled classifier's held-out macro-F1 improvement over the last 3 AL rounds is within the bootstrap CI, or when the referee's edit rate on the acquisition batch drops below ~5%.

---

## 15. Concrete prompt templates for your pipeline (ADAPTED — ready to instantiate)

### 15.1 Taxonomy proposal from a query-log sample
Use the **TnT-LLM generation prompt verbatim** (§1.6b), swapping the Data block for:
```xml
<queries>
  <q><id>{i}</id><text>{query}</text><freq>{count}</freq><example_clicks>{top_urls}</example_clicks></q>
  ...
</queries>
```
and `{use_case}` = e.g. *"Identify the user's search intent in a K-12 education search engine, so that we can route queries to the right vertical and measure unmet need. Intents must describe what the user is trying to accomplish, not the subject matter."* Add to Requirements: `- Weight your categories by the <freq> field: a category must account for a non-trivial share of query volume.` and `- Ignore navigational/brand-name queries unless they form a coherent intent.`

### 15.2 Taxonomy merge / dedup (P8)
Combine TopicGPT's refinement contract with an embedding prefilter:
```
You will receive a list of candidate label pairs from the same level of a taxonomy, pre-selected
because their descriptions are semantically similar. For each pair decide exactly one of:
MERGE (they are paraphrases or one subsumes the other for our use case),
KEEP_BOTH_SHARPEN (genuinely different but the descriptions do not separate them — rewrite both
descriptions so a labeler could tell them apart), or
KEEP_BOTH (already distinct).

Use case: {use_case}
Rules:
- Merge only when a single labeler could not reliably route a query between the two.
- When you MERGE, output the surviving id, a new name (<= {name_len} words) and description
  (<= {desc_len} words), and list every original id that maps into it.
- When you KEEP_BOTH_SHARPEN, the two rewritten descriptions must each contain an explicit
  contrast clause naming the other category.
- Never output "Other", "General", "Miscellaneous", "Unclear" or "Undefined".

[Candidate pairs with volumes]
{pairs}

Output XML only:
<decisions>
  <decision><pair>a,b</pair><action>MERGE|KEEP_BOTH_SHARPEN|KEEP_BOTH</action>
    <survivor_id/><name/><description/><absorbed_ids/><rationale/></decision>
</decisions>
```
Prefilter pairs with **label-embedding cosine (TopicGPT: ≥0.5) or geodesic arccos < 0.8 with a vMF gate at τ=0.7 (Dial-In)**; feed **5 pairs per call** (TopicGPT). The `<absorbed_ids>` output *is* your executed remap table.

### 15.3 Label a query given a taxonomy + adjudication rules
```
You are labeling search queries with a fixed taxonomy. You may not invent labels.

<taxonomy>{taxonomy_xml}</taxonomy>
<adjudication_rules>
1. If a query matches two categories, choose the one describing the user's PRIMARY goal, not the
   surface topic.
2. {pairwise tie-breakers mined from the human-vs-human confusion matrix, e.g.
   "X vs Y: choose X when the query asks for a definition; choose Y when it asks for steps."}
3. If the query is a brand/URL/navigational string, label it {NAV_LABEL} regardless of topic.
4. If no category applies, output OTHER and describe the missing category in <gap>.
</adjudication_rules>

Label each query below. For each: give a one-sentence reason grounded in the query text, then the
label id and name, then a same/different vote among your top-2 candidates.

<queries>{batch_of_15_with_ids}</queries>

Output XML only, one <item> per input id, echoing the id:
<items>
 <item><id/><reason/><label_id/><label_name/><runner_up_id/><decisive_because/><abstain>true|false</abstain></item>
</items>
```
Run **3 paraphrases of the instruction block** and majority-vote; the vote fraction is your routing score. Batch 10–25 with ids echoed.

### 15.4 Adjudicate a disagreement (referee — do more than tie-break)
```
Two independent annotators disagreed on the label for the queries below.

<taxonomy>{taxonomy_xml}</taxonomy>
<adjudication_rules>{rules}</adjudication_rules>
<cases>
 <case><id/><query/><annotator_a_label/><annotator_a_reason/><annotator_b_label/><annotator_b_reason/></case>
</cases>

For each case:
1. State which annotator's reasoning is better grounded in the query text, or that both are wrong.
2. Give the final label, or DEFER_TO_HUMAN if the taxonomy genuinely underdetermines the answer.
3. Classify the ROOT CAUSE: AMBIGUOUS_QUERY | OVERLAPPING_DEFINITIONS | MISSING_CATEGORY |
   RULE_GAP | ANNOTATOR_ERROR.
Then, across all cases, propose at most 3 edits to the taxonomy descriptions or adjudication rules
that would have prevented the majority of these disagreements. Quote the exact current text and the
exact replacement text.

<output>
 <case_decisions>...</case_decisions>
 <proposed_edits><edit><target/><current_text/><replacement_text/><cases_fixed/></edit></proposed_edits>
</output>
```
The `proposed_edits` block is the DiZiNER mechanism (arXiv:2604.15866) — the referee rewrites the instrument, not just the labels. Feed edits back into the next annotation round and re-measure κ.

### 15.5 Blind cluster naming (P7)
```
You will name a group of search queries. You have NOT been told what this group is called and no
existing labels exist. Do not guess at any pre-existing naming scheme.

Naming convention: {action}-{object}, lowercase, hyphenated, each part 1-2 words.
Examples of the convention (from an unrelated domain): inquire-promotion, answer-amount,
provide-location.

<queries_in_this_group>{20 stratified members, shuffled}</queries_in_this_group>
<distinctive_keywords>{c-TF-IDF top terms}</distinctive_keywords>
<queries_in_the_NEAREST_OTHER_group>{10 members of the sibling cluster, shuffled}</queries_in_the_NEAREST_OTHER_group>

Tasks:
1. In <common>, state in one sentence what ALL of the queries in this group are trying to do.
2. In <contrast>, state what distinguishes them from the nearest other group.
3. In <name>, give the label under the convention.
4. In <description>, give <= 25 words that a human annotator could use to route a NEW query into or
   out of this group, including the contrast from step 2.
5. In <coherence>, output GOOD if all queries share one intent, MIXED otherwise; if MIXED, list in
   <outliers> the ids that do not belong.
Do not use the words other, general, miscellaneous, unclear, undefined, various.
```
Run 5 agents (varying model, temperature, member sample, and instruction paraphrase); medoid-select the name; escalate on low agreement or any MIXED vote.

### 15.6 Audit a cluster tree
Use the **TnT-LLM review prompt verbatim** (§1.6d — deliberately data-free) plus a structural pass:
```
<tree>{L1 and L2 names+descriptions+volumes, NO member text}</tree>

Audit and output XML:
1. <duplicates>   pairs at the SAME level that are paraphrases (give ids + evidence).
2. <misplacements> L2 nodes whose description does not entail its L1 parent's description.
3. <granularity>  L1 nodes whose children are inconsistent in specificity relative to siblings
                  elsewhere in the tree.
4. <overlaps>     pairs whose descriptions could both be true of the same query; give a concrete
                  example query that both would claim.
5. <gaps>         plausible intents in this domain absent from the tree.
6. <naming>       nodes violating the convention or using banned vocabulary.
7. <rating_score> 0-100 with <explanation>.
```
Withhold member text from this agent so its judgments are about the *taxonomy artifact*; run a separate data-grounded pass (assign-back consistency) for empirical coverage/overlap.

---

## 16. Version notes (verified on PyPI, August 2026)
`langgraph 1.2.11` · `langchain 1.3.15` · `bertopic 0.17.4` (Python ≥3.10) · `scikit-learn 1.9.0` · `statsmodels 0.14.6` (`statsmodels.stats.inter_rater` for Fleiss κ and `cohens_kappa` with CIs) · `krippendorff 0.8.2`. LangSmith Hub prompt commits are publicly fetchable without auth at `https://api.smith.langchain.com/commits/{owner}/{name}/latest` — pin the returned `commit_hash` in your repo so your prompts are versioned artifacts.


---

## Recommendations carried into the design

- Implement P2 taxonomy induction as a literal TnT-LLM port: minibatch=200, generation prompt at temperature 0.5, update prompt at 0.2 (emitting rating_score + explanation + suggestions + updated_table in one call), review prompt with no data block, XML-tagged outputs, per-prompt guardrail tests, and 5 retries with +0.1 temperature per retry — and pull the five verbatim prompts from LangSmith Hub (wfh/tnt-llm-*) pinned by commit hash.
- Replace the 'Cohen's kappa >= 0.9' gate with a defensible panel — raw agreement, Cohen/Fleiss kappa, Krippendorff's alpha, PABAK, the label prevalence vector, per-class 1-vs-rest kappa, and bootstrap CIs — because published expert agreement on real intent taxonomies is 0.55-0.62, and 0.9 would indicate prevalence skew, anchoring, or a circular post-adjudication computation.
- Size the doubly-annotated gold set at 300-500 items (stratified so every class has >=20), since certifying kappa >= 0.8 needs n >= ~300 and a 100-item pilot cannot distinguish kappa 0.70 from 0.85; use n=200-300 for taxonomy quality eval and Wilson intervals for every reported rate.
- Make the referee agent rewrite the instrument, not just break ties: have it classify each disagreement's root cause (ambiguous query / overlapping definitions / missing category / rule gap / annotator error) and propose exact current-text -> replacement-text edits to label descriptions and adjudication rules, then re-measure agreement per round (the DiZiNER mechanism, arXiv:2604.15866).
- Build P2's confusion-driven refinement loop: compute human-vs-human and LLM-vs-human confusion matrices, identify the top boundary pairs, and write explicit contrastive tie-breaker clauses into both label descriptions — this is the single highest-ROI iteration in TnT-LLM's reported experience.
- Fine-tune a 7B binary cluster-coherence evaluator on ~1,500-2,000 human good/bad cluster judgments (bootstrap the 'bad' class by injecting foreign members into known-good clusters); Dial-In LLM shows this reaches 96.25% vs 94.17% for GPT-4o and 75.63% for the same base model unfine-tuned, and it becomes your automatic mixed-cluster detector, K-selection signal (good/bad ratio), and P6 refinement oracle.
- Adopt an action-object naming convention for P7 (e.g. compare-prices, find-tutorial, check-eligibility), which is both more informative than objective-only labels and gives you a free second facet by grouping on the action verb.
- Enforce blindness in P7 in code, not in the prompt: an explicit context allowlist assertion that the naming agent sees only the convention spec, shuffled stratified member samples, c-TF-IDF keywords, and sibling-cluster negatives — never any existing label, parent name, or ordered cluster id.
- Aggregate the 5 naming agents by embedding their candidate names, taking the medoid as the label and the mean pairwise cosine as a naming-confidence score, and vary model family / temperature / member sample / instruction paraphrase across agents (not just seed) so the ensemble captures the real variance components.
- Execute P8 merges the Dial-In way — on label embeddings rather than member embeddings: L2-normalize label vectors, build an affinity graph with geodesic arccos < 0.8, gate edges with a von Mises-Fisher same-intent probability at tau=0.7, merge connected components, and emit TopicGPT-style absorbed_ids rows straight into your lookup-table remap.
- Add a computable MECE gate using GoalEx's assignment matrix: run the 'Predicate: <description>. Text: <query>. Is the Predicate true? Yes or No. When uncertain, output No.' probe over a sample against every L1 description, then report P(m=0) as the coverage gap and P(m>=2) as the overlap rate, with the top offending label pairs named.
- Determine K in P5 by triangulating three cheap oracles rather than one: ClusterLLM's pairwise hierarchical sampling (only 198-594 LLM calls between k_min=2 and k_max=200, scored by F-beta consistency), Dial-In's good/bad ratio maximized per iteration, and your existing stability peak — and expect the LLM oracles to disagree with BIC/elbow by design.
- Adopt ClusterLLM's entropy-based triplet sampling as the acquisition function for human/LLM annotation budget: soft-assign with Student-t, keep only the max(2%*K, 2) nearest clusters, rank by renormalized entropy, slice [gamma_high*N : gamma_low*N], and cap at a fixed Q (1,024 sufficed for both 3k and 50k-item corpora).
- Restrict ClusterLLM-style triplet fine-tuning to L2 sub-intents only — it demonstrably fails on coarse 10-18 cluster domain tasks and was ineffective at 1,500+ clusters — and use TnT-LLM-style prompt induction for L1.
- Use k-LLMmeans summary-as-centroid updates (LLM call count = k per summarization step, independent of N; ~$1 and ~1 minute for a benchmark, $2.5 for a 50k corpus) to give every cluster an auditable textual prototype you can diff across quarterly P12 reruns for drift detection.
- For P2 sample sizing, follow TopicGPT's empirical stopping rule rather than a fixed number: keep generating until no new topic survives refinement for ~200 consecutive documents (their refined-topic count plateaued at ~600 docs from 14k-32k corpora), and use ~10k summaries for a taxonomy of up to 100 labels per TnT-LLM.
- Batch labeling at 10-25 items with a shared system prompt, a strict XML envelope, echoed per-item ids, and an output correction layer (PromptPack reports 89% cost reduction and 2.5x throughput at batch 20 with AUC preserved) — but validate accuracy at your chosen batch size against batch size 1 on your own gold.
- Use 3 paraphrased instruction variants x 1 sample and majority-vote, rather than 1 prompt x 3 samples, because inter-prompt variance dominates on interpretative tasks; report the vote fraction as your routing confidence and report Inter-Prompt Reliability (pairwise agreement rate) alongside kappa.
- Never route P10 margin decisions on verbalized confidence: it takes only a handful of distinct values (the score granularity gap), tracks commitment rather than correctness, and flips relative to token likelihood depending on scoring protocol — use the distilled classifier's calibrated probability or the committee vote fraction instead.
- Defend every pairwise or single-choice LLM judgment against position and selection bias by randomizing option order, averaging over multiple runs, requiring evidence before the rating (Multiple Evidence Calibration), and optionally estimating the option-ID prior on a small permuted subset (PriDe) — and remember that high test-retest reliability does not imply low position bias.
- Validate the taxonomy by its downstream utility, not just its intrinsic quality: train a logistic regression on frozen embeddings over LLM pseudo-labels and require it to match the LLM classifier by paired t-test, then report the accuracy gap per segment (language, device, frequency decile), where TnT-LLM saw +2.3%/-2.7% for ada2 but +8.4%/-9.9% for Instructor-XL.
- Ban 'Other', 'General', 'Unclear', 'Miscellaneous' and 'Undefined' from the taxonomy itself as a hard guardrail failure, while keeping an 'Other' bucket in the assignment prompt purely as the coverage instrument (TnT-LLM achieved >99.5% coverage this way).
- Require every label assignment to return a verbatim supporting span from the query plus a one-sentence reason (TopicGPT's contract), so hallucinated assignments are detectable by substring check and P11's deterministic sample display has real evidence to show.
- For active learning, seed the labeled pool with diversity sampling over frozen high-quality embeddings and use BADGE as the default acquisition strategy with margin sampling as the cheap fallback, since the 2025 benchmark shows strategy effectiveness is highly sensitive to embedding quality and BADGE is the most robust across tasks.
- Use TextClusterLab's synthetic generator (controllable class imbalance, intra-cluster compactness, inter-cluster diversity) to build deterministic unit tests for the P4 algorithm battery and P5 K-selection, so the pipeline is testable without depending on a real labeled corpus.

## Unverified or version-dependent

- TnT-LLM's five prompt templates are rendered as figure IMAGES in both the arXiv PDF and the HTML, so their exact text is not machine-extractable from the paper. The verbatim prompts I supply come from the LangChain/LangGraph reference implementation on LangSmith Hub (wfh/tnt-llm-*), which reproduces the paper's design faithfully (same XML contract, same quality clauses, same rating/explanation/suggestions/updated_table output structure) but may differ in wording from the Microsoft originals. Compare against the paper's Figures 8-10 visually before treating any clause as authoritative.
- The LangSmith Hub prompts are community-published and mutable; I fetched them on 2026-08-17 and recorded commit hashes in the fetch, but the hub owner could push new commits. Pin by commit hash.
- Dial-In LLM (2412.09049) is at v3/v4 on arXiv and, per the paper, the fine-tuned Chinese coherence/naming models and the full 1,507-cluster dataset were to be released 'upon acceptance' — I could not verify that these artifacts are actually downloadable today. Its numbers are also on Chinese customer-service dialogue, not search queries; the action-object convention should transfer but the theta=0.8 and tau=0.7 merge thresholds are dataset- and embedder-specific (BGE-large-zh) and must be re-tuned.
- I did not read the full text of several 2026 papers (2606.19544, 2604.23178, 2606.22179, 2605.27752, 2606.29490, 2604.16413, 2604.15866, 2607.20528, 2511.04108, 2606.28328, 2603.23518) — those findings come from verified arXiv abstracts only. Effect sizes and protocol details should be checked in the full texts before you hard-code any threshold from them.
- k-LLMmeans (2502.09667) is labeled 'Published as a conference paper at ICLR 2026' in the v5 PDF; I verified the arXiv abstract and Discussion/cost section but not the venue listing independently.
- ClusterLLM's cost figure (~$0.60/dataset) is from gpt-3.5-turbo pricing in 2023. Absolute costs today will differ substantially; the structural claim that matters (query count is independent of dataset size: 1,024 triplets + 198-594 pairs) is what should carry over.
- The kappa standard-error table I computed uses the simplified Fleiss (1969) large-sample approximation with an assumed p_e; the true SE depends on the full confusion matrix, so treat those n-requirements as planning figures and use a nonparametric bootstrap on real data.
- The Wilson-interval sample sizes assume simple random sampling. Query logs are heavily Zipfian; if you sample by volume-weighted or stratified schemes (which you should), the effective sample size is smaller and you need design-effect-corrected intervals.
- TnT-LLM's agreement numbers come from three of the paper's own authors as raters (Phase 1) and four authors (Phase 2), not independent crowdworkers, on English-only subsets. Independent-annotator kappa on a comparable task could plausibly be lower, which strengthens rather than weakens my argument against a 0.9 gate.
- Whether TnT-LLM's minibatch=200 and ~10k-sample guidance transfers to short search queries is untested — queries are far shorter and far more numerous than chat conversations, so you can likely afford larger minibatches (or should batch by unique-query rather than by impression) and the summarization stage is probably skippable. Run a sensitivity sweep.
- I found no published work specifically on blind/anti-anchoring protocols for LLM cluster naming; Section 10 is my synthesis from LLM bias literature plus TnT-LLM's blinding practice, not a citable protocol.
- Reported claims that LLM judges show 33-41pp kappa deflation and that style bias dominates position bias are from two different 2026 studies with different benchmarks and rubrics; the two papers disagree on the relative magnitude of position bias (>0.10 vs <=0.04), which is likely a rubric/protocol artifact. Measure your own.

## Sources

- https://arxiv.org/abs/2403.12173
- https://arxiv.org/html/2403.12173v1
- https://dl.acm.org/doi/abs/10.1145/3637528.3671647
- https://api.smith.langchain.com/commits/wfh/tnt-llm-taxonomy-generation/latest
- https://api.smith.langchain.com/commits/wfh/tnt-llm-taxonomy-update/latest
- https://api.smith.langchain.com/commits/wfh/tnt-llm-taxonomy-review/latest
- https://api.smith.langchain.com/commits/wfh/tnt-llm-summary-generation/latest
- https://api.smith.langchain.com/commits/wfh/tnt-llm-classify/latest
- https://github.langchain.ac.cn/langgraph/tutorials/tnt-llm/tnt-llm/
- https://arxiv.org/abs/2305.14871
- https://arxiv.org/html/2305.14871v2
- https://arxiv.org/abs/2305.19783
- https://ar5iv.labs.arxiv.org/html/2305.19783
- https://aclanthology.org/2023.nlp4convai-1.7/
- https://arxiv.org/abs/2412.09049
- https://arxiv.org/html/2412.09049v3
- https://arxiv.org/abs/2502.09667
- https://arxiv.org/html/2502.09667v3
- https://arxiv.org/abs/2311.01449
- https://arxiv.org/html/2311.01449v2
- https://raw.githubusercontent.com/chtmp223/topicGPT/main/prompt/generation_1.txt
- https://raw.githubusercontent.com/chtmp223/topicGPT/main/prompt/refinement.txt
- https://raw.githubusercontent.com/chtmp223/topicGPT/main/prompt/assignment.txt
- https://raw.githubusercontent.com/chtmp223/topicGPT/main/prompt/correction.txt
- https://arxiv.org/abs/2305.13749
- https://ar5iv.labs.arxiv.org/html/2305.13749
- https://aclanthology.org/2023.emnlp-main.657/
- https://arxiv.org/abs/2305.12152
- https://arxiv.org/abs/2402.07386
- https://arxiv.org/abs/2511.05913
- https://arxiv.org/abs/2510.14640
- https://arxiv.org/abs/2503.15351
- https://arxiv.org/abs/2510.06747
- https://arxiv.org/abs/2510.15125
- https://arxiv.org/abs/2606.28328
- https://arxiv.org/abs/2603.23518
- https://aclanthology.org/2024.emnlp-main.1025/
- https://arxiv.org/abs/2606.19544
- https://arxiv.org/abs/2604.23178
- https://arxiv.org/abs/2606.22179
- https://arxiv.org/abs/2605.27752
- https://arxiv.org/abs/2606.29490
- https://arxiv.org/abs/2604.16413
- https://arxiv.org/abs/2604.15866
- https://arxiv.org/abs/2602.11962
- https://arxiv.org/abs/2506.01992
- https://arxiv.org/abs/2502.16892
- https://arxiv.org/abs/2406.12114
- https://arxiv.org/abs/2305.17926
- https://arxiv.org/abs/2310.01432
- https://arxiv.org/abs/2306.05685
- https://arxiv.org/abs/2309.03882
- https://arxiv.org/abs/2305.14975
- https://arxiv.org/abs/2203.11171
- https://arxiv.org/abs/2301.08721
- https://arxiv.org/abs/2607.20528
- https://arxiv.org/abs/2511.04108
- https://maartengr.github.io/BERTopic/getting_started/representation/llm.html
- https://raw.githubusercontent.com/MaartenGr/BERTopic/master/bertopic/representation/_openai.py
- https://pypi.org/pypi/bertopic/json
- https://pypi.org/pypi/langgraph/json
- https://pypi.org/pypi/langchain/json
- https://pypi.org/pypi/krippendorff/json
- https://pypi.org/pypi/scikit-learn/json
- https://pypi.org/pypi/statsmodels/json
- https://bmcmedresmethodol.biomedcentral.com/articles/10.1186/1471-2288-9-5
- https://mbrenndoerfer.com/writing/inter-annotator-agreement-kappa-alpha-reliability