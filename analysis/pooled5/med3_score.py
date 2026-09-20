#!/usr/bin/env python
"""用 health-pool3 自己的产物给新快照打标签，**不重新推导任何体系**。

要求是「标签/家族/叶与 health-pool3 保持一致」，所以这里一个模型都不重训、一个聚类都不重跑：

- **稠密**：`shibing624/text2vec-base-chinese`（该运行 P3a 挑中的），无 instruction 前缀。
- **稀疏**：char(1,3) TF-IDF → 256d SVD。**拟合对象是原语料的 20,316 条文本**，不是新数据——
  只有这样新串才落进同一个空间。实测：在原语料上重拟合复现 `emb_svd_char.npy` **逐位相同**
  （max abs diff 0.000e+00，vocab 50,732 与 evr 0.218149 都对得上）。
- **混合**：`hybrid(dense, svd, alpha=0.85)`，alpha 取自该运行的 `representation.json`。
- **叶与家族**：`centroid_classifier.joblib`（37 个叶心 × 1024 维，自带 margin 阈值与 leaf→family）。
- **L1 意图**：`topdown_model.joblib` 里的 RuleEngine + StandardScaler + LogisticRegression，特征按
  `build_features` 原样拼（稠密 ⊕ 归一化表层统计 ⊕ 规则位）。
- **L2 子意图**：该运行没有存子簇中心，但存了每一行的 `td_l2`。所以从原语料的
  (`emb_base`, `td_l1`, `td_l2`) 现算出每个 L1 内部的子中心，再把新串按 L1 内最近子中心归类——
  与叶用最近中心是同一种做法。

**两个快照的老标签一律不重打。** 医疗随机 的两个快照直接用运行交付的 `labels_full.csv`；
只有新快照走上面这条打分路径。重打老行只会引入与交付不一致的风险，而且没有任何好处。

打分器本身的准确性是**测出来的**：脚本先用同一条路径给原语料的 20,316 行打一遍，与交付标签逐行比，
一致率写进 `work/医疗3/score_validation.json`。报告引用的是那个数，不是"应该一样"。

    HF_HOME=$(pwd)/.hf .venv/bin/python analysis/pooled5/med3_score.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

QM = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(QM / "src"))
sys.path.insert(0, str(QM / "analysis/pooled5"))
from qmine.ops.represent import build_sparse, encode_corpus, hybrid, load_encoder  # noqa: E402
from qmine.ops.classify import build_features  # noqa: E402
from qmine.ops.audit import char_profile  # noqa: E402

GEN = QM / "runs/health-pool3/gen01"
BASE = QM / "analysis/pooled5/work/医疗3"
ENCODER = "shibing624/text2vec-base-chinese"
ALPHA = 0.85
NGRAM, MIN_DF, SVD_DIMS, SEED = (1, 3), 2, 256, 0


class Space:
    """health-pool3 的表示空间，按它自己的产物重建，并在重建时自证。"""

    def __init__(self) -> None:
        self.texts = pd.read_parquet(GEN / "corpus.parquet")["query"].astype(str).tolist()
        sp = build_sparse(self.texts, analyzer="char", ngram_range=NGRAM, min_df=MIN_DF,
                          svd_dims=SVD_DIMS, seed=SEED, tokenizer="jieba")
        saved = np.load(GEN / "emb_svd_char.npy")
        d = float(np.abs(sp["svd_block"] - saved).max())
        assert d < 1e-5, (f"the refitted sparse block does not reproduce the run's own "
                          f"emb_svd_char.npy (max abs diff {d:.3e}) — the space is NOT the same, stop")
        self.vec, self.svd, self.repro_diff = sp["vectorizer"], sp["svd"], d
        self.enc = load_encoder(ENCODER, offline=False, cache_folder=str(QM / ".hf"))

    def dense(self, texts: list[str]) -> np.ndarray:
        return encode_corpus(self.enc, texts, instruction=None)

    def sparse(self, texts: list[str]) -> np.ndarray:
        from sklearn.preprocessing import normalize
        return normalize(self.svd.transform(self.vec.transform(texts)).astype(np.float32))

    def embed(self, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
        dn = self.dense(texts)
        return dn, hybrid(dn, self.sparse(texts), ALPHA)


def sub_centroids() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """每个 L1 内部的子中心，从原语料的 (emb_base, td_l1, td_l2) 现算。"""
    X = np.load(GEN / "emb_base.npy")
    lab = pd.read_csv(GEN / "labels_full.csv", encoding="utf-8-sig")
    assert len(lab) == len(X), f"labels_full {len(lab)} vs emb_base {len(X)}"
    out = {}
    for l1, g in lab.groupby("td_l1"):
        subs = sorted(g["td_l2"].astype(str).unique())
        C = np.stack([X[g.index[g["td_l2"].astype(str) == s]].mean(0) for s in subs])
        C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-12
        out[str(l1)] = (C, np.array(subs))
    return out


def score(texts: list[str], space: Space, subc: dict) -> pd.DataFrame:
    df = pd.DataFrame([char_profile(t) for t in texts])
    dn, H = space.embed(texts)

    cc = joblib.load(GEN / "centroid_classifier.joblib")
    sims = H @ cc.centroids.T
    leaf = sims.argmax(1)
    part = np.partition(sims, -2, axis=1)
    margin = part[:, -1] - part[:, -2]
    family = np.asarray(cc.leaf_family)[leaf]

    td = joblib.load(GEN / "topdown_model.joblib")
    rule_feats = td["engine"].feature_matrix(texts)
    X, _ = build_features(df, dn, rule_features=rule_feats, scaler=td["scaler"])
    proba = td["model"].predict_proba(X)
    cls = td["model"].classes_
    l1 = cls[proba.argmax(1)]
    p = np.sort(proba, axis=1)
    conf, tmar = p[:, -1], p[:, -1] - p[:, -2]

    l2 = []
    for i, c in enumerate(l1):
        C, names = subc.get(str(c), (None, None))
        l2.append(str(names[(C @ dn[i]).argmax()]) if C is not None else f"{c}__0")

    names = cc.names if isinstance(cc.names, dict) else {}
    return pd.DataFrame({
        "query": texts, "bu_leaf": leaf, "bu_leaf_name": [names.get(int(x), names.get(str(x), "")) for x in leaf],
        "bu_family_final": family, "bu_margin": margin.round(4),
        "bu_ambiguous": margin < float(cc.margin_threshold),
        "td_l1": l1, "td_confidence": conf.round(4), "td_margin": tmar.round(4),
        "td_ambiguous": tmar < 0.02, "td_l2": l2,
    })


def validate(space: Space, subc: dict) -> dict:
    """同一条路径给原语料打一遍，与交付标签逐行比。报告引用的一致率就是这里的数。"""
    lab = pd.read_csv(GEN / "labels_full.csv", encoding="utf-8-sig")
    got = score(lab["query"].astype(str).tolist(), space, subc)
    out = {"n": int(len(lab)), "sparse_reproduction_max_abs_diff": space.repro_diff}
    for col in ("bu_leaf", "bu_family_final", "td_l1", "td_l2"):
        a, b = lab[col].astype(str).to_numpy(), got[col].astype(str).to_numpy()
        out[f"{col}_一致率%"] = round(100 * float((a == b).mean()), 2)
    return out


def main() -> int:
    BASE.mkdir(parents=True, exist_ok=True)
    space = Space()
    print(f"space rebuilt; sparse reproduces the run's matrix to {space.repro_diff:.3e}")
    subc = sub_centroids()
    print(f"sub-centroids for {len(subc)} L1 classes")

    v = validate(space, subc)
    (BASE / "score_validation.json").write_text(json.dumps(v, ensure_ascii=False, indent=1), encoding="utf-8")
    print("validation on the ORIGINAL 20,316 rows vs the delivered labels:")
    for k, x in v.items():
        if k.endswith("一致率%"):
            print(f"   {k}: {x}%")

    src = BASE / "new_rows.csv"
    if not src.exists():
        print(f"\n{src} not built yet — run the corpus builder first; validation above still stands")
        return 0
    new = pd.read_csv(src, encoding="utf-8-sig", keep_default_na=False)
    got = score(new["query"].astype(str).tolist(), space, subc)
    out = pd.concat([new.drop(columns=[c for c in got.columns if c in new.columns and c != "query"]),
                     got.drop(columns=["query"])], axis=1)
    out.to_csv(BASE / "new_labels.csv", index=False, encoding="utf-8-sig")
    print(f"\n→ {BASE / 'new_labels.csv'}  ({len(out):,} rows)")
    print(out["td_l1"].value_counts().head(8).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
