# -*- coding: utf-8 -*-
"""Derived numbers quoted in sva_dom_film.md: control-band CIs, assistant-only theme aggregate, head PV ratios."""
import math, pandas as pd
SP=__import__("os").environ.get("SVA_WORKDIR", __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "work"))
def wilson(k,n,z=1.96):
    p=k/n; den=1+z*z/n; c=(p+z*z/(2*n))/den; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den; return p,max(0,c-h),min(1,c+h)
W=lambda k,n: "%.1f[%.1f–%.1f] (k=%d,n=%d)"%(wilson(k,n)[0]*100,wilson(k,n)[1]*100,wilson(k,n)[2]*100,k,n)
# controls: counts from sva_dom_film_nn.py output (n=1000 each): A 同57 近802 中96 远45 ; B 同15 近527 中179 远279
for nm,ks in (("对照A",(57,802,96,45)),("对照B",(15,527,179,279))):
    assert sum(ks)==1000; print(nm," | ".join(W(k,1000) for k in ks))
N=pd.read_parquet(f"{SP}/sva_dom_film_nn_themes.parquet")
NEW=["T4 剧情·人物·因果讨论与评判","T2 假设·对战·同人设定·续写","T1 生成/编辑视频图像","T3 识别作品/画面/人物","T6 求'视频'形态的非片源内容(教程/片段/台词/歌)","T5 平台账号与创作者操作"]
assert set(NEW)<=set(N.theme.unique())
for sf in ("assistant_top1k","assistant_random1k"):
    x=N[N.surface==sf]; low=x[x.sim<0.7]; hi=x[x.sim>=0.8]
    k=int(low.theme.isin(NEW).sum()); kh=int(hi.theme.isin(NEW).sum())
    print(f"{sf}: new-need themes among low-sim rows = share of all rows {W(k,len(x))} ; share of low-sim rows {W(k,len(low))} ; same themes among sim≥0.80 rows {W(kh,len(hi))}")
    if sf=="assistant_top1k": print(f"   PV share of head user PV: {low.pv[low.theme.isin(NEW)].sum()/x.pv.sum()*100:.1f}%")
t=pd.read_csv(f"{SP}/sva_dom_film_tiers.csv"); h=t[(t.ver=="v3")&(t.surface=="assistant_top1k")]
user_pv=int(h[h.tier=="user"].pv.iloc[0]); tot=int(h.pv.sum()); s1k=int(t[(t.ver=="v3")&(t.surface=="search_top1000_span")].pv.sum())
print(f"head PV total {tot}; user {user_pv} ({user_pv/tot*100:.1f}%); removed {(tot-user_pv)/tot*100:.1f}% ; head user PV / search top1000 PV = {user_pv/s1k*100:.1f}%")
