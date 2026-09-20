# -*- coding: utf-8 -*-
"""影视 §4 candidate matched pairs: (1) entity string matches search top10k ↔ assistant; (2) near-paraphrase NN pairs (0.70≤sim<0.999)
with intent change (neighbour u_ds known when it is in search top1000). Adult-like rows masked."""
import sys, re, pandas as pd
SP=__import__("os").environ.get("SVA_WORKDIR", __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "work"))
sys.path.insert(0,SP); from sva_common import *; import sva_dom_film_rx as R
a,s=load(); c=cells(a,s,"影视"); T=c["搜索top10k"].reset_index(drop=True); T["query"]=T["query"].astype(str); T["rank"]=range(1,len(T)+1)
F=pd.read_parquet(f"{SP}/sva_final_rows.parquet"); U=F[(F.domain=="影视")&(F.tier=="user")].copy(); U["query"]=U["query"].astype(str)
lab1=U[U.surface=="search_top1000"].set_index("query").u_ds.to_dict(); own10=T.set_index("query").td_l1_name.to_dict()
ok=lambda q,u=None,o=None: not R.adult_like(q,u,o)
ENT=["藏锋","问心","人间中毒","鬼灭之刃","火影","海绵宝宝","甄嬛传","我的前半生","心动的信号","奔跑吧","动画片","韩国电影","票房","演员表","结局","叫什么","推荐"]
rows=[]
for e in ENT:
    sx=T[T["query"].str.contains(e,regex=False)]; sx=sx.loc[pd.Series([ok(q,None,o) for q,o in zip(sx["query"],sx.td_l1_name)],index=sx.index,dtype=bool)]
    print(f"\n### {e}: search10k {len(sx)} rows; 头: "+" | ".join(f"{q}(#{r},{lab1.get(q,'-')})" for q,r in zip(sx["query"].head(6),sx["rank"].head(6))))
    for sf in ("assistant_top1k","assistant_random1k"):
        ax=U[(U.surface==sf)&U["query"].str.contains(e,regex=False)]; ax=ax.loc[pd.Series([ok(q,u,o) for q,u,o in zip(ax["query"],ax.u_ds,ax.own_intent)],index=ax.index,dtype=bool)]
        if len(ax):
            print(f"   {sf} {len(ax)}: "+" | ".join(f"{q[:40]}({u},PV{int(p)})" for q,u,p in zip(ax.sort_values('pv',ascending=False)["query"].head(10),ax.sort_values('pv',ascending=False).u_ds.head(10),ax.sort_values('pv',ascending=False).pv.head(10))))
            for q,u,p in zip(ax["query"],ax.u_ds,ax.pv): rows.append(dict(entity=e,surface=sf,query=q,u_ds=u,pv=p,search_example=(sx["query"].iloc[0] if len(sx) else None),search_rank=(int(sx["rank"].iloc[0]) if len(sx) else None)))
NN=pd.read_parquet(f"{SP}/sva_dom_film_nn.parquet"); NN["query"]=NN["query"].astype(str)
NN["u_nn"]=NN.nn_query.map(lab1); NN["own_nn"]=NN.nn_query.map(own10)
for sf in ("assistant_top1k","assistant_random1k"):
    x=NN[(NN.surface==sf)&(NN.sim>=0.70)&(NN.sim<0.999)]; x=x.loc[pd.Series([ok(q,u,o) and ok(n_) for q,u,o,n_ in zip(x["query"],x.u_ds,x.own_intent,x.nn_query)],index=x.index,dtype=bool)]
    x=x.sort_values("sim",ascending=False)
    print(f"\n### NN {sf} 0.70≤sim<0.999 n_safe={len(x)}; neighbour in top1000 (u_ds known) {x.u_nn.notna().sum()}")
    ch=x[x.u_nn.notna()&(x.u_nn!=x.u_ds)]
    print("  intent-changed (neighbour labelled): "+str(len(ch))); [print(f"   {r.nn_query}(#{r.nn_rank},{r.u_nn}) → {r.query[:44]}({r.u_ds},sim{r.sim:.2f})") for r in ch.itertuples()]
    print("  all (first 70 by sim, neighbour own_intent shown):"); [print(f"   {r.nn_query}(#{r.nn_rank},{r.own_nn}) → {r.query[:44]}({r.u_ds},{r.sim:.2f})") for r in x.head(70).itertuples()]
    for r in x.itertuples(): rows.append(dict(entity="NN",surface=sf,query=r.query,u_ds=r.u_ds,pv=r.pv,search_example=r.nn_query,search_rank=r.nn_rank,sim=r.sim,u_nn=r.u_nn))
pd.DataFrame(rows).to_csv(f"{SP}/sva_dom_film_pair_candidates.csv",index=False,encoding="utf-8-sig")
