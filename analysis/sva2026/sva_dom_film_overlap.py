# -*- coding: utf-8 -*-
"""影视: exact overlap (assistant user rows vs search top10k/top1000 user rows), reverse overlap, wrapping (a search
top1000 query embedded in a longer assistant query) with u_ds transitions; intent examples for the main shifts."""
import sys, re, math, pandas as pd
SP=__import__("os").environ.get("SVA_WORKDIR", __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "work"))
sys.path.insert(0,SP); from sva_common import *
fm=frame(); NAME={k:v[0] for k,v in fm.FRAME.items()}
SUGG=re.compile(r"(性感|妩媚|泳装|泳衣|比基尼|内衣|丝袜|脱|裤子|舌吻|胸|扭动|摇摆|撩|诱惑|湿身|裸|色情|黄片|成人)")
def safe(q,u,o): return not ((u=="U12") or ("成人" in str(o)) or ("情色" in str(o)) or SUGG.search(q))
def wilson(k,n,z=1.96):
    p=k/n; den=1+z*z/n; c=(p+z*z/(2*n))/den; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den; return p,max(0,c-h),min(1,c+h)
a,s=load(); c=cells(a,s,"影视"); T10=c["搜索top10k"]["query"].astype(str).tolist(); rank={q:i+1 for i,q in enumerate(T10)}
F=pd.read_parquet(f"{SP}/sva_final_rows.parquet"); U=F[(F.domain=="影视")&(F.tier=="user")].copy(); U["query"]=U["query"].astype(str)
S1=U[U.surface=="search_top1000"]; S1set=set(S1["query"]); lab1=S1.set_index("query").u_ds.to_dict()
for sf in ("assistant_top1k","assistant_random1k"):
    x=U[U.surface==sf]; m=x["query"].isin(set(T10)); m1=x["query"].isin(S1set); k=int(m.sum()); p,lo,hi=wilson(k,len(x))
    print(f"\n## EXACT {sf}: vs search top10k {k}/{len(x)} = {p*100:.1f}% [{lo*100:.1f}-{hi*100:.1f}], PV {x.pv[m].sum()/x.pv.sum()*100:.1f}% ; vs top1000 {int(m1.sum())} ({m1.mean()*100:.1f}%)")
    if k:
        y=x[m]; print("   u_ds of overlapping rows: "+" ; ".join(f"{cc}{NAME[cc]} {v}" for cc,v in y.u_ds.value_counts().items()))
        y=y[[safe(q,u,o) for q,u,o in zip(y["query"],y.u_ds,y.own_intent)]].sort_values("pv",ascending=False)
        print("   例: "+" | ".join(f"{q}(助手PV{int(pv)},搜索#{rank[q]})" for q,pv in zip(y["query"].head(20),y.pv.head(20))))
aset=set(U[U.surface.str.startswith("assistant")]["query"]); rev=[q for q in S1["query"] if q in aset]
print(f"\n## REVERSE search top1000 in assistant (head∪tail): {len(rev)}/1000; e.g. "+" | ".join(rev[:25]))
GEN=re.compile(r"^(电视剧|电影|动漫|动画片|动画|短剧|视频|综艺|韩剧|日剧|美剧|泰剧|国产剧|港剧|免费观看|在线观看|全集|完整版|高清|免费|在线播放|电视剧大全|电影大全|抖音|快手|在线|观看|直播|影视|大全|电视|免费观看全集|在线观看免费|免费在线观看|高清完整版|电视剧免费观看|电影免费观看|全集免费观看|免费观看完整版|完整版免费观看|动漫免费观看|在线观看电视剧|韩剧在线观看|日剧在线观看)+$")
cores=sorted([q for q in S1["query"].unique() if len(q)>=3 and not q.isdigit()],key=len,reverse=True)
pairs=[]
for sf in ("assistant_top1k","assistant_random1k"):
    for q,u,o,pv in U[U.surface==sf][["query","u_ds","own_intent","pv"]].itertuples(index=False):
        for core in cores:
            if core in q and len(q)>=len(core)+2:
                pairs.append(dict(surface=sf,core=core,query=q,u_core=lab1[core],u_wrap=u,own=o,pv=pv,generic=bool(GEN.match(core)))); break
P=pd.DataFrame(pairs); P.to_csv(f"{SP}/sva_dom_film_wrap_pairs.csv",index=False,encoding="utf-8-sig")
for sf in ("assistant_top1k","assistant_random1k"):
    n=int((U.surface==sf).sum()); x=P[P.surface==sf]; xg=x[~x.generic]
    p,lo,hi=wilson(len(x),n); p2,lo2,hi2=wilson(len(xg),n)
    print(f"\n## WRAP {sf}: all cores {len(x)}/{n} = {p*100:.1f}% [{lo*100:.1f}-{hi*100:.1f}] ; non-generic cores {len(xg)}/{n} = {p2*100:.1f}% [{lo2*100:.1f}-{hi2*100:.1f}]")
    print("   most frequent cores: "+" | ".join(f"{k}({v})" for k,v in x.core.value_counts().head(15).items()))
    for nm,z in (("all",x),("non-generic",xg)):
        if not len(z): continue
        print(f"   [{nm}] intent unchanged {(z.u_core==z.u_wrap).mean()*100:.1f}% (n={len(z)})")
        for (c1,c2),v in z[z.u_core!=z.u_wrap].groupby(["u_core","u_wrap"]).size().sort_values(ascending=False).head(8).items():
            e=z[(z.u_core==c1)&(z.u_wrap==c2)]; e=e[[safe(q,u,o) for q,u,o in zip(e["query"],e.u_wrap,e.own)]]
            print(f"     {c1}{NAME[c1]}→{c2}{NAME[c2]} {v} ({v/len(z)*100:.1f}%) 例: "+" | ".join(f"[{a_}]→{b_[:34]}" for a_,b_ in zip(e.core.head(4),e["query"].head(4))))
print("\n## INTENT EXAMPLES (adult-masked): top-5 by PV + 5 random")
for code in ["U01","U02","U03","U04","U05","U07","U08","U09","U10","U11","U13"]:
    for sf in ("search_top1000","assistant_top1k","assistant_random1k"):
        x=U[(U.surface==sf)&(U.u_ds==code)]; x=x[[safe(q,u,o) for q,u,o in zip(x["query"],x.u_ds,x.own_intent)]]
        if len(x): print(f"  {code}{NAME[code]} [{sf}] n_safe={len(x)}: 头 "+" | ".join(x.sort_values('pv',ascending=False)["query"].head(5).str[:30])+"  ‖ 随机 "+" | ".join(x["query"].sample(min(5,len(x)),random_state=5).str[:30]))
