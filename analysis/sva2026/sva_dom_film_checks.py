# -*- coding: utf-8 -*-
"""影视 checks: World Cup / date evidence; U01 decomposition (live TV vs other) and U01-ex-live difference; pirate-site names in U09;
「AI视频」 and '比例为' template rows; head concentration."""
import sys, re, math, pandas as pd
SP=__import__("os").environ.get("SVA_WORKDIR", __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "work"))
sys.path.insert(0,SP); from sva_common import *; import sva_dom_film_rx as R
def wilson(k,n,z=1.96):
    p=k/n; den=1+z*z/n; c=(p+z*z/(2*n))/den; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den; return p,max(0,c-h),min(1,c+h)
def newcombe(k1,n1,k2,n2):
    p1,l1,u1=wilson(k1,n1); p2,l2,u2=wilson(k2,n2); d=p2-p1
    return d, d-math.sqrt((p2-l2)**2+(u1-p1)**2), d+math.sqrt((u2-p2)**2+(p1-l1)**2)
W=lambda k,n: "%.1f%%[%.1f–%.1f](k=%d,n=%d)"%(wilson(k,n)[0]*100,wilson(k,n)[1]*100,wilson(k,n)[2]*100,k,n)
a,s=load(); c=cells(a,s,"影视")
F=pd.read_parquet(f"{SP}/sva_final_rows.parquet"); U=F[(F.domain=="影视")&(F.tier=="user")].copy(); U["query"]=U["query"].astype(str)
S1=U[U.surface=="search_top1000"]; AH=U[U.surface=="assistant_top1k"]; AT=U[U.surface=="assistant_random1k"]
print("## World Cup / sports mentions")
for pat in ["世界杯","欧洲杯","奥运","女排","NBA","中超","斯诺克","赛程","足球","体育"]:
    sa=s["query"].astype(str); aa=a["query"].astype(str)
    print(f"  {pat}: search 影视 top10k {int(c['搜索top10k']['query'].astype(str).str.contains(pat,regex=False).sum())} (top1000 {int(S1['query'].str.contains(pat,regex=False).sum())}); search all verticals {int(sa.str.contains(pat,regex=False).sum())}; assistant 影视 user head {int(AH['query'].str.contains(pat,regex=False).sum())} tail {int(AT['query'].str.contains(pat,regex=False).sum())}; assistant all cats all tiers {int(aa.str.contains(pat,regex=False).sum())}")
wc=c['搜索top10k'][c['搜索top10k']['query'].astype(str).str.contains("世界杯")]; print("  search 世界杯 rows: "+" | ".join(f"{q}(#{i+1})" for q,i in zip(wc['query'].head(10),wc.index[:10])))
print("\n## date evidence: assistant rows mentioning 2026年8月2x日 / 9月 (all cats, all tiers) by tier")
aa=a.assign(q=a["query"].astype(str))
m=aa.q.str.contains(r"2026年8月(2\d|3[01])日",regex=True); print("  2026年8月21–31日:",int(m.sum()),aa[m].tier.value_counts().to_dict(),"例:"," | ".join(aa[m].q.head(6).str[:40]))
m=aa.q.str.contains(r"2026年9月\d{1,2}日",regex=True); print("  2026年9月x日:",int(m.sum())," | ".join(aa[m].q.head(6).str[:40]))
print("\n## U01 decomposition")
for nm,x in (("search_top1000",S1),("assistant_top1k",AH),("assistant_random1k",AT)):
    u=x[x.u_ds=="U01"]; live=u["query"].map(lambda q: R.hit("电视频道/直播",q)); plat=u["query"].map(lambda q: R.hit("平台/站点名",q))
    print(f"  {nm}: U01 n={len(u)}; live {int(live.sum())} (PV {u.pv[live].sum()/x.pv.sum()*100:.1f}% of surface); platform/site {int((plat&~live).sum())}; other {int((~plat&~live).sum())}: "+" | ".join(u[~plat&~live]['query'].head(8)))
k1=int(((S1.u_ds=="U01")&~S1["query"].map(lambda q: R.hit("电视频道/直播",q))).sum()); k2=int(((AH.u_ds=="U01")&~AH["query"].map(lambda q: R.hit("电视频道/直播",q))).sum())
d,lo,hi=newcombe(k1,len(S1),k2,len(AH)); print(f"  U01 excluding live-TV rows: search {W(k1,len(S1))} → head {W(k2,len(AH))} diff {d*100:+.1f}[{lo*100:+.1f},{hi*100:+.1f}]")
kl1=int(S1["query"].map(lambda q: R.hit("电视频道/直播",q)).sum()); kl2=int(AH["query"].map(lambda q: R.hit("电视频道/直播",q)).sum()); d,lo,hi=newcombe(kl1,len(S1),kl2,len(AH)); print(f"  live-TV marker diff {d*100:+.1f}[{lo*100:+.1f},{hi*100:+.1f}]")
print("\n## Newcombe for key marker diffs search_top1000→assistant_top1k")
for mk in ["播放页/资源词","识别作品/人物","推荐/清单","演员/角色","剧情/结局","短剧","儿童向IP/动画片","视频/图像生成请求","疑问句","第一人称","平台/站点名","指定集数/季"]:
    k1=int(S1["query"].map(lambda q: R.hit(mk,q)).sum()); k2=int(AH["query"].map(lambda q: R.hit(mk,q)).sum()); d,lo,hi=newcombe(k1,len(S1),k2,len(AH))
    k3=int(AT["query"].map(lambda q: R.hit(mk,q)).sum()); d2,lo2,hi2=newcombe(k2,len(AH),k3,len(AT))
    print(f"  {mk}: {k1/len(S1)*100:.1f}→{k2/len(AH)*100:.1f} {d*100:+.1f}[{lo*100:+.1f},{hi*100:+.1f}] ; head→tail {d2*100:+.1f}[{lo2*100:+.1f},{hi2*100:+.1f}]")
print("\n## pirate/aggregator site names inside U09 rows")
SITE=re.compile(r"(影院|电影网|影视网|樱花动漫|神马|人人视频|天美|青柠|YY|农夫|老王|四库|8090|78动漫)")
for nm,x in (("search_top1000",S1),("assistant_top1k",AH),("assistant_random1k",AT)):
    u=x[x.u_ds=="U09"]; k=int(u["query"].map(lambda q: bool(SITE.search(q))).sum()); ka=int(x["query"].map(lambda q: bool(SITE.search(q))).sum())
    print(f"  {nm}: site names in U09 {W(k,len(u))}; in all rows {W(ka,len(x))}; 例: "+" | ".join(x[x['query'].map(lambda q: bool(SITE.search(q)))]['query'].head(8)))
print("\n## template / card rows kept as user")
for pat in ["「AI视频」","比例为","#"]:
    y=AH[AH["query"].str.contains(pat,regex=False)]; print(f"  head '{pat}': n={len(y)} PV {y.pv.sum()/AH.pv.sum()*100:.2f}% of head user PV; u_ds {y.u_ds.value_counts().to_dict()}")
    y=AT[AT["query"].str.contains(pat,regex=False)]; print(f"  tail '{pat}': n={len(y)}; u_ds {y.u_ds.value_counts().to_dict()}")
print("\n## concentration: top-10 user rows PV share")
for nm,x in (("search_top1000",S1),("assistant_top1k",AH)):
    print(f"  {nm}: top1 {x.pv.max()/x.pv.sum()*100:.1f}% ; top10 {x.pv.nlargest(10).sum()/x.pv.sum()*100:.1f}% ; total PV {int(x.pv.sum())}; PV floor {int(x.pv.min())}")
print("\n## personal flag examples (types only)")
y=AT[AT.p_ds==1]; print("  n=",len(y)); print("  "+" | ".join(y['query'].str[:30].head(34)))
