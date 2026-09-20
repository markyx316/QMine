# -*- coding: utf-8 -*-
"""影视 §2 query-form markers on 4 surfaces (user tier). search_top1000/assistant_* from sva_final_rows.parquet;
search_top10k from sva_common.cells(). Row share + Wilson CI; PV share for heads; adult-masked examples."""
import sys, re, math, random, numpy as np, pandas as pd
SP=__import__("os").environ.get("SVA_WORKDIR", __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "work"))
sys.path.insert(0,SP); from sva_common import *; import sva_dom_film_rx as R
def wilson(k,n,z=1.96):
    p=k/n; den=1+z*z/n; c=(p+z*z/(2*n))/den; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den; return p,max(0,c-h),min(1,c+h)
a,s=load(); c=cells(a,s,"影视")
F=pd.read_parquet(f"{SP}/sva_final_rows.parquet"); U=F[(F.domain=="影视")&(F.tier=="user")]
t10=c["搜索top10k"]; t10=pd.DataFrame({"query":t10["query"].astype(str),"pv":t10.wise_pv.astype(float),"u_ds":None,"own_intent":t10.td_l1_name})
S={"search_top1000":U[U.surface=="search_top1000"],"search_top10k":t10,"assistant_top1k":U[U.surface=="assistant_top1k"],"assistant_random1k":U[U.surface=="assistant_random1k"]}
S={k:v.assign(query=v["query"].astype(str)).reset_index(drop=True) for k,v in S.items()}
rows=[]
print("n:",{k:len(v) for k,v in S.items()})
print("\n## 长度: median / p90 / >30字% / >60字%")
for k,v in S.items():
    L=v["query"].str.len(); print(f"  {k}: {L.median():.0f} / {L.quantile(.9):.0f} / {(L>30).mean()*100:.1f}% / {(L>60).mean()*100:.2f}%  PV加权中位长度 "+(f"{np.average(L,weights=v.pv):.1f}(均值)" if 'random' not in k else "-"))
def bare(q):
    return len(q)<=6 and not any(R.hit(n,q) for n in ["播放页/资源词","电视频道/直播","平台/站点名","疑问句","委托/动作开头","指定集数/季","网盘/下载词"])
EXPL=re.compile(r"(性感|妩媚|泳装|泳衣|比基尼|内衣|丝袜|脱衣|脱掉|脱下|脱光|往下拉|舌吻|大胸|胸部|露胸|湿身|裸|色情|黄片|成人|诱惑|撩人)")
names=list(R.RX)+["机械裸短串(≤6字无动作/疑问/资源词)","露骨关键词(计数,不举例)"]
rng=random.Random(7)
for nm in names:
    line=[]; exl=[]
    for k,v in S.items():
        q=v["query"].tolist(); u=v["u_ds"].tolist(); o=v["own_intent"].tolist()
        if nm.startswith("机械裸"): m=[bare(x) for x in q]
        elif nm.startswith("露骨关键词"): m=[bool(EXPL.search(x)) for x in q]
        else: m=[R.hit(nm,x) for x in q]
        m=np.array(m); kk=int(m.sum()); n=len(m); p,lo,hi=wilson(kk,n)
        pvs=float((v.pv.values*m).sum()/v.pv.sum()) if k!="assistant_random1k" else float("nan")
        rows.append(dict(marker=nm,surface=k,n=n,k=kk,share=p,lo=lo,hi=hi,pv_share=pvs))
        line.append(f"{k[:9]}{k[-5:]} {p*100:5.1f}[{lo*100:4.1f}-{hi*100:4.1f}] k={kk:<4}"+(f" PV{pvs*100:5.1f}" if k!="assistant_random1k" else ""))
        if "计数" not in nm and kk:
            vv=v[m].copy(); vv=vv[[not R.adult_like(x,uu,oo) for x,uu,oo in zip(vv["query"],vv["u_ds"],vv["own_intent"])]]
            top=vv.sort_values("pv",ascending=False)["query"].head(3).str[:30].tolist()
            rnd=vv["query"].sample(min(3,len(vv)),random_state=3).str[:30].tolist() if len(vv) else []
            exl.append(f"     例[{k}] 头:{' | '.join(top)}  随机:{' | '.join(rnd)}")
    print(f"\n## {nm}\n  "+"\n  ".join(line)); [print(e) for e in exl]
pd.DataFrame(rows).to_csv(f"{SP}/sva_dom_film_markers.csv",index=False,encoding="utf-8-sig")
