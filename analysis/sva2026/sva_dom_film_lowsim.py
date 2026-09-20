# -*- coding: utf-8 -*-
"""影视 §5: assistant user rows whose nearest search-top10k neighbour has sim<0.70, themed by a stated priority rule;
contrast with rows sim≥0.80. Wilson CIs. Adult-like rows counted but never printed."""
import sys, re, math, pandas as pd
SP=__import__("os").environ.get("SVA_WORKDIR", __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "work"))
sys.path.insert(0,SP); import sva_dom_film_rx as R
def wilson(k,n,z=1.96):
    if n==0: return float('nan'),float('nan'),float('nan')
    p=k/n; den=1+z*z/n; c=(p+z*z/(2*n))/den; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den; return p,max(0,c-h),min(1,c+h)
NN=pd.read_parquet(f"{SP}/sva_dom_film_nn.parquet"); NN["query"]=NN["query"].astype(str)
VID=re.compile(r"(视频|教程|演示|片段|名场面|台词|歌|主题曲|插曲|BGM|bgm|混剪|剪辑)")
def theme(q,u,o):
    h=lambda n: R.hit(n,q)
    if h("视频/图像生成请求") or (u=="U10" and re.search(r"(视频|画|图|照片)",q) and not (h("假设/对战") or h("创作/续写/设定"))): return "T1 生成/编辑视频图像"
    if h("假设/对战") or h("创作/续写/设定") or u=="U10" or (len(q)>60 and "虚构续写" in str(o)): return "T2 假设·对战·同人设定·续写"
    if h("识别作品/人物") or h("指代画面/上文"): return "T3 识别作品/画面/人物"
    if h("平台运营/创作者问题") or u=="U06": return "T5 平台账号与创作者操作"
    if h("剧情/结局") or h("因果追问(为什么)") or h("演员/角色") or u in ("U04","U05","U07"): return "T4 剧情·人物·因果讨论与评判"
    if u in ("U08","U09","U02","U01","U03") and VID.search(q) and not h("播放页/资源词"): return "T6 求'视频'形态的非片源内容(教程/片段/台词/歌)"
    if u in ("U08","U09","U02","U01","U03"): return "T7 找片·片单·事实(与搜索同类但措辞远)"
    if u=="U11" or h("对助手说话"): return "T8 会话·情绪·对助手说话"
    if u=="U13": return "T9 无法判定碎片/贴文"
    return "T0 其他"
NN["theme"]=[theme(q,u,o) for q,u,o in zip(NN["query"],NN.u_ds,NN.own_intent)]
NN["adult"]=[R.adult_like(q,u,o) for q,u,o in zip(NN["query"],NN.u_ds,NN.own_intent)]
NN["band"]=pd.cut(NN.sim,[-1,0.7,0.8,0.999,2],right=False,labels=["远<0.70","中0.70–0.80","近0.80–0.999","同≥0.999"])
out=[]
for sf in ("assistant_top1k","assistant_random1k"):
    x=NN[NN.surface==sf]; lo_=x[x.sim<0.7]; hi_=x[x.sim>=0.8]; n=len(x)
    print(f"\n######## {sf}: n={n}; 远<0.70 {len(lo_)} ({len(lo_)/n*100:.1f}%) ; ≥0.80 {len(hi_)}")
    print("  u_ds × band (row%):"); print((pd.crosstab(x.u_ds,x.band,normalize="columns")*100).round(1).to_string())
    for t in sorted(NN.theme.unique()):
        k=int((lo_.theme==t).sum()); p,l,u=wilson(k,len(lo_)); pa,la,ua=wilson(k,n); kh=int((hi_.theme==t).sum()); ph,lh,uh=wilson(kh,len(hi_))
        pv=(lo_.pv[lo_.theme==t].sum()/x.pv.sum()) if sf=="assistant_top1k" else float('nan')
        out.append(dict(surface=sf,theme=t,n_low=len(lo_),k_low=k,share_low=p,lo_low=l,hi_low=u,n_all=n,share_all=pa,lo_all=la,hi_all=ua,n_high=len(hi_),k_high=kh,share_high=ph,pv_share_all=pv,adult_in_theme=int((lo_.adult&(lo_.theme==t)).sum())))
        print(f"  {t:<40} 远行中 k={k:>3} {p*100:5.1f}[{l*100:4.1f}-{u*100:4.1f}] | 占全体 {pa*100:4.1f}[{la*100:4.1f}-{ua*100:4.1f}] | 近邻≥0.80中 k={kh} {ph*100:4.1f}%"+(f" | 头部PV {pv*100:.1f}%" if sf=='assistant_top1k' else "")+f" | 其中成人暗示 {int((lo_.adult&(lo_.theme==t)).sum())}")
        ex=lo_[(lo_.theme==t)&~lo_.adult]
        if len(ex): print("      例: "+" | ".join(f"{q[:38]}(→{nq[:10]},{s:.2f})" for q,nq,s in zip(ex.sort_values('pv',ascending=False)["query"].head(3),ex.sort_values('pv',ascending=False).nn_query.head(3),ex.sort_values('pv',ascending=False).sim.head(3)))+" ‖ "+" | ".join(f"{q[:38]}" for q in ex["query"].sample(min(5,len(ex)),random_state=9)))
pd.DataFrame(out).to_csv(f"{SP}/sva_dom_film_lowsim_themes.csv",index=False,encoding="utf-8-sig")
NN[["surface","query","pv","u_ds","sim","nn_query","nn_rank","band","theme","adult"]].to_parquet(f"{SP}/sva_dom_film_nn_themes.parquet",index=False)
