# -*- coding: utf-8 -*-
"""Assert every example quoted in sva_dom_film.md exists on the stated surface/tier; print ranks/PV; sim-band CIs; save selected pairs."""
import sys, math, pandas as pd
SP=__import__("os").environ.get("SVA_WORKDIR", __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "work"))
sys.path.insert(0,SP); from sva_common import *
def wilson(k,n,z=1.96):
    p=k/n; den=1+z*z/n; c=(p+z*z/(2*n))/den; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den; return p,max(0,c-h),min(1,c+h)
a,s=load(); c=cells(a,s,"影视"); T=c["搜索top10k"].reset_index(drop=True); T["query"]=T["query"].astype(str); rank={q:i+1 for i,q in enumerate(T["query"])}
F=pd.read_parquet(f"{SP}/sva_final_rows.parquet"); U=F[(F.domain=="影视")&(F.tier=="user")].copy(); U["query"]=U["query"].astype(str)
S1=set(U[U.surface=="search_top1000"]["query"]); AH=U[U.surface=="assistant_top1k"].set_index("query"); AT=U[U.surface=="assistant_random1k"].set_index("query")
aa=a[a.l1=="影视动漫"].assign(query=lambda d: d["query"].astype(str))
EX={
"S1":["cctv5在线直播","莫离 电视剧免费观看全集高清","莫离电视剧","问心2一共多少集","昨夜将至剧情介绍","电视剧推荐2026热播的剧","家业电视剧1-40集完整版","爱情有烟火的电视剧免费观看","千香引在线观看电视剧","哔哩哔哩","樱花动漫","韩国电影","四渡赤水简介","电影","免费视频","检察官的提案","昨夜将至","问心2","西游记","熊出没","鬼灭之刃","悬案","短剧","汪汪队立大功全集免费","小猪佩奇全集免费观看","主角电视剧40集观看完整版高清","cctv5 节目表"],
"S10":["鬼灭之刃第一季","唐朝诡事录之蜀道","醒来电视剧","蜡笔小新电影","舞蹈视频","动画片儿童3-6岁免费大全","边凤梅和江德才是什么电视剧","重器电视剧免费观看全集于和伟","CCTV5世界杯直播"],
"AH":["cctv5在线直播","cctv5节目表","早春晴朗电视剧免费观看全集高清完整","藏锋","《醒来》的结局是什么","醒来演员表","给我《醒来》的完整演员表","《重器》大结局是什么","甜宠短剧推荐","神豪短剧推荐","这是什么电影","这是什么动漫","她是谁","有视频吗","请帮我生成一个视频：跳舞，比例为原比例。","早春晴朗1-24集完整百度云","日剧韩剧在线观看大全","青柠影院电视剧免费播放视频","抖音","快手","樱花动漫网imomoe官网下载","好看的动画片有什么？","吴谨言的代表作品有哪些","视频","什么电影","王上救下宫女叫什么剧","这是什么电视剧？","赛文奥特曼12集为什么被禁","牛来票房","小猪佩奇","早春晴朗电视剧","《早春晴朗》电视剧","这个角色出自哪部动漫","帮我生成视频","我爱你","「AI视频」一家三口温馨比心","电视剧醒来全部演员表","重器 电视剧 在线看","《藏锋》的结局是什么"],
"AT":["鬼灭之刃最新一集","哪个平台有《汪汪队立大功》的最新一季全集","《唐朝诡事录之蜀道》什么时候播出","悬案原型案件是什么","在熊出没第一部光头强为何开枪不打死熊大熊二？","86版西游记和日版哪个更受欢迎","假如蜡笔小新里的美伢和广智看到原版的邋遢大王奇遇记和修复版的是什么反应？","天神古兹的光头强打得过鬼灭之刃的无惨吗？","帮我生成这只狗跳舞的视频，时长1分钟","推荐几部适合6岁孩子的教育电影","如果灰太狼提前获得时间规则会怎样","诛仙里的张小凡是不是就是鬼厉","第一季和第二季的剧情连续吗？","孟飞和向南最后结局谁更惨","帝工企业犯罪组织是什么","如何去掉哔哩哔哩视频的水印","我的抖音视频播放量有1.7万，怎么没收益？","能推荐一个适合新手的教程视频吗","哪里可以找到草莓牛奶安静书视频教程","乡村情景剧相亲片段","请帮我生成一个视频：夕阳中散步，比例为原比例。","这套cos服出自哪部动漫或游戏","能帮我确认一下这张图出自哪部越剧吗","再推荐几部不同的","小时候熊出没看多了，一看到有人在锯树，就说哎呦妈呀，这哪来的光头强呀？","有没有适合低年级小学生的纪录片","电视剧重器一共多少集","张予曦演了哪些电视剧？","《醒来》第二集有哪些重要细节","如果 黑暗特利迦和欧布奥特曼联手打合成兽斯菲亚雷德王 能赢吗","生成白狼母子寻崽子桥段","汉尼拔的结局是什么"],
}
REM={"top1k":["生成视频","帮我生成一个视频","滤镜","👌 好的，继续吧","哈哈哈","能否分享一些类似《萌宝加油站》的情景剧？","「AI视频」回忆中老照片应有的样子"],"random1k":["有没有更多有感情线的推理日剧","如何把推荐他人的作品关掉"]}
bad=[]
for q in EX["S1"]:
    if q not in S1: bad.append(("S1",q))
    else: print(f"  S1 {q} #{rank[q]}")
for q in EX["S10"]:
    if q not in rank: bad.append(("S10",q))
    else: print(f"  S10 {q} #{rank[q]}")
for nm,D in (("AH",AH),("AT",AT)):
    for q in EX[nm]:
        if q not in D.index: bad.append((nm,q))
        else:
            r=D.loc[[q]].iloc[0]; print(f"  {nm} {q} PV{int(r.pv)} {r.u_ds}")
for snap,L in REM.items():
    for q in L:
        y=aa[(aa.snapshot==snap)&(aa["query"]==q)]
        if not len(y) or (y.tier=="user").all(): bad.append(("REM "+snap,q))
        else: print(f"  REM {snap} {q} PV{int(y.search_num.max())} tiers {y.tier.unique().tolist()}")
print("MISSING:",bad)
assert not bad, bad
NN=pd.read_parquet(f"{SP}/sva_dom_film_nn.parquet")
for sf in ("assistant_top1k","assistant_random1k"):
    x=NN[NN.surface==sf]; n=len(x)
    for lab,m in (("同≥0.999",x.sim>=0.999),("近0.80–0.999",(x.sim>=0.8)&(x.sim<0.999)),("中0.70–0.80",(x.sim>=0.7)&(x.sim<0.8)),("远<0.70",x.sim<0.7)):
        p,lo,hi=wilson(int(m.sum()),n); print(f"  {sf} {lab}: {p*100:.1f}[{lo*100:.1f}–{hi*100:.1f}] k={int(m.sum())} n={n}")
P=[("原样照搬","search_top1000","cctv5在线直播","assistant_top1k","cctv5在线直播"),
   ("加限定：版本/时效","search_top10k","鬼灭之刃第一季","assistant_random1k","鬼灭之刃最新一集"),
   ("加限定：平台+最新季","search_top1000","汪汪队立大功全集免费","assistant_random1k","哪个平台有《汪汪队立大功》的最新一季全集"),
   ("加限定：播出时间","search_top10k","唐朝诡事录之蜀道","assistant_random1k","《唐朝诡事录之蜀道》什么时候播出"),
   ("从看到了解：演员/结局","search_top10k","醒来电视剧","assistant_top1k","醒来演员表 / 《醒来》的结局是什么"),
   ("从看到了解：原型","search_top1000","悬案","assistant_random1k","悬案原型案件是什么"),
   ("加入因果追问","search_top1000","熊出没","assistant_random1k","在熊出没第一部光头强为何开枪不打死熊大熊二？"),
   ("加入比较评判","search_top1000","西游记","assistant_random1k","86版西游记和日版哪个更受欢迎"),
   ("假设/对战","search_top1000","鬼灭之刃","assistant_random1k","天神古兹的光头强打得过鬼灭之刃的无惨吗？"),
   ("假设/同人","search_top10k","蜡笔小新电影","assistant_random1k","假如蜡笔小新里的美伢和广智看到原版的邋遢大王奇遇记和修复版的是什么反应？"),
   ("从找现成到要生成","search_top10k","舞蹈视频","assistant_top1k","请帮我生成一个视频：跳舞，比例为原比例。"),
   ("加受众约束并求推荐","search_top10k","动画片儿童3-6岁免费大全","assistant_random1k","推荐几部适合6岁孩子的教育电影"),
   ("文字线索→画面指代","search_top10k","边凤梅和江德才是什么电视剧","assistant_top1k","这是什么电视剧？")]
pd.DataFrame(P,columns=["change","search_surface","search_query","assistant_surface","assistant_query"]).assign(search_rank=lambda d: d.search_query.map(rank)).to_csv(f"{SP}/sva_dom_film_pairs_selected.csv",index=False,encoding="utf-8-sig")
print("saved pairs")
