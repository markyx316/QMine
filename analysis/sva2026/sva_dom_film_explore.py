# -*- coding: utf-8 -*-
"""Exploration: term-level hit counts for candidate marker vocabularies; date evidence; top10k own taxonomy."""
import sys, re, pandas as pd
SP=__import__("os").environ.get("SVA_WORKDIR", __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "work"))
sys.path.insert(0,SP); from sva_common import *
a,s=load(); c=cells(a,s,"影视")
F=pd.read_parquet(f"{SP}/sva_final_rows.parquet"); U=F[(F.domain=="影视")&(F.tier=="user")]
Q={"S1k":U[U.surface=="search_top1000"]["query"].astype(str).tolist(),"S10k":c["搜索top10k"]["query"].astype(str).tolist(),
   "Ahead":U[U.surface=="assistant_top1k"]["query"].astype(str).tolist(),"Atail":U[U.surface=="assistant_random1k"]["query"].astype(str).tolist()}
print("n:",{k:len(v) for k,v in Q.items()})
TERMS={
 "resource":["免费观看","在线观看","在线播放","在线看","免费看","免费播放","全集","完整版","高清","观看","百度网盘","百度云","网盘","下载","迅雷","磁力","资源","无删减","未删减","播放"],
 "live":["cctv","CCTV","央视","中央","卫视","直播","节目表","节目单","体育"],
 "platform":["哔哩哔哩","bilibili","B站","b站","樱花动漫","爱奇艺","腾讯视频","优酷","芒果","抖音","快手","西瓜视频","咪咕","影院","影视网","电影网","app","APP","官网","网站","软件","网页版"],
 "ident":["这是什么","什么电影","什么电视剧","什么动漫","什么剧","叫什么","剧名","片名","出自","哪部","哪个剧","是谁","她是谁","他是谁","图片","截图","这个视频","画面"],
 "reco":["推荐","好看的","有哪些","片单","类似","排行","榜","必看","值得看","代表作"],
 "cast":["演员表","演员","主演","扮演","饰演","谁演","男主","女主","配音","导演"],
 "plot":["结局","剧情","大结局","分集","人物关系","简介","讲的是","讲了什么","原型","为什么","为何","为啥","怎么回事","关系"],
 "hypo":["如果","假如","假设","要是","会怎样","会发生什么","能赢吗","打得过","谁厉害","谁更强","谁强"],
 "fic":["续写","同人","番外","写一篇","写一个","写个","帮我写","剧本","分镜","脚本","角色扮演","扮演","设定","人设","oc","OC","小说"],
 "episode":["第1集","集","第一季","季","哪一集","几集","多少集","最新一集","更新"],
 "gen":["生成","视频","AI视频","比例为","做成视频","变成视频","帮我画","画一"],
 "kids":["奥特曼","迪迦","赛罗","喜羊羊","灰太狼","懒羊羊","熊出没","熊大","熊二","光头强","小猪佩奇","佩奇","汪汪队","超级飞侠","猪猪侠","海绵宝宝","小马宝莉","巴啦啦","小花仙","叶罗丽","宝宝巴士","贝乐虎","蜡笔小新","哆啦A梦","葫芦娃","黑猫警长","开心超人","果宝特攻","铠甲勇士","星卡梦少女","变形金刚","猫和老鼠","宝可梦","斗罗大陆","柯南","动画片","动画","小学生","孩子","岁"],
 "shortdrama":["短剧","竖屏","霸总","甜宠","神豪","逆袭","重生"],
 "feed":["推送","给我推","刷到","首页","算法","屏蔽","不想看","推荐给我","老是推","一直推"],
 "hot_s":["爱情有烟火","千香","莫离","昨夜将至","问心","炽夏","检察官的提案","南部档案","四渡","主角","云秀行","人间中毒"],
 "hot_a":["早春晴朗","醒来","重器","藏锋","花开锦绣","我们的少年时代","牛来","卖房子的女人","抓特务","聊斋"],
}
for g,ts in TERMS.items():
    print(f"\n## {g}")
    for t in ts:
        print(f"  {t:<8} "+"  ".join(f"{k} {sum(t in q for q in v):>4} ({sum(t in q for q in v)/len(v)*100:4.1f}%)" for k,v in Q.items()))
# date evidence in the assistant log (all categories, all tiers) and in the 影视 cells
A=a["query"].astype(str)
for pat in [r"2026年[789]月",r"2026年9月",r"2026年8月",r"2026年7月",r"8月\d{1,2}日",r"9月\d{1,2}日",r"7月\d{1,2}日"]:
    m=A.map(lambda q: bool(re.search(pat,q)))
    print(f"assistant all cats {pat}: {int(m.sum())} rows; e.g. "+" | ".join(A[m].head(4).str[:40]))
S=s["query"].astype(str)
for pat in [r"2026年[789]月",r"8月\d{1,2}日",r"7月\d{1,2}日",r"6月\d{1,2}日"]:
    m=S.map(lambda q: bool(re.search(pat,q))); print(f"search all verticals {pat}: {int(m.sum())}; e.g. "+" | ".join(S[m].head(4).str[:40]))
# top10k own taxonomy
x=c["搜索top10k"]; g=x.groupby("td_l1_name").agg(k=("query","size"),pv=("wise_pv","sum")).sort_values("k",ascending=False)
g["row%"]=g.k/len(x)*100; g["pv%"]=g.pv/x.wise_pv.sum()*100; print("\n## search top10k own taxonomy n=",len(x)); print(g.round(1).to_string())
x1=x.head(1000); g1=x1.groupby("td_l1_name").size()/len(x1)*100; print("top1000 (cells) row% check:"); print(g1.round(1).sort_values(ascending=False).to_string())
# hot-title ranks in search top10k
rk={q:i+1 for i,q in enumerate(x["query"].astype(str))}
for t in TERMS["hot_a"]+TERMS["hot_s"]:
    hits=[(q,r) for q,r in rk.items() if t in q]
    print(f"  {t}: search10k hits {len(hits)}, best rank {min([r for _,r in hits]) if hits else None}; assistant head hits {sum(t in q for q in Q['Ahead'])}, tail {sum(t in q for q in Q['Atail'])}")
