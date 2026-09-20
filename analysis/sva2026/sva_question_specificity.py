# -*- coding: utf-8 -*-
import sys; sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from sva_common import *
a,s=load()
Q=r"(怎么|如何|为什么|什么|哪个|哪家|哪些|多少|吗|呢|是不是|能不能|可不可以|\?|？)"
SUB={"因果(为什么/怎么回事)":r"(为什么|为啥|怎么回事|什么原因|原因是)",
 "方法(怎么/如何/怎样)":r"(怎么(做|办|弄|用|治|写|查|算|去|买|选|设置|处理|才能|吃|喝|调理|消除|恢复)|如何|怎样|咋)",
 "是非核实(是不是/是否/真的/吗)":r"(是不是|是否|真的|吗[？?。，,]?$|对吗|对不对)",
 "可行许可(能不能/可以/能否)":r"(能不能|可不可以|可以.{0,6}吗|能否|能.{0,6}吗)",
 "选择清单(哪个/哪些/哪家)":r"(哪个|哪些|哪家|哪种|哪里|哪儿)",
 "数量(多少/几/多久)":r"(多少|几(个|次|天|岁|年|周|月|粒|片|号|点)|多久|多长)",
 "定义(什么意思/是什么)":r"(什么意思|是什么|啥意思|指什么)"}
SPEC={"带数量+单位":r"\d+(\.\d+)?\s?(岁|周|天|个月|年|mg|毫克|克|斤|公斤|kg|元|块|万|千|cm|厘米|米|分|次|粒|片|度|℃|mmol|%)",
 "带具体日期":r"(\d{4}年\d{1,2}月|\d{1,2}月\d{1,2}(日|号)|昨天|上周|下周|前天)",
 "带地点(省市县区镇)":r"[一-鿿]{1,6}(省|市|县|区|镇|乡|村|街道)",
 "具名二选一比较":r"[一-鿿A-Za-z0-9]{2,}(和|与|跟|还是)[一-鿿A-Za-z0-9]{2,}.{0,8}(哪个|区别|对比|更好|更适合|选)"}
C={d:{k:v["query"].astype(str) for k,v in cells(a,s,d).items()} for d in CAT}
P={k:pd.concat([C[d][k] for d in CAT]) for k in SURF}
print("#### n per cell (final tiers)")
for d in CAT: print(f"  {d}: "+"  ".join(f"{k} {len(v):,}" for k,v in C[d].items()))
print("  合计: "+"  ".join(f"{k} {len(v):,}" for k,v in P.items()))
for title,D in (("QUESTION SUBTYPES — row share % (and share among question rows)",SUB),("SPECIFICITY — row share %",SPEC)):
    print(f"\n######## {title}")
    for name,pat in D.items():
        print(f"\n#### {name}")
        print(f"  {'':<6}"+"".join(f"{k:>22}" for k in SURF))
        for d in list(CAT)+["合计"]:
            c=P if d=="合计" else C[d]; cells_txt=[]
            for k in SURF:
                q=c[k]; m=q.str.contains(pat,regex=True); qq=q.str.contains(Q,regex=True)
                cells_txt.append(f"{m.mean()*100:6.2f}% ({(m&qq).sum()/max(qq.sum(),1)*100:5.1f}%)" if D is SUB else f"{m.mean()*100:6.2f}%")
            print(f"  {d:<6}"+"".join(f"{t:>22}" for t in cells_txt))
        for k in ("搜索top10k","助手top1k","助手random1k"):
            q=P[k]; m=q[q.str.contains(pat,regex=True)]
            if len(m): print(f"   例[{k}] ({len(m):,}): "+" | ".join(m.sample(min(7,len(m)),random_state=21).str[:40]))
print("\n######## RANK-BAND CONTROL on CLEANED search (length median / question% / first-person%)")
FP=r"(我|我的|我家|我们|本人)"
for d in CAT:
    q=C[d]["搜索top10k"].reset_index(drop=True); out=[]
    for lo,hi in [(0,1000),(1000,3000),(3000,6000),(6000,len(q))]:
        x=q.iloc[lo:hi]; out.append(f"{x.str.len().median():.0f}/{x.str.contains(Q,regex=True).mean()*100:.1f}%/{x.str.contains(FP,regex=True).mean()*100:.1f}%")
    print(f"  {d}: rank1-1000 {out[0]} | 1001-3000 {out[1]} | 3001-6000 {out[2]} | 6001+ {out[3]}")
