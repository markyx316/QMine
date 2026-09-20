# -*- coding: utf-8 -*-
"""教育: length-controlled low-sim vs search-internal controls; low-sim themes; matched pairs with cosine."""
import os; os.environ["HF_HUB_OFFLINE"]="1"
from sva_dom_edu_lib import *
import torch
from sentence_transformers import SentenceTransformer
a,s=load(); e=edu(); U=users(e)
C=cells(a,s,"教育"); S10=C["搜索top10k"].reset_index(drop=True); S10["query"]=S10["query"].astype(str); sq=S10["query"].tolist()
m=SentenceTransformer("BAAI/bge-base-zh-v1.5",device="mps" if torch.backends.mps.is_available() else "cpu")
enc=lambda q: m.encode(list(q),batch_size=256,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False).astype(np.float32)
def nn(E,R):
    best=np.full(len(E),-1.0,np.float32)
    for i in range(0,len(E),512):
        sim=E[i:i+512]@R.T; best[i:i+512]=sim.max(1)
    return best
ES=enc(sq)
cA=pd.DataFrame({"query":sq[:1000],"sim":nn(ES[:1000],ES[1000:])}); t=len(ES)-1000; cB=pd.DataFrame({"query":sq[t:],"sim":nn(ES[t:],ES[:t])})
R=pd.read_csv(f"{SP}/sva_dom_edu_nn_rows.csv"); R["query"]=R["query"].astype(str)
LB=[(1,2,"1–2字"),(3,6,"3–6字"),(7,10,"7–10字"),(11,15,"11–15字"),(16,25,"16–25字"),(26,9999,">25字")]
print("LENGTH-CONTROLLED share with nearest search-top10k neighbour sim<0.70 (Wilson):")
for lab,d in [("对照A 搜索top1000→1001+",cA),("对照B 搜索最深1000→其余",cB),("助手头",R[R.surface=="assistant_top1k"]),("助手头(非U13)",R[(R.surface=="assistant_top1k")&(R.u_ds!="U13")]),("助手尾",R[R.surface=="assistant_random1k"])]:
    L=d["query"].str.len(); parts=[]
    for lo,hi,nm in LB:
        x=d[(L>=lo)&(L<=hi)]
        if len(x)>=20: k=int((x.sim<0.70).sum()); w=wilson(k,len(x)); parts.append(f"{nm} n={len(x)} {k/len(x)*100:.0f}% [{w[0]:.0f}–{w[1]:.0f}]")
        else: parts.append(f"{nm} n={len(x)} —")
    print(f"  {lab}: "+" | ".join(parts))
# low-sim themes (interpretable rows)
TH={"写作生成(作文/读后感/日记/仿写/续写/演讲稿/寄语)":r"(作文|读后感|观后感|日记|周记|仿写|续写|扩写|改写|演讲稿|寄语|感言|书信|一封信|写一|写个|写篇|帮我写)",
    "绘图/手抄报/简笔画生成":r"(帮我画|画一|手抄报|简笔画|思维导图|小报|配画|绘画|画画)",
    "题目求解与答案(这道题/答案/竖式/计算/选择)":r"(这道题|这题|题目|答案|解题|第.题|第.问|选择题|填空|阅读题|竖式|脱式|计算|方程|解析|\d+[\+\-×÷\*/]\d+)",
    "跟进改写(再/换/简化/缩短/改成/字数)":r"(^(再|换|简化|简洁|缩短|精简|重写|重新|继续|直接|不用)|改成|改到|改得|更简单|更短|更有|字数|上面|刚才|上一)",
    "字词与语言知识(拼音/读音/组词/意思/翻译/语法)":r"(拼音|读音|怎么读|组词|造句|意思|近义词|反义词|词性|成语|翻译|英文|英语|单词|语法|从句|笔顺|偏旁|部首)",
    "具体学校/机构事实(分数线/招生/学费/升学率/开学)":r"(中学|小学|学校|学院|大学|一中|附中|实验|幼儿园|机构|新东方|培训).{0,14}(分数|录取|招生|简章|学费|收费|价目|分班|名单|电话|地址|放假|开学|升学率|师资|派位|摇号|排名|口碑|怎么样|优势)",
    "升学/专业/职业选择与个案判断":r"(志愿|专业|就业|前景|报考|选科|哪个好|哪个更|更适合|值得|复读|能上|能报|有帮助|有前途|考公|考编|事业编|公务员|教资|证)"}
for lab,d in [("助手头 低相似(<0.70,非U13)",R[(R.surface=="assistant_top1k")&(R.u_ds!="U13")&(R.sim<0.70)]),("助手尾 低相似(<0.70,非U13)",R[(R.surface=="assistant_random1k")&(R.u_ds!="U13")&(R.sim<0.70)])]:
    print(f"\n{lab}: n={len(d)}")
    for nm,pat in TH.items():
        mm=has(d["query"],pat); k=int(mm.sum()); exm=d[mm].sort_values("pv",ascending=False).head(8) if "头" in lab else d[mm].sample(min(10,k),random_state=13)
        print(f"   {nm}: {fmt(k,len(d))} 例 "+" | ".join(q[:40] for q in exm['query']))
    anyt=np.zeros(len(d),bool)
    for pat in TH.values(): anyt|=has(d["query"],pat).values
    print(f"   任一主题: {fmt(int(anyt.sum()),len(d))}; 未归入任何主题例: "+" | ".join(d[~anyt].sample(min(15,int((~anyt).sum())),random_state=1)['query'].str[:36]))
# matched pairs
PAIRS=[("取现成内容→让助手生成","拼豆图纸","帮我画 拼豆图纸"),("取现成内容→让助手生成","防溺水手抄报","帮我画 防溺水手抄报"),("取现成内容→让助手生成","作文","帮我写一篇作文"),("取现成内容→让助手生成","暑假","关于暑假的作文600字"),
 ("工具名→带方向的指令","翻译","翻译成中文"),("工具名→带方向的指令","拼音","加拼音"),("工具名→带方向的指令","簟字怎么读","这个字怎么读"),
 ("裸校名→加专业/地区约束","山西财经大学","山西财经大学财政学专业就业前景如何"),("裸校名→加专业/地区约束","成都理工大学","成都理工大学在湖北的录取分数大概是多少"),("裸校名→加专业/地区约束","上海大学","上海大学哪个专业最受欢迎"),("裸校名→加专业/地区约束","太原理工大学","太原理工大学安全工程怎么样"),
 ("加入自身处境求判断","高考","平时400多分，高考能上本科吗"),("加入自身处境求判断","专升本","专科期间考过六级对专升本有帮助吗"),("加入自身处境求判断","研究生报考条件与要求2026","考研2026年340分，能报什么学校2027材料科学与工程985"),
 ("概念→具体情境对比","学硕和专硕的区别","体育专业考研，学硕和专硕的区别"),("概念→具体情境对比","单招是什么意思啊","单招和高考的区别是什么"),("概念→具体情境对比","高考志愿填报","高考平行志愿怎么填"),
 ("要谜底→要助手再出题","斗方名士打一肖","能再给我几个类似的生肖谜语吗"),("写出字→口述字形","三点水的字","三点水加个卒字念什么"),("价目查询→指名机构","糕点培训班学费价目表","新东方西点学费价目表")]
srank={q:i+1 for i,q in enumerate(sq)}; spv=dict(zip(S10["query"],S10.wise_pv))
lab={}; ex=pd.read_parquet(f"{SP}/label_extra/labels.parquet"); lab.update(dict(zip(ex["query"].astype(str),ex["u_ds"]))); lab.update(dict(zip(U["search_top1000"]["query"],U["search_top1000"]["u_ds"])))
AU=pd.concat([U["assistant_top1k"],U["assistant_random1k"]])
rows=[]
for grp,sqq,aq in PAIRS:
    ar=AU[AU["query"]==aq]; ok_s=sqq in srank; ok_a=len(ar)>0
    cs=float(enc([sqq])@enc([aq]).T) if ok_s and ok_a else np.nan
    rows.append(dict(group=grp,search_query=sqq,search_rank=srank.get(sqq),search_pv=spv.get(sqq),search_u=lab.get(sqq),assistant_query=aq,assistant_surface=(ar.surface.iloc[0] if ok_a else None),assistant_pv=(ar.pv.iloc[0] if ok_a else None),assistant_u=(ar.u_ds.iloc[0] if ok_a else None),cos=cs))
P=pd.DataFrame(rows); print("\nPAIRS:"); print(P.to_string(index=False)); P.to_csv(f"{SP}/sva_dom_edu_pairs.csv",index=False)
