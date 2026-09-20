# -*- coding: utf-8 -*-
"""教育: pairs with cosine (fixed), exact user ranks, taxonomy class lists, English-phrase spread, OCR/voice tests."""
import os; os.environ["HF_HUB_OFFLINE"]="1"
from sva_dom_edu_lib import *
import torch
from sentence_transformers import SentenceTransformer
a,s=load(); e=edu(); U=users(e)
C=cells(a,s,"教育"); S10=C["搜索top10k"].reset_index(drop=True); S10["query"]=S10["query"].astype(str)
srank={q:i+1 for i,q in enumerate(S10["query"])}; spv=dict(zip(S10["query"],S10.wise_pv))
m=SentenceTransformer("BAAI/bge-base-zh-v1.5",device="mps" if torch.backends.mps.is_available() else "cpu")
enc=lambda q: m.encode(list(q),batch_size=64,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False)
PAIRS=[("取现成内容→让助手生成","拼豆图纸","帮我画 拼豆图纸"),("取现成内容→让助手生成","防溺水手抄报","帮我画 防溺水手抄报"),("取现成内容→让助手生成","作文","帮我写一篇作文"),("取现成内容→让助手生成","暑假","关于暑假的作文600字"),
 ("工具名→带方向的指令","翻译","翻译成中文"),("工具名→带方向的指令","拼音","加拼音"),("工具名→带方向的指令","簟字怎么读","这个字怎么读"),
 ("裸校名→加专业/地区约束","山西财经大学","山西财经大学财政学专业就业前景如何"),("裸校名→加专业/地区约束","成都理工大学","成都理工大学在湖北的录取分数大概是多少"),("裸校名→加专业/地区约束","上海大学","上海大学哪个专业最受欢迎"),("裸校名→加专业/地区约束","太原理工大学","太原理工大学安全工程怎么样"),
 ("加入自身处境求判断","高考","平时400多分，高考能上本科吗"),("加入自身处境求判断","专升本","专科期间考过六级对专升本有帮助吗"),("加入自身处境求判断","研究生报考条件与要求2026","考研2026年340分，能报什么学校2027材料科学与工程985"),
 ("概念→具体情境对比","学硕和专硕的区别","体育专业考研，学硕和专硕的区别"),("概念→具体情境对比","单招是什么意思啊","单招和高考的区别是什么"),("概念→具体情境对比","高考志愿填报","高考平行志愿怎么填"),
 ("要谜底→要助手再出题","斗方名士打一肖","能再给我几个类似的生肖谜语吗"),("写出字→口述字形","三点水的字","三点水加个卒字念什么"),("价目查询→指名机构","糕点培训班学费价目表","新东方西点学费价目表")]
lab={}; ex=pd.read_parquet(f"{SP}/label_extra/labels.parquet"); lab.update(dict(zip(ex["query"].astype(str),ex["u_ds"]))); lab.update(dict(zip(U["search_top1000"]["query"],U["search_top1000"]["u_ds"])))
AU=pd.concat([U["assistant_top1k"],U["assistant_random1k"]])
rows=[]
for grp,sqq,aq in PAIRS:
    ar=AU[AU["query"]==aq]; ok_s=sqq in srank; ok_a=len(ar)>0
    cs=float((enc([sqq])@enc([aq]).T)[0,0]) if ok_s and ok_a else np.nan
    rows.append(dict(group=grp,search_query=sqq,search_rank=srank.get(sqq),search_pv=spv.get(sqq),search_u=lab.get(sqq),assistant_query=aq,assistant_surface=(ar.surface.iloc[0] if ok_a else None),assistant_pv=(ar.pv.iloc[0] if ok_a else None),assistant_u=(ar.u_ds.iloc[0] if ok_a else None),cos=round(cs,3)))
P=pd.DataFrame(rows); print(P.to_string(index=False)); P.to_csv(f"{SP}/sva_dom_edu_pairs.csv",index=False)
print("\nEXACT user ranks in search top10k:")
H=U["assistant_top1k"]
for q in ["翻译","百度翻译","我","一","S","拼豆图纸","南京理工大学","作文","高考","高考志愿填报","志愿填报","拼音","这个","价格","答案","怎么读","斗方名士打一肖","阳光高考网官网","学信网","中南大学"]:
    print(f"  {q}: 搜索 rank {srank.get(q)} PV {spv.get(q)} | 助手头 user PV {H.loc[H['query']==q,'pv'].sum():,.0f} u_ds {H.loc[H['query']==q,'u_ds'].tolist()}")
# taxonomy class lists
print("\nsearch edu-pool classes (top10k user, all):",S10.td_l1_name.value_counts().to_dict())
AA=e[(e.surface!="search_top1000")&(e.tier=="user")]
print("assistant ai04 classes in 教育 user rows (head+tail):",AA.own_intent.value_counts().to_dict())
print("assistant ai04 distinct classes across ALL 33 categories (top1k+random1k, all tiers):",a.td_l1_name.nunique(), sorted(a.td_l1_name.dropna().unique().tolist()))
# English phrases spread
at=a.copy(); at["query"]=at["query"].astype(str)
for q in ["I don't know","Yeah","My god","Come on","No","Thank you","S","A","B"]:
    x=at[at["query"]==q]; print(f"  '{q}': rows {len(x)} categories {x.l1.nunique()} l1/l2 {x.groupby(['l1','l2']).search_num.sum().sort_values(ascending=False).head(4).to_dict()}")
# OCR / voice / follow-up tests
TT=U["assistant_random1k"]; h13=H[H.u_ds=="U13"]
TESTS={"≥8字":r"^.{8,}$","试卷格式(如图/（ ）/A．选项/题号)":r"(如图|（\s*）|\(\s*\)|[A-D][．.、]\s*\S|^\d{1,2}[\.．、]\S)","零宽字符(U+200B/C)":"[​‌]",
       "口语填充(嗯/呃/额+标点或空格,长度>8)":r"^.{0,}(嗯|呃|额)[，,。\s].{3,}$","口述字形(旁/字头/加个…念什么)":r"(旁|字头|底|加个|加一个|左边|右边|上面|下面).{0,6}(字|念什么|读什么|怎么读)",
       "单独连词(然后/但是/所以/如果/假如/其实/而且/可是)":r"^(然后|但是|所以|如果|假如|其实|而且|可是|就是|那么)[。.，,]?$"}
for nm,pat in TESTS.items():
    print(f"  {nm}: 头U13 {fmt(int(has(h13['query'],pat).sum()),len(h13))} | 头全体 {fmt(int(has(H['query'],pat).sum()),len(H))} | 尾全体 {fmt(int(has(TT['query'],pat).sum()),len(TT))}"
          +" | 例[尾] "+" | ".join(TT[has(TT['query'],pat)]['query'].head(5).str[:40]))
# search top10k removed PV share
ss=s[s.domain=="教育"]; print(f"\nsearch top10k all tiers PV {ss.wise_pv.sum():,}; non-user PV {ss.loc[ss.tier!='user','wise_pv'].sum():,} = {ss.loc[ss.tier!='user','wise_pv'].sum()/ss.wise_pv.sum()*100:.2f}%; non-user rows {int((ss.tier!='user').sum())}: "+" | ".join(ss[ss.tier!='user'].sort_values('wise_pv',ascending=False)['query'].astype(str).head(10)))
