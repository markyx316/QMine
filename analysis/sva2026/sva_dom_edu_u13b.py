# -*- coding: utf-8 -*-
"""教育 head: shape buckets for U13 rows (first match wins), and the same buckets over ALL head user rows."""
from sva_dom_edu_lib import *
e=edu(); U=users(e); H=U["assistant_top1k"].copy()
print("assistant own_run:",e[e.surface!="search_top1000"].own_run.unique(), "| search own_run:",e[e.surface=="search_top1000"].own_run.unique())
B=[
 ("A 英文短语(语言培训)", r"^[A-Za-z]{2,}[A-Za-z\s'’,.!?]*$"),
 ("B 单个拉丁字母", r"^[A-Za-z]$"),
 ("C 作答/解题跟进(题目不在文本中)", r"(答案|解答|解题|答题|做题|这道题|这一题|这题|第[一二三四五六七八九十\d]+题|选什么|多选题|单选题|填空|竖式|脱式|列式|列算式|算式|解方程|过程|解析|^回答$|^答[。.]?$|^题目$|的方法$|算术法)"),
 ("D 识图/指代识别(这是谁/这是什么字…)", r"^(这是谁|谁|这是什么字|这是什么学校|这是哪个学校|这是哪个大学|哪个学校|哪里|这首诗的作者是谁|这句诗出自哪里|这句话出自哪里|这首诗的全文是什么|什么字|这个怎么读|这个字怎么读|这个字念什么|这个字读什么|这个是什么意思)"),
 ("E 字词/诗文工具短指令(无对象)", r"^(怎么读|怎么写|怎么念|怎么拼|读音|读什么|念什么|意思|什么意思|啥意思|中文意思|原文|出处|解释|带拼音|翻译|赏析|作批注|写批注|仿写|造句|下一句|阅读感想|写出来|写后续|这首诗表达|英文版|的英文|报名英语|\d+词$|有错别字)"),
 ("F 字数短答/字数询问", r"^(\d+|[两三四五六七八九十几])(个)?(字|词)(左右|的)?[。.？?]?$|多少字|^少字$|^文$"),
 ("G 选项数字/年级短答", r"^([一二三四五六七八九十]|\d{1,2}|[一二三四五六]年级|初一水平|初中水平)[。.]?$"),
 ("H 属性词短答(价格/年龄/地址…)", r"^(价格|价钱|价位|价位多少|学费多少|年龄|地址|品牌|型号|位置|人物|作者|来源|作用|用法|成分|内容|介绍|教程|对比|检查|整理|分析|打分|评分|预习|概括|主旨|全部|推荐|一个|两个|个)[。.？?]?$"),
 ("I 改写/缩短/重做指令碎片", r"^(简单|简便|少|短|减少|缩小|最少|晚一点|随便|重写|修改|添加|回复|一句话)[。.]?$"),
 ("J 图像编辑/绘图指令碎片", r"(上色|上颜色|填色|涂色|去字|去裙|高清|正面|清晰|放大|^图$|光线|衣襟|^裸$|躺下|跪下|坐下|站直|站着|^关闭$|怎么画|^播放$)"),
 ("K 生肖谜语碎片", r"生肖"),
 ("L 是非/拒绝/情绪回应", r"^(不要|不知道|不行|没事|不|错|否|真假|一样|还行|算了|准|吗|没|行|对|为什么)[。.？?！!]*$"),
 ("M 指代词单独(这个/那个/这/那/他)", r"^(这个|那个|这|那|他|她|它)[。.？?]?$"),
 ("N 其余1–2字碎片(疑似语音截断/未打完)", r"^.{1,2}$"),
]
def bucket(q):
    for name,pat in B:
        if re.search(pat,q): return name
    return "O 其他"
H["bucket"]=H["query"].map(bucket)
h13=H[H.u_ds=="U13"]
print(f"\nhead U13 n={len(h13)} PV={h13.pv.sum():,.0f}")
rows=[]
for name,g in h13.groupby("bucket"):
    lo,hi=wilson(len(g),len(h13)); pvp=g.pv.sum()/h13.pv.sum()*100
    rows.append(dict(bucket=name,k=len(g),n=len(h13),pct=len(g)/len(h13)*100,lo=lo,hi=hi,pv_pct=pvp,l2=g.l2.value_counts().to_dict()))
    print(f"  {name}: {len(g)/len(h13)*100:.1f}% [{lo:.1f}–{hi:.1f}] k={len(g)} PV {pvp:.1f}% | l2 {g.l2.value_counts().to_dict()} | 例 "+" | ".join(f"{q}({p:,.0f})" for q,p in zip(g.sort_values('pv',ascending=False)['query'].head(14),g.sort_values('pv',ascending=False).pv.head(14))))
pd.DataFrame(rows).to_csv(f"{SP}/sva_dom_edu_u13_buckets.csv",index=False)
# groups
G={"依赖上文或图片的指令/追问(C+D+E+I+J+M)":list("CDEIJM"),"对助手提问的短答(B+F+G+H+L)":list("BFGHL"),"英文口语练习回应(A)":["A"],"生肖谜语碎片(K)":["K"],"其余1–2字碎片(N)":["N"],"其他(O)":["O"]}
print("\nGROUPS within head U13:")
for gname,letters in G.items():
    m=h13.bucket.str[0].isin(letters); k=int(m.sum()); lo,hi=wilson(k,len(h13))
    print(f"  {gname}: {k/len(h13)*100:.1f}% [{lo:.1f}–{hi:.1f}] k={k} PV {h13.loc[m,'pv'].sum()/h13.pv.sum()*100:.1f}%")
print("\nSAME BUCKETS over ALL head user rows (n=%d): share, and u_ds within bucket"%len(H))
for name,g in H.groupby("bucket"):
    print(f"  {name}: {fmt(len(g),len(H))} PV {g.pv.sum()/H.pv.sum()*100:.1f}% | u_ds {g.u_ds.value_counts().to_dict()}")
for gname,letters in G.items():
    m=H.bucket.str[0].isin(letters); print(f"  [ALL] {gname}: {fmt(int(m.sum()),len(H))} PV {H.loc[m,'pv'].sum()/H.pv.sum()*100:.1f}%")
# O examples
o=h13[h13.bucket=="O 其他"].sort_values("pv",ascending=False); print("\nO 其他 examples: "+" | ".join(o['query'].head(40)))
# search head U13 rows
s13=U["search_top1000"][U["search_top1000"].u_ds=="U13"].sort_values("pv",ascending=False)
print("\nsearch head U13 (n=%d): "%len(s13)+" | ".join(f"{q}({p:,.0f})" for q,p in zip(s13['query'],s13.pv)))
H.to_csv(f"{SP}/sva_dom_edu_head_buckets.csv",index=False,columns=["query","pv","l2","u_ds","p_ds","own_intent","own_leaf","bucket"])
