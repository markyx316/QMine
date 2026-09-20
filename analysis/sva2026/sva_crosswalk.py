# -*- coding: utf-8 -*-
import sys; sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from sva_common import *
U=frame(); a,s=load()
s["U"]=[U.frame_code(RUN[d],c) for d,c in zip(s.domain,s.td_l1)]; a["U"]=[U.frame_code("ai04",c) for c in a.td_l1]
codes=list(U.FRAME); rows={}
for d in CAT:
    for k,v in cells(a,s,d).items(): rows[(d,k)]=v.U.value_counts(normalize=True).reindex(codes).fillna(0)*100
t=pd.DataFrame(rows).T; t.columns=[f"{c}{U.FRAME[c][0]}" for c in codes]
pd.set_option("display.width",320)
print(t.round(1).to_string()); print("\nPOOLED mean of 5 domains"); print(t.groupby(level=1).mean().round(1).to_string())
