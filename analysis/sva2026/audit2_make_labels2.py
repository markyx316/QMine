# -*- coding: utf-8 -*-
"""Audit 2 round-2 labelling record, by id in audit2_round2_list.txt (blind; same fixed criteria as round 1).

All 410 lines were read. Ids not listed were judged B. Writes audit2_labels2.py with exact-string dicts.
Judgement calls held constant from round 1: '<topic>最新消息/最新进展/后续' noun phrases are keyword-shaped (B);
a bare name + 去世 on search is a verification lookup (B); a pasted answer WITH a typed request appended is B,
a pasted answer or offer with nothing but an acknowledgement is A.
"""
import pandas as pd

SP = __import__("os").environ.get("SVA_WORKDIR", __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "work"))

A = {
    "headline": [2005, 2006, 2007, 2010, 2019, 2020, 2021, 2022, 2025, 2026, 2030, 2031, 2037, 2040, 2041, 2045, 2046,
                 2049, 2050, 2052, 2056, 2058, 2063, 2064, 2065, 2066, 2071, 2072, 2073, 2082, 2084, 2089, 2096, 2115,
                 2116, 2118, 2119, 2120, 2121, 2126, 2129, 2131, 2132, 2133, 2134, 2135, 2142, 2144, 2145, 2146, 2148,
                 2153, 2159, 2162, 2169, 2170, 2181, 2182, 2183, 2190, 2194, 2195, 2197, 2199, 2203, 2204, 2208, 2212,
                 2214, 2216, 2217, 2218, 2220, 2223, 2227, 2228, 2231, 2233, 2234, 2237, 2238, 2245, 2251, 2253, 2259,
                 2260, 2262, 2265, 2266, 2268, 2269, 2277, 2281, 2287, 2288, 2301, 2303, 2310, 2326, 2328, 2334, 2342,
                 2343, 2345, 2348, 2349, 2353, 2362, 2366, 2369, 2378, 2382, 2384, 2390, 2392, 2396, 2397, 2398, 2404,
                 2407],
    "feed_copy": [2013, 2017, 2033, 2057, 2069, 2079, 2092, 2124, 2141, 2175, 2178, 2215, 2242, 2276, 2278, 2306, 2313,
                  2321, 2333, 2337, 2359, 2360, 2365, 2368, 2371, 2373, 2374, 2395, 2401],
    "template": [2004, 2028, 2029, 2035, 2078, 2080, 2101, 2136, 2168, 2179, 2184, 2246, 2372, 2391, 2406],
    "chip": [2062, 2074, 2076, 2112, 2122, 2279, 2300, 2311, 2357, 2380, 2393, 2402, 2409],
    "card": [2008, 2027, 2113, 2188, 2221, 2230, 2304],
    "content_free": [2061, 2085, 2093, 2107, 2110, 2130, 2138, 2161, 2354],
}
C = {
    "origin_equal": [2018, 2088, 2154, 2156, 2167, 2244, 2249, 2263, 2340, 2355, 2389, 2405],
    "fragment": [2189, 2298, 2336],
}

u = pd.read_parquet(f"{SP}/audit2_round2_list.parquet")
assert u["id"].is_unique and u["id"].min() == 2001 and u["id"].max() == 2410
ids = {}
for lab, groups in (("A", A), ("C", C)):
    for reason, lst in groups.items():
        assert len(lst) == len(set(lst)), (lab, reason)
        for i in lst:
            assert 2001 <= i <= 2410, i
            assert i not in ids, f"id {i} labelled twice"
            ids[i] = (lab, reason)
la, ls, wa, ws = {}, {}, {}, {}
for r in u.itertuples():
    lab, reason = ids.get(r.id, ("B", "user"))
    if r.surface == "assistant":
        la[r.qs], wa[r.qs] = lab, reason
    else:
        ls[r.qs], ws[r.qs] = lab, reason
with open(__import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "audit2_labels2.py"), "w", encoding="utf-8") as fh:
    fh.write('# -*- coding: utf-8 -*-\n"""Audit 2 ROUND-2 labels (U05 census, rule holdouts, 2025 search replay, decoys),\n'
             'keyed by EXACT string. Generated from audit2_make_labels2.py."""\nimport pandas as pd\n\n')
    fh.write(f"LAB2_ASSISTANT = {la!r}\n\nLAB2_SEARCH = {ls!r}\n\nREASON2_ASSISTANT = {wa!r}\n\nREASON2_SEARCH = {ws!r}\n\n")
    fh.write('''
def check(samples: pd.DataFrame) -> None:
    for surf, lab in (("assistant", LAB2_ASSISTANT), ("search", LAB2_SEARCH)):
        rows = set(samples.loc[samples.surface == surf, "qs"])
        bad = [k for k in lab if k not in rows]
        assert not bad, f"{surf}: {len(bad)} keys match no row"
        miss = [q for q in rows if q not in lab]
        assert not miss, f"{surf}: {len(miss)} rows unlabelled"
''')
print("A:", sum(map(len, A.values())), "C:", sum(map(len, C.values())), "B:", len(u) - len(ids))
