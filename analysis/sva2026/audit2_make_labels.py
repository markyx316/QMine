# -*- coding: utf-8 -*-
"""Audit 2 labelling record, by id in audit2_label_list.txt (blind: id, surface, category, string).

Every one of the 1,006 lines was read. Ids not listed below were judged B (user).
This script turns the id record into literal dicts keyed by EXACT string and writes
audit2_labels.py, which asserts every key matches a sampled row.
"""
import pandas as pd

SP = __import__("os").environ.get("SVA_WORKDIR", __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "work"))

A = {
    # news-headline syntax: a complete reported-event clause / article title / hot-search tag
    "headline": [5, 7, 11, 13, 15, 18, 22, 28, 43, 45, 60, 61, 62, 75, 76, 80, 85, 87, 91, 109, 112, 121, 137,
                 151, 172, 183, 194, 200, 202, 209, 215, 220, 224, 225, 232, 252, 256, 263, 267, 272, 275, 283,
                 293, 295, 312, 318, 323, 347, 358, 360, 366, 367, 368, 375, 377, 379, 382, 408, 410, 425, 426,
                 435, 449, 451, 455, 464, 468, 472, 473, 483, 485, 491, 497, 505, 506, 549, 557, 558, 570, 585,
                 600, 608, 612, 650, 657, 659, 662, 668, 671, 673, 693, 700, 701, 727, 740, 744, 758, 760, 761,
                 767, 773, 778, 804, 810, 811, 825, 826, 827, 828, 829, 837, 842, 845, 856, 860, 863, 867, 876,
                 882, 896, 898, 906, 912, 921, 938, 941, 960, 964, 966, 975, 984, 988, 993, 1002, 1003],
    # suggested follow-up chip: 能否/有没有更/该公司/如果以上 in product voice, or strong LLM-voice follow-up
    "chip": [14, 19, 23, 32, 33, 34, 37, 39, 40, 42, 46, 49, 50, 65, 66, 81, 88, 89, 93, 97, 102, 106, 119, 127,
             128, 133, 144, 145, 150, 159, 161, 162, 170, 175, 180, 187, 190, 192, 193, 195, 203, 205, 206, 210,
             213, 216, 222, 223, 229, 230, 241, 242, 244, 249, 253, 258, 260, 261, 270, 276, 277, 279, 280, 299,
             305, 309, 313, 314, 321, 326, 328, 330, 333, 334, 336, 339, 340, 349, 374, 380, 384, 392, 395, 401,
             402, 403, 405, 406, 412, 413, 414, 417, 419, 424, 434, 440, 442, 445, 454, 474, 482, 489, 490, 493,
             508, 510, 517, 522, 524, 525, 526, 528, 529, 537, 542, 546, 551, 552, 553, 555, 560, 565, 566, 572,
             581, 583, 584, 593, 596, 597, 599, 605, 606, 609, 614, 616, 618, 626, 629, 632, 637, 638, 639, 644,
             647, 648, 652, 654, 658, 666, 672, 674, 675, 677, 680, 687, 690, 698, 702, 704, 710, 711, 715, 716,
             723, 751, 752, 759, 768, 769, 776, 781, 789, 790, 794, 800, 807, 812, 818, 830, 831, 833, 835, 836,
             838, 847, 849, 857, 869, 874, 884, 891, 893, 897, 905, 907, 909, 911, 918, 939, 951, 958, 961, 968,
             971, 989, 990, 991, 999, 1005],
    # acknowledgement / greeting / insult / "whatever" / "what?"
    "content_free": [126, 311, 353, 415, 481, 512],
    # starter prompt or slot template
    "template": [71, 779, 840],
    # feature entry or call-to-action label
    "feature": [577],
    # pasted bio card or encyclopedia/answer text
    "card": [660, 676, 709],
    # feed hashtag copy
    "feed_copy": [931],
}
C = {
    # bare fragment with no request type
    "fragment": [51, 82, 141, 198, 240, 286, 288, 351, 369, 399, 444, 466, 544, 556, 594, 611, 642, 669, 720, 762,
                 770, 772, 799, 805, 903, 953, 965, 978, 982, 994, 996, 998],
    # typed and system origin equally supported (formal follow-up without a chip prefix, a work's templated
    # evaluative question, a quick-action-shaped rewrite, a button-shaped command)
    "origin_equal": [103, 114, 149, 157, 158, 182, 246, 294, 316, 338, 387, 398, 447, 478, 479, 480, 516, 530,
                     567, 576, 603, 661, 682, 738, 746, 750, 763, 775, 850, 1004],
}

u = pd.read_parquet(f"{SP}/audit2_label_list.parquet")
assert u["id"].is_unique and len(u) == 1006
ids = {}
for lab, groups in (("A", A), ("C", C)):
    for reason, lst in groups.items():
        assert len(lst) == len(set(lst)), (lab, reason)
        for i in lst:
            assert 1 <= i <= 1006, i
            assert i not in ids, f"id {i} labelled twice"
            ids[i] = (lab, reason)
lab_as, lab_se, why_as, why_se = {}, {}, {}, {}
for r in u.itertuples():
    lab, reason = ids.get(r.id, ("B", "user"))
    if r.surface == "assistant":
        lab_as[r.qs], why_as[r.qs] = lab, reason
    else:
        lab_se[r.qs], why_se[r.qs] = lab, reason

with open(__import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "audit2_labels.py"), "w", encoding="utf-8") as fh:
    fh.write('# -*- coding: utf-8 -*-\n"""Audit 2 labels, one per distinct string per surface, keyed by EXACT string.\n'
             'Generated from the id record in audit2_make_labels.py. A = system-authored or content-free,\n'
             'B = user, C = ambiguous. REASON gives the criterion used for A and C."""\n')
    fh.write("import pandas as pd\n\n")
    fh.write(f"LAB_ASSISTANT = {lab_as!r}\n\n")
    fh.write(f"LAB_SEARCH = {lab_se!r}\n\n")
    fh.write(f"REASON_ASSISTANT = {why_as!r}\n\n")
    fh.write(f"REASON_SEARCH = {why_se!r}\n\n")
    fh.write('''
def check(samples: pd.DataFrame) -> None:
    """Every key matches a sampled row on its surface, and every sampled row has a label."""
    for surf, lab in (("assistant", LAB_ASSISTANT), ("search", LAB_SEARCH)):
        rows = set(samples.loc[samples.surface == surf, "qs"])
        missing_rows = [k for k in lab if k not in rows]
        assert not missing_rows, f"{surf}: {len(missing_rows)} label keys match no row: {missing_rows[:3]}"
        unlabelled = [q for q in rows if q not in lab]
        assert not unlabelled, f"{surf}: {len(unlabelled)} rows unlabelled: {unlabelled[:3]}"
        assert set(lab.values()) <= {"A", "B", "C"}


if __name__ == "__main__":
    SP = __import__("os").environ.get("SVA_WORKDIR", __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "work"))
    check(pd.read_parquet(f"{SP}/audit2_samples.parquet"))
    print("labels OK:", len(LAB_ASSISTANT), "assistant,", len(LAB_SEARCH), "search")
''')
print("A:", sum(len(v) for v in A.values()), "C:", sum(len(v) for v in C.values()), "B:", 1006 - len(ids))
