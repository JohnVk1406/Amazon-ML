import re
from rapidfuzz import fuzz
from .normalize import basic_clean, normalize_name, normalize_address, name_no_suffix

def jaccard(a: str, b: str) -> float:
    sa = set(a.split()); sb = set(b.split())
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)

def feats_for_pair(s1_name, s1_addr, s1_country, s2_name, s2_addr, s2_country, tfidf_score=None):
    n1 = normalize_name(s1_name or "")
    a1 = normalize_address(s1_addr or "")
    n2 = normalize_name(s2_name or "")
    a2 = normalize_address(s2_addr or "")
    n1s = " ".join([t for t in n1.split() if t not in
        {'corp','corporation','inc','incorporated','llc','ltd','limited','pvt','private','llp','pllc','co','company','enterprises','enterprise','group','holdings','services','solutions'}])
    n2s = " ".join([t for t in n2.split() if t not in
        {'corp','corporation','inc','incorporated','llc','ltd','limited','pvt','private','llp','pllc','co','company','enterprises','enterprise','group','holdings','services','solutions'}])
    # rapidfuzz (0-100 -> /100)
    try:
        nr = fuzz.token_set_ratio(n1, n2) / 100.0
    except: nr = 0.0
    try:
        nr2 = fuzz.WRatio(n1, n2) / 100.0
    except: nr2 = 0.0
    try:
        nsr = fuzz.token_set_ratio(n1s, n2s) / 100.0 if (n1s and n2s) else 0.0
    except: nsr = 0.0
    try:
        ar = fuzz.token_set_ratio(a1, a2) / 100.0 if (a1 and a2) else 0.0
    except: ar = 0.0
    try:
        ar2 = fuzz.WRatio(a1, a2) / 100.0 if (a1 and a2) else 0.0
    except: ar2 = 0.0
    jn = jaccard(n1, n2)
    ja = jaccard(a1, a2) if (a1 and a2) else 0.0
    jns = jaccard(n1s, n2s) if (n1s and n2s) else 0.0
    cm = 1.0 if (s1_country or "") == (s2_country or "") else 0.0
    # length features
    lnd = abs(len(n1) - len(n2)) / max(1, max(len(n1), len(n2)))
    lad = abs(len(a1) - len(a2)) / max(1, max(len(a1), len(a2)))
    empty_a = 1.0 if (a1 == "" or a2 == "") else 0.0
    tf = float(tfidf_score) if tfidf_score is not None else 0.0
    return [nr, nr2, nsr, ar, ar2, jn, ja, jns, cm, lnd, lad, empty_a, tf]

FEAT_NAMES = ["name_set","name_wr","name_nosuffix_set","addr_set","addr_wr",
              "j_name","j_addr","j_name_nosuf","country_match","len_name_diff","len_addr_diff","empty_addr","tfidf"]
