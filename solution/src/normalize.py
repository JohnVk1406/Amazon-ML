import re
import unicodedata

LEGAL_SUFFIX = {
    'corp','corporation','inc','incorporated','llc','ltd','limited',
    'pvt','private','llp','pllc','co','company','enterprises','enterprise',
    'group','holdings','services','solutions','associates','partners'
}

ADDR_ABBR = {
    'rd':'road','st':'street','ave':'avenue','av':'avenue','blvd':'boulevard',
    'dr':'drive','ln':'lane','ct':'court','pl':'place','trl':'trail',
    'pkwy':'parkway','hwy':'highway','ste':'suite','apt':'apartment',
    'tn':'tamil nadu','tn.':'tamil nadu'
}

def basic_clean(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    if s.lower() == "nan" or s.strip().lower() == "null":
        return ""
    # unicode normalize, keep letters/numbers from any script
    s = unicodedata.normalize("NFKC", s)
    s = s.lower()
    s = s.replace("&", " and ")
    # split domain dots: foo.com -> foo com
    s = re.sub(r"\.(?=[a-z])", " ", s)
    # keep alnum + spaces (unicode aware)
    s = re.sub(r"[^\w\s]", " ", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def normalize_name(s: str) -> str:
    s = basic_clean(s)
    toks = s.split()
    # expand? remove legal suffix for blocking key but keep full for scoring
    # return full cleaned; caller can also get stripped version
    return s

def normalize_address(s: str) -> str:
    s = basic_clean(s)
    toks = []
    for t in s.split():
        t = ADDR_ABBR.get(t, t)
        if t == "null":
            continue
        toks.append(t)
    return " ".join(toks)

def combined_text(name: str, addr: str) -> str:
    n = normalize_name(name)
    a = normalize_address(addr)
    if a:
        return f"{n} {a}"
    return n

def name_no_suffix(name: str) -> str:
    n = normalize_name(name)
    toks = [t for t in n.split() if t not in LEGAL_SUFFIX]
    return " ".join(toks)
