import re
import unicodedata

LEGAL_SUFFIX = {
    'corp', 'corporation', 'inc', 'incorporated', 'llc', 'ltd', 'limited',
    'pvt', 'private', 'llp', 'pllc', 'co', 'company', 'enterprises', 'enterprise',
    'group', 'holdings', 'services', 'solutions', 'associates', 'partners'
}

ADDR_ABBR = {
    'rd': 'road', 'st': 'street', 'ave': 'avenue', 'av': 'avenue', 'blvd': 'boulevard',
    'dr': 'drive', 'ln': 'lane', 'ct': 'court', 'pl': 'place', 'trl': 'trail',
    'pkwy': 'parkway', 'hwy': 'highway', 'ste': 'suite', 'apt': 'apartment',
    'tn': 'tamil nadu', 'tn.': 'tamil nadu'
}


def basic_clean(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    if s.lower() == "nan" or s.strip().lower() == "null":
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = s.lower()
    s = s.replace("&", " and ")
    s = re.sub(r"\.(?=[a-z])", " ", s)
    s = re.sub(r"[^\w\s]", " ", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_name(s: str) -> str:
    return basic_clean(s)


def normalize_address(s: str) -> str:
    s = basic_clean(s)
    toks = []
    for token in s.split():
        token = ADDR_ABBR.get(token, token)
        if token == "null":
            continue
        toks.append(token)
    return " ".join(toks)


def combined_text(name: str, addr: str) -> str:
    n = normalize_name(name)
    a = normalize_address(addr)
    if a:
        return f"{n} {a}"
    return n


def name_no_suffix(name: str) -> str:
    n = normalize_name(name)
    toks = [token for token in n.split() if token not in LEGAL_SUFFIX]
    return " ".join(toks)
