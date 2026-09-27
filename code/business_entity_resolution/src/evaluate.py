def f05_macro(pred_dict, true_dict):
    """pred_dict/true_dict: s1 -> list/iterable of entity_ids."""
    fsum = 0.0
    psum = 0.0
    rsum = 0.0
    n = len(true_dict)
    for s1, true_ids in true_dict.items():
        pred_ids = pred_dict.get(s1, [])
        tset = set(true_ids) if true_ids else set()
        pset = set(pred_ids) if pred_ids else set()
        if len(tset) == 0 and len(pset) == 0:
            f = 1.0
            p = 1.0
            r = 1.0
        elif len(tset) == 0 or len(pset) == 0:
            f = 0.0
            p = 0.0 if len(pset) > 0 else 1.0
            r = 0.0 if len(tset) > 0 else 1.0
            if len(tset) == 0 and len(pset) > 0:
                p = 0.0
                r = 0.0
            if len(pset) == 0 and len(tset) > 0:
                p = 0.0
                r = 0.0
        else:
            tp = len(tset & pset)
            p = tp / len(pset) if pset else 0.0
            r = tp / len(tset) if tset else 0.0
            if p + r == 0:
                f = 0.0
            else:
                f = (1.25 * p * r) / (0.25 * p + r)
        fsum += f
        psum += p
        rsum += r
    return {"F05": fsum / n, "P": psum / n, "R": rsum / n, "n": n}


def blocking_recall(cand_dict, true_dict):
    """Recall ceiling: fraction of true pairs covered by candidates."""
    tp_total = 0
    gt_total = 0
    rec_sum = 0.0
    n = 0
    for s1, true_ids in true_dict.items():
        tset = set(true_ids) if true_ids else set()
        if not tset:
            continue
        cset = set(cand_dict.get(s1, []))
        tp = len(tset & cset)
        tp_total += tp
        gt_total += len(tset)
        rec_sum += tp / len(tset)
        n += 1
    return {
        "micro_recall": tp_total / max(1, gt_total),
        "macro_recall": rec_sum / max(1, n),
        "avg_cands": sum(len(v) for v in cand_dict.values()) / max(1, len(cand_dict)),
    }
