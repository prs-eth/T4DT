import tntorch as tn

def reduce_tucker(ts, eps, rmax, algorithm):
    d = dict()
    for i, elem in enumerate(ts):
        climb = 0  # For going up the tree
        while climb in d:
            elem = tn.round_tucker(tn.cat([d[climb], elem], dim=-1), eps=eps, rmax=rmax, algorithm=algorithm)
            d.pop(climb)
            climb += 1
        d[climb] = elem
    keys = list(d.keys())
    result = d[keys[0]]
    for key in keys[1:]:
        result = tn.round_tucker(tn.cat([result, d[key]], dim=-1), eps=eps, rmax=rmax, algorithm=algorithm)
    return result
