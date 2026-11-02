"""Fixed-length review selection. Targets are used for evaluation, never ranking."""
import hashlib
import numpy as np


def stable_keys(well, count, seed=20260907):
    return np.array([int.from_bytes(hashlib.sha256(
        f'{seed}|{well}|{i}'.encode()).digest()[:8], 'big')
        for i in range(count)], dtype=np.uint64)


def evaluate_blocks(scores, starts, sizes, target, budget, well='synthetic'):
    scores=np.asarray(scores,float); starts=np.asarray(starts,int)
    sizes=np.asarray(sizes,int); target=np.asarray(target,bool)
    if not 0<=budget<=1 or not np.isfinite(scores).all():
        raise ValueError('Finite scores and a budget between zero and one required')
    if not (len(scores)==len(starts)==len(sizes)) or np.any(sizes<=0):
        raise ValueError('One positive size and start per block required')
    cover=np.zeros(len(target),int)
    for s,n in zip(starts,sizes):
        if s<0 or s+n>len(target): raise ValueError('Block outside input range')
        cover[s:s+n]+=1
    if not np.all(cover==1): raise ValueError('Blocks must partition the input')
    order=np.lexsort((stable_keys(well,len(scores)),-scores))
    need=int(np.floor(budget*len(target))); remaining=need
    selected=np.zeros(len(target),bool)
    for b in order:
        take=min(remaining,int(sizes[b]))
        selected[starts[b]:starts[b]+take]=True
        remaining-=take
        if remaining==0: break
    assert selected.sum()==need
    captured=int((selected&target).sum()); total=int(target.sum())
    return {'budget_rows':need,'reviewed_rows':int(selected.sum()),
            'target_rows':total,'captured_rows':captured,
            'capture_rate':captured/total if total else None,
            'review_yield':captured/need if need else None}


def self_test():
    a=evaluate_blocks([2,1],[0,5],[5,5],[1,0,1,0,1,1,0,0,0,0],.3)
    assert a['reviewed_rows']==3 and a['captured_rows']==2
    assert a['capture_rate']==.5 and a['review_yield']==2/3
    b=evaluate_blocks([2,1],[0,5],[5,5],[0]*10,.3)
    assert b['capture_rate'] is None and b['review_yield']==0
    c=evaluate_blocks([2,1],[0,5],[5,5],[1]*10,0)
    assert c['reviewed_rows']==0 and c['capture_rate']==0 and c['review_yield'] is None
    d=evaluate_blocks([1,1],[0,5],[5,5],[1]*10,1)
    assert d['reviewed_rows']==10 and d['capture_rate']==1
    try: evaluate_blocks([1,1],[0,4],[5,6],[0]*10,.3)
    except ValueError: pass
    else: raise AssertionError('Overlapping blocks accepted')
    return {'fixed_budget_denominator_tests':5}


if __name__=='__main__': print(self_test())
