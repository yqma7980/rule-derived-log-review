"""Small generated example. No private measurements or fitted forest required."""
import json
import numpy as np
from external_core import rule, interpolate, partition
from review_priority import evaluate_blocks, self_test

z=np.arange(200)*.152
gr=55+33*np.sin(z*2.5)
den=2.32+.16*np.cos(z*.9)
fine=rule(gr,den,z,.152)[0]
coarse=rule(interpolate(gr,z,5,0),interpolate(den,z,5,0),z,.152)[0]
# An explicitly constructed comparator exercises the accounting, not model skill.
comparator=coarse.copy(); comparator[::17]=(comparator[::17]+1)%6
parts=partition(fine,coarse,comparator)
assert np.all(parts.sum(axis=0)==1)
starts=np.arange(0,len(z),20); sizes=np.minimum(20,len(z)-starts)
scores=np.linspace(1,0,len(starts))
review=evaluate_blocks(scores,starts,sizes,fine!=coarse,.1)
print(json.dumps({'kind':'generated demonstration, not manuscript evidence',
                  'rows':len(z),'paired_counts':dict(zip('ABCDE',parts.sum(axis=1).tolist())),
                  'review':review,'tests':self_test()},indent=2))
