"""Depth-aware operational rule and diagnostic accounting, not geological truth."""
import numpy as np
import pandas as pd

def rule(gr, den, depth, native_step):
    gr, den, z = map(lambda a: np.asarray(a, float), (gr, den, depth))
    if not len(z) or not np.isfinite([gr,den,z]).all() or np.any(np.diff(z)<=0):
        raise ValueError("Finite ordered segment required")
    lith = np.select([(den<1.75)&(gr<70), (gr>=80)&(den>=2.30),
                      (gr>=60)&(gr<80), (gr>=40)&(gr<60),
                      (gr>=20)&(gr<40), gr<20], [0,1,2,3,4,5], default=1)
    n = len(z); i = np.arange(n)
    starts = np.r_[0, np.flatnonzero(np.diff(lith)!=0)+1]
    stops = np.r_[starts[1:], n]
    thickness = np.repeat(z[stops-1]-z[starts]+native_step, stops-starts)
    def count(mask, left, right):
        cs = np.r_[0,np.cumsum(mask)]
        return cs[right]-cs[left]
    # Half-open heterogeneity range matches original [-10,+10) indexing at 0.05 m.
    left = np.searchsorted(z,z-.50-1e-8,side="left")
    right = np.searchsorted(z,z+.50-1e-8,side="left")
    hetero = (sum(count(lith==k,left,right)>0 for k in range(6))>=3)
    hetero &= (count(lith==2,left,right)>0)&(count(lith==3,left,right)>0)
    deeper = count(np.isin(lith,[1,2]),i+1,np.searchsorted(z,z+.25+1e-8,side="right"))>0
    shallow = count(np.isin(lith,[4,5]),np.searchsorted(z,z-.25-1e-8,side="left"),i)>0
    context = (lith==3)&(thickness<2.-1e-8)&hetero&~(deeper&shallow)
    base = np.select([lith==0,lith==1,lith==2,np.isin(lith,[4,5])],
                     [0,np.where(gr>90,5,4),3,1],default=2).astype(np.int8)
    final = np.where(context,3,base).astype(np.int8)
    starts = np.r_[0,np.flatnonzero(np.diff(final)!=0)+1]; stops=np.r_[starts[1:],n]
    runs = np.repeat(z[stops-1]-z[starts]+native_step, stops-starts)
    return final,base,runs

def partition(f,c,p):
    same=f==c
    masks=[same&(p==f),same&(p!=f),~same&(p==c),~same&(p==f),~same&(p!=c)&(p!=f)]
    assert np.all(np.sum(masks,axis=0)==1)
    return np.asarray(masks)

def interpolate(x,z,stride,phase):
    if stride==1:return x.copy()
    a=np.unique(np.r_[0,np.arange(phase,len(z),stride),len(z)-1])
    return np.interp(z,z[a],x[a])

def features(gr,den,z,step):
    values=[gr,den]
    spans=[]
    for span in (.20,.50,1.):
        half=max(1,int(round(span/(2*step))))
        spans.append(2*half*step)
        i=np.arange(len(z)); l=np.maximum(0,i-half); r=np.minimum(len(z)-1,i+half)
        for a in (gr,den):
            rolling=pd.Series(a).rolling(2*half+1,center=True,min_periods=1)
            slope=np.divide(a[r]-a[l],z[r]-z[l],out=np.zeros(len(z)),where=z[r]!=z[l])
            values += [rolling.mean().to_numpy(),rolling.std(ddof=0).fillna(0).to_numpy(),
                       (rolling.max()-rolling.min()).to_numpy(),slope]
    X=np.asarray(values,dtype=np.float32).T
    assert X.shape==(len(z),26) and np.isfinite(X).all()
    return X,spans

def bootstrap_mean(x,seed=20261031):
    x=np.asarray(x,float); x=x[np.isfinite(x)]
    if not len(x):return [np.nan,np.nan]
    rng=np.random.default_rng(seed)
    means=x[rng.integers(0,len(x),size=(2000,len(x)))].mean(axis=1)
    return np.quantile(means,[.025,.975]).tolist()
