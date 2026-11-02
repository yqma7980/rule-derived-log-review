"""Optional read-only NPZ value comparison; never used to generate results."""
import argparse
import json
from pathlib import Path
import numpy as np


def compare(left, right):
    report = {'left':Path(left).name,'right':Path(right).name,'arrays':{},'equal':True}
    with np.load(left,allow_pickle=False) as a, np.load(right,allow_pickle=False) as b:
        if set(a.files) != set(b.files):
            return dict(report,equal=False,error='array keys differ')
        for name in sorted(a.files):
            x,y = a[name],b[name]
            same_shape = x.shape == y.shape
            numeric = np.issubdtype(x.dtype,np.number) and np.issubdtype(y.dtype,np.number)
            equal = same_shape and (np.array_equal(x,y,equal_nan=True) if numeric else np.array_equal(x,y))
            item = {'shape_equal':same_shape,'dtype_equal':x.dtype==y.dtype,'values_equal':equal}
            if same_shape and numeric:
                mask = (x == y) | (np.isnan(x) & np.isnan(y))
                item['different_elements'] = int(np.count_nonzero(~mask))
                finite = np.isfinite(x) & np.isfinite(y)
                item['max_abs_finite_difference'] = float(np.max(np.abs(x[finite].astype(float)-y[finite].astype(float)))) if np.any(finite) else None
            report['arrays'][name] = item
            report['equal'] &= bool(equal and item['dtype_equal'])
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('left',type=Path)
    p.add_argument('right',type=Path)
    a = p.parse_args()
    r = compare(a.left,a.right)
    print(json.dumps(r,indent=2))
    raise SystemExit(0 if r['equal'] else 1)
